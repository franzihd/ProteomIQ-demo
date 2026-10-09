"""
Retrieval within one sample (Stage 1/2) and interpretation with Claude (Stage 3).

Usage:
    uv run python src/model/interpret.py \\
        --query "receptor tyrosine kinase signaling" \\
        --patients 11LU013 --cohort luad
"""
import re
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pandas as pd
import anthropic

from src.model.retrieve_and_annotate import (
    build_gene_lookup,
    load_esm2_lookup,
    load_model,
    encode_query_vector,
)
from src.preprocessing.normalize_uploaded_sample import normalize_uploaded_sample, build_gene_to_ensp_lookup

MODEL = "claude-sonnet-5"
DEFAULT_TOP_K = 20

PROMPT_TEMPLATE = (
    "You are helping a researcher explore a proteomics sample. Below is a list "
    "of proteins retrieved from this specific patient's own measured sample as "
    "semantically relevant to their query, together with each protein's actual "
    "measured abundance percentile (0-100, higher = more highly expressed "
    "relative to everything else measured in this sample) and its UniProt "
    "functional annotation.\n\n"
    "Researcher's query: \"{query}\"\n\n"
    "Retrieved proteins (sorted by relevance to the query):\n"
    "{protein_list}\n\n"
    "Write a short (2-4 sentence) natural-language interpretation of what "
    "this pattern might suggest biologically. Rules:\n"
    "- Use ONLY the facts given above. Do not invent additional proteins, "
    "functions, or clinical facts not listed.\n"
    "- Refer to proteins by their gene name.\n"
    "- Frame this as hypothesis generation for further investigation, not a "
    "clinical diagnosis -- do not claim certainty about disease state or "
    "make treatment recommendations.\n"
    "- If the retrieved proteins don't obviously relate to the query, say so "
    "honestly rather than forcing a connection.\n\n"
    "Respond with only the interpretation, nothing else."
)


def build_annotation_lookup():
    """ENSP_ID -> unmasked UniProt annotation (gene name included). Unlike the
    training-time text (which masks the gene name to avoid the name-shortcut),
    Stage 3's output is read by a human, not used as a
    contrastive-training target -- so gene identity should stay in the text."""
    df = pd.read_csv("data/full_proteome/protein_annotations.csv",
                      usecols=["ENSP_ID", "Annotation"])
    return dict(zip(df["ENSP_ID"], df["Annotation"]))


def get_patient_measured_proteins(patient_id: str, cohort: str) -> dict[str, float]:
    """ENSP_ID -> abundance percentile for every protein actually measured
    (non-NaN) in one real CPTAC patient."""
    abundance_df = pd.read_csv(f"data/cptac/{cohort}/rank_normalized_proteomics.csv",
                                header=[0, 1], index_col=0)
    if patient_id not in abundance_df.index:
        raise ValueError(f"Patient {patient_id} not found in data/cptac/{cohort}/rank_normalized_proteomics.csv")
    patient_row = abundance_df.loc[patient_id]
    return {ensp_id: float(val) for (_, ensp_id), val in patient_row.items() if not pd.isna(val)}


def get_uploaded_measured_proteins(uploaded_csv: str, gene_col: str, abundance_col: str) -> dict[str, float]:
    """Same job as get_patient_measured_proteins, but for a researcher's own
    freshly uploaded sample. normalize_uploaded_sample.py
    handles rank-normalization and gene -> ENSP_ID matching."""
    result_df, unmatched, n_ambiguous = normalize_uploaded_sample(uploaded_csv, gene_col, abundance_col)
    print(f"  Uploaded sample: {len(result_df)} proteins usable "
          f"({n_ambiguous} dropped as ambiguous, {len(unmatched)} not in our reference set)")
    return dict(zip(result_df["ENSP_ID"], result_df["percentile"]))


def retrieve_within_sample(query: str, measured_proteins: dict[str, float], esm2_lookup,
                            protein_head, text_head, tokenizer, biobert_model, device):
    """Rank ONLY the proteins actually measured in this sample, by cosine
    similarity to the query (semantic relevance) alone -- abundance
    percentile is attached per candidate as context, not used to rank or
    filter (see this file's module docstring for why the earlier
    activity_score blend was rejected). Returns a list of
    (protein_id, cos_sim, percentile), sorted by cos_sim descending."""
    protein_ids = [pid for pid in measured_proteins if pid in esm2_lookup]
    esm2_matrix = torch.stack([esm2_lookup[pid] for pid in protein_ids]).to(device)

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix)
        query_vec = encode_query_vector(query, tokenizer, biobert_model, text_head, device)
        cos_sims = (protein_vecs @ query_vec.T).squeeze(-1)

    scored = [
        (pid, cos_sim, measured_proteins[pid])
        for pid, cos_sim in zip(protein_ids, cos_sims.tolist())
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def build_prompt(query, selected, gene_lookup, annotation_lookup):
    lines = []
    for protein_id, cos_sim, percentile in selected:
        gene = gene_lookup.get(protein_id, protein_id)
        annotation = annotation_lookup.get(protein_id, "(no annotation available)")
        lines.append(
            f"- {gene}: {percentile * 100:.0f}th percentile abundance, "
            f"relevance score {cos_sim:.3f}. Function: {annotation}"
        )
    protein_list = "\n".join(lines)
    return PROMPT_TEMPLATE.format(query=query, protein_list=protein_list)


def stage3_interpret(client, query, scored_candidates, gene_lookup, annotation_lookup, top_k):
    selected = scored_candidates[:top_k]
    if not selected:
        return "No proteins in this sample matched anything usable for interpretation."

    prompt = build_prompt(query, selected, gene_lookup, annotation_lookup)
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip()


# --- Multi-turn chat variant, used by the Streamlit app's Step 4 -----------
# Same underlying data and guardrails as stage3_interpret above, but the
# protein list + rules live in a `system` prompt (sent once, persists for the
# whole conversation) instead of a one-shot `user` prompt asking for a fixed
# 2-4 sentence summary -- so a researcher can ask natural follow-up questions
# ("why do these look more like RNA-binding than DNA-binding proteins?")
# against the SAME retrieved result set, without a new retrieval. Retrieval
# (Stage 1/2) is unchanged -- only this interpretation layer becomes
# conversational.
# Unlike a single fixed query, the researcher can run MORE than one search
# over the course of one conversation (see build_chat_system_prompt below --
# app.py's Step 4 gained an in-chat search box for exactly this, so a
# follow-up that needs different data doesn't have to restart the whole
# chat). Every search run so far gets its own labeled section here, kept
# distinct rather than merged into one undifferentiated list, so Claude (and
# the researcher, reading its answers) can tell which search a given protein
# came from.
CHAT_SYSTEM_PROMPT_HEADER = (
    "You are helping a researcher explore a proteomics sample through natural-language "
    "conversation. The researcher can run more than one semantic search query against "
    "this sample's measured proteins over the course of the conversation -- below are all "
    "searches run so far. Each result shows a protein's abundance percentile in this "
    "specific sample (0-100, higher = more highly expressed relative to everything else "
    "measured in this sample) and its UniProt functional annotation.\n"
)

CHAT_SYSTEM_PROMPT_SEARCH_BLOCK = (
    "\nSearch: \"{query}\"\n"
    "Retrieved proteins (sorted by relevance to this search):\n"
    "{protein_list}\n"
)

CHAT_SYSTEM_PROMPT_RULES = (
    "\nRules for this whole conversation:\n"
    "- Use ONLY the facts given above, across all searches. Do not invent additional "
    "proteins, functions, or clinical facts not listed. If the researcher asks about "
    "something this data doesn't cover, say so honestly rather than guessing.\n"
    "- Refer to proteins by their gene name.\n"
    "- If more than one search has been run, make clear which search a protein came from "
    "whenever that's relevant to the discussion.\n"
    "- Frame everything as hypothesis generation for further investigation, not a "
    "clinical diagnosis -- never claim certainty about disease state or make treatment "
    "recommendations.\n"
    "- If the retrieved proteins don't obviously relate to a query, say so honestly "
    "rather than forcing a connection -- including when a follow-up question points this "
    "out.\n"
    "- Keep replies conversational and reasonably concise."
)


def _format_protein_list(selected, gene_lookup, annotation_lookup):
    lines = []
    for protein_id, cos_sim, percentile in selected:
        gene = gene_lookup.get(protein_id, protein_id)
        annotation = annotation_lookup.get(protein_id, "(no annotation available)")
        lines.append(
            f"- {gene}: {percentile * 100:.0f}th percentile abundance, "
            f"relevance score {cos_sim:.3f}. Function: {annotation}"
        )
    return "\n".join(lines)


# --- Explicit, named-protein lookups -----------------------------------
# Distinct from search: a search is a new semantic-relevance query (Stage 1
# re-run); a lookup is checking one SPECIFIC gene the researcher already
# named in their own message (e.g. "Could this patient be resistant to EGFR
# inhibitors?" -- EGFR wasn't in the retrieved set, so Claude had no data on
# it and correctly said so). This is deterministic app-level string
# matching, not an LLM decision to go explore something -- the researcher
# already named the protein themselves, so looking it up is directly
# responsive to what was asked, not the tool choosing a new direction on its
# own (same "researcher drives every piece of context" principle as the
# in-chat search box).
GENE_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")


def extract_gene_mentions(text: str, gene_to_ensp: dict) -> list[str]:
    """Gene symbols the researcher typed as their own ALL-CAPS token (e.g.
    'EGFR', 'KRAS') and that exist in our reference set. Requiring the token
    to already be uppercase in the researcher's own text (not case-folded)
    is a deliberate false-positive guard -- several gene symbols collide
    with ordinary lowercase English words (MET, SET, FUS...), and people
    writing about genes conventionally capitalize them anyway."""
    candidates = set(GENE_TOKEN_PATTERN.findall(text))
    return [g for g in candidates if g in gene_to_ensp]


def lookup_gene(gene: str, gene_to_ensp: dict, measured_proteins: dict, annotation_lookup: dict) -> dict:
    """Whether one specific gene was actually measured in this sample --
    'not measured' is itself a real, useful answer (it's exactly what
    resolved the EGFR-inhibitor-resistance question honestly), not a
    failure case to hide."""
    ensp_id = gene_to_ensp.get(gene)
    if ensp_id is None:
        return {"gene": gene, "status": "unknown"}
    if ensp_id not in measured_proteins:
        return {"gene": gene, "status": "not_measured"}
    return {
        "gene": gene,
        "status": "measured",
        "percentile": measured_proteins[ensp_id],
        "annotation": annotation_lookup.get(ensp_id, "(no annotation available)"),
    }


def _format_lookups_block(lookups: dict) -> str:
    if not lookups:
        return ""
    lines = [
        "\nThe researcher has also directly asked about specific proteins by name during "
        "this conversation. Here is whether each was actually measured in this sample -- "
        "use these facts directly; do not say you lack data on a protein listed here:"
    ]
    for gene, info in lookups.items():
        if info["status"] == "measured":
            lines.append(
                f"- {gene}: measured in this sample, {info['percentile'] * 100:.0f}th "
                f"percentile abundance. Function: {info['annotation']}"
            )
        elif info["status"] == "not_measured":
            lines.append(f"- {gene}: in our reference protein set, but NOT measured/detected "
                          "in this specific sample (no abundance value available).")
        else:
            lines.append(f"- {gene}: not in our reference protein set at all -- no data available.")
    return "\n".join(lines) + "\n"


def build_chat_system_prompt(searches: list[dict], gene_lookup, annotation_lookup, lookups: dict | None = None):
    """searches: [{'query': str, 'selected': [(protein_id, cos_sim, percentile), ...]}, ...]
    -- one entry per retrieval run so far in this chat session. lookups:
    {gene: {...}} from lookup_gene, one entry per specifically-named protein
    the researcher has asked about. Rebuilt fresh from both every time either
    changes; the system prompt is resent in full on every API call anyway
    (the Anthropic API is stateless per call), so there's no separate
    "update" step to keep in sync."""
    blocks = [CHAT_SYSTEM_PROMPT_HEADER]
    for s in searches:
        protein_list = _format_protein_list(s["selected"], gene_lookup, annotation_lookup)
        blocks.append(CHAT_SYSTEM_PROMPT_SEARCH_BLOCK.format(query=s["query"], protein_list=protein_list))
    lookups_block = _format_lookups_block(lookups or {})
    if lookups_block:
        blocks.append(lookups_block)
    blocks.append(CHAT_SYSTEM_PROMPT_RULES)
    return "\n".join(blocks)


def stage3_chat(client, system_prompt: str, messages: list[dict]):
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...], strictly
    alternating and starting with 'user' (Anthropic API requirement) -- the
    system prompt is passed separately, not as part of this list. Returns the
    assistant's reply text; append it to `messages` yourself to continue the
    conversation on the next call."""
    response = client.messages.create(
        model=MODEL,
        # Higher than stage3_interpret's 600 -- that function's prompt asks
        # for a fixed 2-4 sentence summary, but this system prompt has no
        # such length constraint (a real conversation shouldn't be forced
        # short), and 600 was observed cutting real replies off mid-sentence.
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--patients", nargs="+", default=[],
                         help="existing CPTAC patient IDs, looked up in --cohort's abundance matrix")
    parser.add_argument("--cohort", default="luad")
    parser.add_argument("--uploaded-csv", default=None,
                         help="path to a researcher's own uploaded sample (gene name + raw abundance)")
    parser.add_argument("--gene-col", default="gene", help="only used with --uploaded-csv")
    parser.add_argument("--abundance-col", default="abundance", help="only used with --uploaded-csv")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                         help="how many top-cosine-similarity proteins to display and send to the LLM")
    args = parser.parse_args()

    if not args.patients and not args.uploaded_csv:
        parser.error("provide at least one of --patients or --uploaded-csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_lookup = build_gene_lookup()
    annotation_lookup = build_annotation_lookup()
    esm2_lookup = load_esm2_lookup()
    protein_head, text_head, tokenizer, biobert_model = load_model(device)
    client = anthropic.Anthropic()

    print(f"Query: \"{args.query}\"")

    samples = {}
    for patient_id in args.patients:
        samples[patient_id] = get_patient_measured_proteins(patient_id, args.cohort)
    if args.uploaded_csv:
        samples["uploaded"] = get_uploaded_measured_proteins(args.uploaded_csv, args.gene_col, args.abundance_col)

    for label, measured in samples.items():
        print(f"\n{'=' * 88}\nSample: {label}  ({len(measured)} proteins measured)\n{'=' * 88}")

        scored = retrieve_within_sample(
            args.query, measured, esm2_lookup, protein_head, text_head, tokenizer, biobert_model, device
        )

        print(f"\nTop {args.top_k} by cosine similarity (relevance to query):")
        for pid, cos_sim, pct in scored[:args.top_k]:
            gene = gene_lookup.get(pid, pid)
            print(f"  {gene:<15} cos_sim={cos_sim:6.3f}  percentile={pct * 100:5.1f}")

        text = stage3_interpret(client, args.query, scored, gene_lookup, annotation_lookup, top_k=args.top_k)
        print("\nInterpretation:\n" + text)


if __name__ == "__main__":
    main()
