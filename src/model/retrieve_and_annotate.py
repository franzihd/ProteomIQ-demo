"""
Stage 1 retrieval over the whole proteome + Stage 2 abundance annotation per patient.

Usage:
    uv run python src/model/retrieve_and_annotate.py \\
        --query "receptor tyrosine kinase signaling" \\
        --patients 11LU013 11LU016 --cohort luad --top-n 50
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import h5py
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from src.embeddings.textual.embed_biobert import MAX_LENGTH, mean_pool
from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.preprocessing.normalize_uploaded_sample import normalize_uploaded_sample

CHECKPOINT_PATH = "data/runs/full_proteome/multiview_experiment/multiview_best.pt"
BIOBERT_FILENAME = "biobert_embeddings_masked.h5"
# HuggingFace model ID, not a machine-specific resolved cache path -- resolves
# via transformers' own cache lookup (finds it locally if already downloaded,
# fetches it otherwise) so this works on any machine, not just the one it was
# first hardcoded on.
BIOBERT_PATH = "dmis-lab/biobert-base-cased-v1.2"


def build_gene_lookup():
    """ENSP_ID -> Gene name, from the unified full-proteome annotation file
    (supersedes merging per-cohort protein_annotations.csv files)."""
    df = pd.read_csv("data/full_proteome/protein_annotations.csv", usecols=["ENSP_ID", "Gene"])
    return dict(zip(df["ENSP_ID"], df["Gene"]))


def load_esm2_lookup():
    """ENSP_ID -> 640-dim ESM2 vector, for every protein in the full human
    proteome (data/full_proteome/esm2_embeddings.h5). Shared by any retrieval
    path that needs a protein's sequence embedding by ID, whether ranking
    against the whole proteome (stage1_retrieve) or a restricted subset
    (interpret.py's sample-restricted retrieval)."""
    with h5py.File("data/full_proteome/esm2_embeddings.h5", "r") as h5f:
        matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
        ids = [pid.decode() for pid in h5f["protein_ids"][:]]
    return dict(zip(ids, matrix))


def load_model(device):
    """Load the trained projection heads (checkpoint) + frozen BioBERT for
    live query encoding. Shared by any script that needs to embed proteins/
    text with the current main-line model."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 512)
    output_dim = ckpt.get("output_dim", 256)
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    protein_head.load_state_dict(ckpt["protein_head"])
    text_head.load_state_dict(ckpt["text_head"])
    protein_head.eval()
    text_head.eval()

    # local_files_only=False (the default) lets transformers download BioBERT
    # on first run on a machine that doesn't have it cached yet; it's reused
    # from the local cache on every run after that, same as any other
    # HuggingFace model.
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH)
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH).to(device)
    biobert_model.eval()
    return protein_head, text_head, tokenizer, biobert_model


def encode_query_vector(query: str, tokenizer, biobert_model, text_head, device):
    """Free text -> BioBERT (mean-pooled) -> text_head -> shared-space vector."""
    with torch.no_grad():
        enc = tokenizer(query, padding=True, truncation=True,
                         max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = biobert_model(**enc)
        query_vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
        query_vec = text_head(query_vec)
    return query_vec


def stage1_retrieve(query: str, top_n: int, device):
    """Wide, patient-invariant retrieval over the WHOLE human proteome
    (~16,893 proteins) -- built for fair pathway-ground-truth spot-checking
, where restricting to one sample's measured proteins
    would be invalid (see this file's docstring). For the interactive,
    sample-restricted retrieval used by the researcher-facing tool, see
    interpret.py's retrieve_within_sample instead."""
    dataset = ProteomicsDataset(cohorts=[], use_abundance=False, biobert_filename=BIOBERT_FILENAME)
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])

    protein_head, text_head, tokenizer, biobert_model = load_model(device)

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))
        query_vec = encode_query_vector(query, tokenizer, biobert_model, text_head, device)
        sims = (protein_vecs @ query_vec.T).squeeze()
        ranked_idx = torch.argsort(sims, descending=True)[:top_n]

    candidates = [(protein_ids[i], sims[i].item()) for i in ranked_idx.tolist()]
    return candidates


def get_kegg_members(query: str):
    """If `query` is a literal KEGG pathway name, return its member gene set.
    Otherwise return None (query is free text, no ground truth available)."""
    import gseapy
    gene_sets = gseapy.get_library("KEGG_2021_Human")
    return set(gene_sets[query]) if query in gene_sets else None


def stage2_annotate(candidates, patient_id: str, cohort: str):
    abundance_df = pd.read_csv(f"data/cptac/{cohort}/rank_normalized_proteomics.csv",
                                header=[0, 1], index_col=0)
    protein_id_to_col = {ensp_id: (gene, ensp_id) for gene, ensp_id in abundance_df.columns}

    if patient_id not in abundance_df.index:
        raise ValueError(f"Patient {patient_id} not found in data/cptac/{cohort}/rank_normalized_proteomics.csv")

    patient_row = abundance_df.loc[patient_id]

    annotated = []
    for protein_id, cos_sim in candidates:
        col = protein_id_to_col.get(protein_id)
        if col is None:
            percentile = None  # protein not present in this cohort's abundance matrix at all
        else:
            val = patient_row[col]
            percentile = None if pd.isna(val) else float(val)
        annotated.append((protein_id, cos_sim, percentile))
    return annotated


def stage2_annotate_uploaded(candidates, uploaded_csv: str, gene_col: str, abundance_col: str):
    """Same job as stage2_annotate, but for a researcher's own freshly uploaded
    sample instead of an existing CPTAC patient. normalize_uploaded_sample.py
 does the rank-normalization and gene->ENSP_ID
    matching; here we just look up each Stage 1 candidate's percentile in that
    already-normalized result, same shape as stage2_annotate's per-patient row."""
    result_df, unmatched, n_ambiguous = normalize_uploaded_sample(uploaded_csv, gene_col, abundance_col)
    print(f"  Uploaded sample: {len(result_df)} proteins usable "
          f"({n_ambiguous} dropped as ambiguous, {len(unmatched)} not in our reference set)")

    ensp_to_percentile = dict(zip(result_df["ENSP_ID"], result_df["percentile"]))

    annotated = []
    for protein_id, cos_sim in candidates:
        percentile = ensp_to_percentile.get(protein_id)  # None if not measured / dropped
        annotated.append((protein_id, cos_sim, percentile))
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--patients", nargs="+", default=[],
                         help="existing CPTAC patient IDs, looked up in --cohort's abundance matrix")
    parser.add_argument("--cohort", default="luad")
    parser.add_argument("--uploaded-csv", default=None,
                         help="path to a researcher's own uploaded sample (gene name + raw abundance, "
                              "one row per protein) -- preprocessed via normalize_uploaded_sample.py "
                              "and annotated alongside any --patients given")
    parser.add_argument("--gene-col", default="gene", help="only used with --uploaded-csv")
    parser.add_argument("--abundance-col", default="abundance", help="only used with --uploaded-csv")
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    if not args.patients and not args.uploaded_csv:
        parser.error("provide at least one of --patients or --uploaded-csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_lookup = build_gene_lookup()

    print(f"Query: \"{args.query}\"")
    print(f"Stage 1: retrieving top {args.top_n} candidates by semantic relevance...")
    candidates = stage1_retrieve(args.query, args.top_n, device)

    per_patient = {}
    for patient_id in args.patients:
        per_patient[patient_id] = stage2_annotate(candidates, patient_id, args.cohort)
    if args.uploaded_csv:
        per_patient["uploaded"] = stage2_annotate_uploaded(
            candidates, args.uploaded_csv, args.gene_col, args.abundance_col
        )

    genes = [gene_lookup.get(pid, pid) for pid, _ in candidates]
    gene_counts = {g: genes.count(g) for g in set(genes)}
    display_names = [
        f"{g} ({pid})" if gene_counts[g] > 1 else g
        for g, (pid, _) in zip(genes, candidates)
    ]

    kegg_members = get_kegg_members(args.query)
    if kegg_members is not None:
        found = sorted(set(genes) & kegg_members)
        print(f"KEGG ground truth: {len(found)} of {len(kegg_members)} known pathway "
              f"members found in top {args.top_n} ({', '.join(found) if found else 'none'})")
    else:
        print("Query is not a literal KEGG pathway name — no ground truth to check against.")

    member_col = kegg_members is not None
    column_labels = list(per_patient.keys())  # --patients IDs, then "uploaded" if given
    header = f"{'Gene':<22} {'cos_sim':>8}"
    if member_col:
        header += f"  {'KEGG?':>6}"
    for label in column_labels:
        header += f"  {label:>14}"
    print("\n" + header)
    print("-" * len(header))

    for i, (_, cos_sim) in enumerate(candidates):
        row = f"{display_names[i]:<22} {cos_sim:>8.4f}"
        if member_col:
            row += f"  {'✓' if genes[i] in kegg_members else '':>6}"
        for label in column_labels:
            _, _, pct = per_patient[label][i]
            cell = "not detected" if pct is None else f"{pct * 100:5.1f}th pctile"
            row += f"  {cell:>14}"
        print(row)


if __name__ == "__main__":
    main()
