from pathlib import Path
import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse



MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
BATCH_SIZE = 256
MAX_LENGTH = 512

# --field choice -> protein_annotations.csv column name, for the solo-field
# multi-view embedding mode.
FIELD_COLUMNS = {"family": "Family", "location": "Location"}


def mean_pool(token_embeddings, attention_mask):
    # Expand mask from (batch, seq_len) to (batch, seq_length, 768) so it matches token_embeddings.shape (BioBERT produces 768-dimensional embeddings)
    # Each 0/1 value is repeated 768 times to create a mask that can be applied to the token embeddings
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Zero out padding token embeddings and compute mean by dividing the sum of token embeddings by the number of non-padding tokens (sum of mask)
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def embed_batch (texts, tokenizer, model, device):
    # Tokenize the batch of texts, applying padding and truncation to ensure all sequences are the same length (max 512 tokens)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    # Move all tensors to the GPU (or CPU if GPU is not available) before passing them to the model
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Run BioBERT without computing gradients - we are not training, just extracting embeddings
    with torch.no_grad():
        outputs = model(**encoded)

    # output.last_hidden_state shape: (batch, seq_len, 768) — one vector per token.
    # mean_pool reduces it to (batch, 768) — one vector per sentence.
    # .cpu() moves result off GPU, .numpy() converts to numpy array.
    return mean_pool(outputs.last_hidden_state, encoded['attention_mask']).cpu().numpy()


def strip_gene_name(gene, annotation):
    # Annotation is always built as "{gene}: {function text}" (fetch_uniprot.py).
    # Strips just that leading prefix -- the gene-name-masking ablation -- so the
    # rest of the functional description is untouched.
    prefix = f"{gene}: "
    if isinstance(gene, str) and annotation.startswith(prefix):
        return annotation[len(prefix):]
    return annotation


def append_family_location(text, family, location, include_family, include_location):
    # Family/Location come from data/full_proteome/protein_annotations.csv's
    # Family/Location columns (added by src/data_acquisition/fetch_family_location.py).
    # Empty cells round-trip through CSV as NaN (pandas turns empty strings
    # into NaN on read), not '' -- the isinstance(x, str) checks below guard
    # against that ("nan" leaking into the text as a literal word).
    # Comma-separated, no field labels (e.g. no "Family: ..." prefix) --
    # deliberate choice:
    # avoids the label-prefixed, term-list-like format that was diagnosed
    # as a self-ID shortcut in the (abandoned) GO-term augmentation ablation
    # (section 10n). Family/Location text is already full sentences (e.g.
    # "Belongs to the small GTPase superfamily. Arf family." / "Located in
    # Cell membrane, Golgi apparatus."), so plain comma-joining reads as one
    # flowing description rather than a labeled field dump.
    parts = [text]
    if include_family and isinstance(family, str) and family.strip():
        parts.append(family.strip())
    if include_location and isinstance(location, str) and location.strip():
        parts.append(location.strip())
    return ", ".join(parts)


def append_go_terms(text, go_bp, go_cc, include_cc=True):
    # go_bp/go_cc are "; "-joined GO term label strings from fetch_uniprot.py
    # (already capped at MAX_GO_TERMS_PER_CATEGORY there), or NaN when a
    # protein has no terms in that category (~8% BP, ~3% CC) -- skip those.
    # include_cc=False drops Cellular Component terms entirely: they are far
    # more generic than BP terms (e.g. "cytosol"/"nucleus"/"cytoplasm" each
    # appear on 30%+ of all proteins -- verified against protein_annotations.csv),
    # which risks diluting the mean-pooled BioBERT embedding with boilerplate
    # shared across huge swaths of unrelated proteins rather than adding
    # distinguishing signal -- the suspected cause of the masked+GO pathway
    # AUROC/recall@K regression vs. masked-only.
    parts = [text]
    if isinstance(go_bp, str) and go_bp.strip():
        parts.append(f"Biological process: {go_bp}.")
    if include_cc and isinstance(go_cc, str) and go_cc.strip():
        parts.append(f"Cellular component: {go_cc}.")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--field", choices=["family", "location"], default=None,
                        help="embed ONLY this field's raw text as the sole input (not "
                             "appended to Function), restricted to proteins that have a "
                             "non-null value for it -- for the multi-view random-selection "
                             "experiment, where Function/Family/Location "
                             "are three separate, independently-embedded text sources and "
                             "one is picked at random per protein per training sample, "
                             "rather than concatenated as in M0-M3. Not combinable with "
                             "--mask-gene-names/--include-go-terms/--include-family/"
                             "--include-location (those all build appended Function-based "
                             "text). Writes to biobert_embeddings_family_view.h5 or "
                             "biobert_embeddings_location_view.h5. Gene names are NOT "
                             "masked out of Family text for this mode (14.1% of Family "
                             "values contain the protein's own gene symbol, e.g. "
                             "'Belongs to the UBR4 family.' for gene UBR4 -- checked and "
                             "left in deliberately, unlike Function text's masking)")
    parser.add_argument("--mask-gene-names", action="store_true",
                        help="strip the leading '{gene}: ' prefix from each annotation "
                             "before embedding; writes to biobert_embeddings_masked.h5 "
                             "instead of biobert_embeddings.h5")
    parser.add_argument("--include-go-terms", action="store_true",
                        help="append GO Biological Process + Cellular Component term "
                             "labels (from protein_annotations.csv) after the annotation "
                             "text; combinable with --mask-gene-names")
    parser.add_argument("--go-bp-only", action="store_true",
                        help="with --include-go-terms, append only Biological Process "
                             "terms and skip Cellular Component (which is highly generic "
                             "-- cytosol/nucleus/cytoplasm each cover 30%%+ of all proteins)")
    parser.add_argument("--include-family", action="store_true",
                        help="append the Family field (UniProt SIMILARITY comment, "
                             "protein family/superfamily membership), comma-separated, "
                             "no label prefix. Independent of --include-location -- "
                             "combine freely (e.g. to test field order/subsets later). "
                             "Requires a Family column in protein_annotations.csv "
                             "(see fetch_family_location.py)")
    parser.add_argument("--include-location", action="store_true",
                        help="append the Location field (UniProt SUBCELLULAR LOCATION, "
                             "top-level compartment names only, deduplicated), "
                             "comma-separated. Independent of --include-family -- can "
                             "be used alone or combined; the default appended ORDER is "
                             "always Function -> Family -> Location regardless of which "
                             "flags are set (see append_family_location) -- there is no "
                             "flag yet to reorder fields, only to include/exclude them")
    args = parser.parse_args()

    if args.field and (args.mask_gene_names or args.include_go_terms or args.include_family or args.include_location):
        raise SystemExit("--field is not combinable with --mask-gene-names/--include-go-terms/"
                          "--include-family/--include-location -- it builds a solo-field "
                          "embedding, those all build appended Function-based text")

    # full_proteome is NOT a CPTAC cohort -- it's the separate, unified
    # proteome-wide directory (data/full_proteome/), never moved under data/cptac/ during that reorg. The other 9
    # real cohort names (luad, brca, ...) do live under data/cptac/{cohort}.
    data_dir = Path("data/full_proteome") if args.cohort == "full_proteome" else Path(f"data/cptac/{args.cohort}")
    annotations_path = data_dir / "protein_annotations.csv"
    if args.field:
        output_filename = f"biobert_embeddings_{args.field}_view.h5"
    else:
        suffix = ""
        if args.mask_gene_names:
            suffix += "_masked"
        if args.include_go_terms:
            suffix += "_go_bp_only" if args.go_bp_only else "_go"
        if args.include_family:
            suffix += "_family"
        if args.include_location:
            suffix += "_location"
        output_filename = f"biobert_embeddings{suffix}.h5"
    output_path = data_dir / output_filename
    # Load protein annotations and drop rows where Sequence or Annotation is missing
    # reset_index is used to reset the DataFrame index after dropping rows, so it goes from 0 to n-1 instead of keeping the original indices
    annotations = pd.read_csv(annotations_path)
    annotations = annotations.dropna(subset=["Sequence", "Annotation"]).reset_index(drop=True)

    if args.field:
        # Solo-field mode: further restrict to proteins that actually HAVE this
        # field (Family 75.4% / Location 92.6% coverage of the base set) -- the
        # resulting .h5 has fewer rows than the standard Function-based files,
        # by design. dataset.py's multi-view random selection checks per-protein
        # which view-files contain that protein_id rather than assuming all
        # three always have the same coverage.
        field_col = FIELD_COLUMNS[args.field]
        n_before = len(annotations)
        annotations = annotations.dropna(subset=[field_col]).reset_index(drop=True)
        print(f"{field_col} coverage: {len(annotations)}/{n_before} ({len(annotations)/n_before*100:.1f}%)")

    n_proteins = len(annotations)
    print(f"Proteins: {n_proteins}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Download BioBERT weights from HuggingFace (cached after first run).
    # .to(device) moves model to GPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    if args.field:
        # Raw field text is the sole embedded input -- no masking (gene names
        # deliberately left in for Family, see --field's help text above), no
        # appending, no GO terms.
        field_col = FIELD_COLUMNS[args.field]
        texts = annotations[field_col].tolist()
    elif args.mask_gene_names:
        texts = [strip_gene_name(g, a) for g, a in zip(annotations["Gene"], annotations["Annotation"])]
        n_masked = sum(t != a for t, a in zip(texts, annotations["Annotation"]))
        print(f"Gene-name prefix stripped from {n_masked}/{n_proteins} annotations")
    else:
        texts = annotations["Annotation"].tolist()

    if args.include_go_terms:
        go_bp = annotations.get("GO_Biological_Process", pd.Series([None] * n_proteins))
        go_cc = annotations.get("GO_Cellular_Component", pd.Series([None] * n_proteins))
        texts = [append_go_terms(t, bp, cc, include_cc=not args.go_bp_only)
                 for t, bp, cc in zip(texts, go_bp, go_cc)]
        if args.go_bp_only:
            n_with_go = sum(isinstance(bp, str) for bp in go_bp)
            print(f"GO Biological Process terms appended to {n_with_go}/{n_proteins} annotations (CC skipped)")
        else:
            n_with_go = sum(isinstance(bp, str) or isinstance(cc, str) for bp, cc in zip(go_bp, go_cc))
            print(f"GO terms appended to {n_with_go}/{n_proteins} annotations")

    if args.include_family or args.include_location:
        for col in (["Family"] if args.include_family else []) + (["Location"] if args.include_location else []):
            if col not in annotations.columns:
                raise SystemExit(f"--include-{col.lower()} requires a '{col}' column in {annotations_path} "
                                 f"(run src/data_acquisition/fetch_family_location.py first)")
        family_col = annotations.get("Family", pd.Series([None] * n_proteins))
        location_col = annotations.get("Location", pd.Series([None] * n_proteins))
        texts = [append_family_location(t, fam, loc, args.include_family, args.include_location)
                 for t, fam, loc in zip(texts, family_col, location_col)]
        if args.include_family:
            n_with_family = sum(isinstance(f, str) for f in family_col)
            print(f"Family appended to {n_with_family}/{n_proteins} annotations")
        if args.include_location:
            n_with_location = sum(isinstance(l, str) for l in location_col)
            print(f"Location appended to {n_with_location}/{n_proteins} annotations")

    # Open HDF5 file for writing. All datasets inside are written directly to disk.
    mode = "r+" if output_path.exists() else "w"
    with h5py.File(output_path, mode) as h5f:

        if mode == "w":
            emb_ds = h5f.create_dataset("embeddings", shape=(n_proteins, 768), dtype="float32")
            h5f.create_dataset("protein_ids", data=np.array(annotations["ENSP_ID"].tolist(), dtype="S30"))
            completed = h5f.create_dataset("completed", data=np.zeros(n_proteins, dtype=bool))
        else:
            emb_ds = h5f["embeddings"]
            completed = h5f["completed"]
            print(f"Resuming from protein {int(completed[:].sum())}/{n_proteins}")

        for i in tqdm(range(0, n_proteins, BATCH_SIZE), desc="Proteins"):

            if completed[i]:
                continue

            batch_emb = embed_batch(texts[i : i + BATCH_SIZE], tokenizer, model, device)
            emb_ds[i : i + BATCH_SIZE] = batch_emb
            completed[i : i + BATCH_SIZE] = True

    print(f"Done. Saved to {output_path}")




if __name__ == "__main__":
    main()
