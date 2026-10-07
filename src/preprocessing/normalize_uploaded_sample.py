"""
Preprocesses a researcher's uploaded proteomics sample (gene name + raw
abundance value, one row per protein) into the same shape Stage 2 already
expects: ENSP_ID -> within-sample abundance percentile.

Three steps, in order:
1. Rank-normalize the raw abundance values to within-sample percentiles --
   the same operation (df.rank(pct=True)) already used for CPTAC cohorts in
   rank_normalize.py, applied here to a single uploaded sample instead of a
   whole cohort's patient x protein matrix. Done FIRST, over the full
   uploaded sample as-is, so a protein's percentile reflects its position
   among everything actually measured -- not a smaller, already-filtered
   denominator.
2. Drop genes that appear more than once in the UPLOADED file itself (e.g. a
   researcher's own file separately measured multiple isoforms of the same
   gene). Rather than guess which measurement is "the" one, or average them
   into an invented blended number, these are dropped entirely -- the same
   principle already used elsewhere in this project for ambiguous cases
   (DeepLoc's dual-localization proteins, EC's multi-class-spanning enzymes):
   when a single value can't be assigned confidently, drop rather than
   fabricate one.
3. Match each remaining gene name to its ENSP_ID via
   data/full_proteome/protein_annotations.csv. A small number of gene names
   (~20 of ~16,893) map to more than one ENSP_ID in OUR reference set --
   genes that produce more than one distinct protein from the same locus
   (e.g. CDKN2A -> two unrelated proteins via alternative reading frames).
   Resolved deterministically by keeping the alphabetically-first ENSP_ID
   per gene name -- affects a tiny fraction of proteins, not worth more
   complex disambiguation. Genes with no matching entry at all (no UniProt
   FUNCTION annotation was ever fetched for them -- see CLAUDE.md section 11b)
   are dropped, since Stage 1 has no way to use a protein it has no text
   representation for.

Usage:
    uv run python src/preprocessing/normalize_uploaded_sample.py \\
        --input path/to/sample.csv --gene-col gene --abundance-col abundance
"""
import argparse
from pathlib import Path
import pandas as pd

PROTEOME_ANNOTATIONS = Path("data/full_proteome/protein_annotations.csv")


def build_gene_to_ensp_lookup() -> dict:
    df = pd.read_csv(PROTEOME_ANNOTATIONS, usecols=["ENSP_ID", "Gene"])
    df = df.sort_values("ENSP_ID").drop_duplicates(subset="Gene", keep="first")
    return dict(zip(df["Gene"], df["ENSP_ID"]))


def normalize_uploaded_sample(input_path, gene_col: str = "gene", abundance_col: str = "abundance"):
    """Returns (result_df, unmatched_genes, n_ambiguous). result_df has columns
    [gene_col, 'ENSP_ID', 'percentile', 'raw_abundance'] -- percentile/ENSP_ID
    are ready to feed directly into Stage 2's existing annotation lookup, same
    shape as a row from rank_normalized_proteomics.csv; raw_abundance is kept
    alongside for visualizations (e.g. a rank-abundance plot) where the
    percentile alone isn't informative on a log scale."""
    df = pd.read_csv(input_path)
    gene_to_ensp = build_gene_to_ensp_lookup()

    # Rank BEFORE filtering -- a protein's percentile must reflect its position
    # among EVERYTHING actually measured in this sample, same order of
    # operations as rank_normalize.py's cohort-level pipeline. Filtering to the
    # annotated/usable subset happens afterward, as separate steps, so it
    # never shrinks the ranking denominator.
    df["percentile"] = df[abundance_col].rank(pct=True)

    # Drop genes that appear more than once in the uploaded file itself (e.g.
    # multiple isoforms measured separately under the same gene name) -- see
    # module docstring for why this is dropped rather than averaged.
    dupe_mask = df[gene_col].duplicated(keep=False)
    n_ambiguous = int(dupe_mask.sum())
    df = df[~dupe_mask].copy()

    df["ENSP_ID"] = df[gene_col].map(gene_to_ensp)
    matched = df.dropna(subset=["ENSP_ID"]).copy()
    unmatched = df[df["ENSP_ID"].isna()][gene_col].tolist()

    result = (
        matched[[gene_col, "ENSP_ID", "percentile", abundance_col]]
        .rename(columns={abundance_col: "raw_abundance"})
        .reset_index(drop=True)
    )
    return result, unmatched, n_ambiguous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gene-col", default="gene")
    parser.add_argument("--abundance-col", default="abundance")
    parser.add_argument("--output", default=None,
                         help="optional path to save the normalized table as CSV")
    args = parser.parse_args()

    result, unmatched, n_ambiguous = normalize_uploaded_sample(args.input, args.gene_col, args.abundance_col)

    print(f"Matched {len(result)} proteins to our reference set")
    if n_ambiguous:
        print(f"{n_ambiguous} rows dropped -- gene name appeared more than once in the uploaded "
              f"file itself (ambiguous, e.g. multiple isoforms measured separately)")
    if unmatched:
        preview = ", ".join(unmatched[:10]) + ("..." if len(unmatched) > 10 else "")
        print(f"{len(unmatched)} gene names not found in our reference set (dropped): {preview}")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
