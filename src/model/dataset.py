import random
from pathlib import Path
import pandas as pd
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from src.embeddings.textual.embed_biobert import strip_gene_name, append_family_location, append_go_terms

# Same seed=42 convention used throughout this project (train_protein_level.py,
# protein_train_validation_test.py, ...). Seeds the multi-view random-selection
# RNG below -- reproducible given the same __getitem__ call order (this project
# does not use multi-process DataLoader workers, so a single Dataset-owned
# random.Random instance is sufficient).
MULTI_VIEW_SEED = 42


def parse_biobert_text_flags(biobert_filename: str) -> dict:
    """
    Recovers the embed_biobert.py CLI flags that would reproduce the raw text
    underlying a given biobert_embeddings*.h5 filename, from its suffix
    naming convention (see embed_biobert.py's own suffix-building logic in
    main()) -- so live/on-the-fly BioBERT encoding (BioBERT-unfreezing
    experiment, train_protein_level.py) can build the exact same text a
    precomputed .h5 file would have been built from, without a second,
    hand-maintained flag surface that could drift out of sync with the
    filenames.
    """
    stem = biobert_filename.removeprefix("biobert_embeddings").removesuffix(".h5")
    return {
        "mask_gene_names": "_masked" in stem,
        "include_go_terms": "_go" in stem,
        "go_bp_only": "_go_bp_only" in stem,
        "include_family": "_family" in stem,
        "include_location": "_location" in stem,
    }


def build_live_protein_texts(mask_gene_names: bool = False, include_go_terms: bool = False,
                              go_bp_only: bool = False, include_family: bool = False,
                              include_location: bool = False) -> dict:
    """
    Builds the same per-protein text strings embed_biobert.py would embed,
    keyed by ENSP_ID, for on-the-fly BioBERT encoding instead of reading a
    precomputed embedding from disk (needed when BioBERT Transformer layers
    are unfrozen and require a live forward pass with gradients). Mirrors
    embed_biobert.py's main() text-construction logic exactly (same order:
    mask -> GO terms -> family/location).
    """
    annotations_path = Path("data/full_proteome/protein_annotations.csv")
    annotations = pd.read_csv(annotations_path)
    annotations = annotations.dropna(subset=["Sequence", "Annotation"]).reset_index(drop=True)
    n_proteins = len(annotations)

    if mask_gene_names:
        texts = [strip_gene_name(g, a) for g, a in zip(annotations["Gene"], annotations["Annotation"])]
    else:
        texts = annotations["Annotation"].tolist()

    if include_go_terms:
        go_bp = annotations.get("GO_Biological_Process", pd.Series([None] * n_proteins))
        go_cc = annotations.get("GO_Cellular_Component", pd.Series([None] * n_proteins))
        texts = [append_go_terms(t, bp, cc, include_cc=not go_bp_only)
                 for t, bp, cc in zip(texts, go_bp, go_cc)]

    if include_family or include_location:
        family_col = annotations.get("Family", pd.Series([None] * n_proteins))
        location_col = annotations.get("Location", pd.Series([None] * n_proteins))
        texts = [append_family_location(t, fam, loc, include_family, include_location)
                 for t, fam, loc in zip(texts, family_col, location_col)]

    return dict(zip(annotations["ENSP_ID"], texts))


class ProteomicsDataset(Dataset):
    def __init__(
        self,
        cohorts: list[str],
        test_patient_ids: set[str] | None = None,
        use_abundance: bool = True,
        held_out_protein_ids: set[str] | None = None,
        restrict_to_protein_ids: set[str] | None = None,
        biobert_filename: str = "biobert_embeddings.h5",
        live_text: bool = False,
        text_flags: dict | None = None,
        view_biobert_filenames: list[str] | None = None,
        multi_positive: bool = False,
    ):
        if test_patient_ids is None:
            test_patient_ids = set()
        if held_out_protein_ids is None:
            held_out_protein_ids = set()

        self.use_abundance = use_abundance
        self.live_text = live_text
        self.multi_view = view_biobert_filenames is not None
        self.multi_positive = multi_positive
        if self.multi_view and live_text:
            raise ValueError("view_biobert_filenames (precomputed multi-view random "
                              "selection) and live_text (live BioBERT unfreeze encoding) "
                              "are not combinable -- not needed for either experiment as "
                              "currently scoped, would add live-encoding complexity to "
                              "every one of the (up to 3) views per sample")
        if multi_positive and not self.multi_view:
            raise ValueError("multi_positive requires view_biobert_filenames -- it needs "
                              "more than one text view per protein to have multiple "
                              "positives to work with in the first place")

        if not use_abundance:
            # Patient-invariant path (Stage 1):
            # protein identity/text now comes from a single canonical source
            # (data/full_proteome/), covering all reviewed human UniProt
            # proteins with a valid annotation -- not just the ones measured
            # in some CPTAC cohort's abundance matrix, and not built by
            # merging per-cohort files anymore. `cohorts` is unused on this
            # path; kept in the signature only for call-site compatibility
            # with the use_abundance=True path below.
            data_dir = Path("data/full_proteome")

            with h5py.File(data_dir / "esm2_embeddings.h5", "r") as h5f:
                esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
                esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]
            esm2_id_to_idx = {pid: i for i, pid in enumerate(esm2_protein_ids)}

            if self.multi_view:
                # Multi-view random selection: Function/Family/Location are
                # three SEPARATE, independently-embedded text sources (unlike
                # M0-M3, which concatenate them into one string) -- one is
                # picked at random per protein PER __getitem__ CALL (not fixed
                # once per training run), the same "vary what's shown, don't
                # add more content per example" principle as image augmentation
                # (random crop/flip), motivated by 11z's finding that
                # concatenating more distinguishing text helps closed-set
                # self-ID tasks while hurting open-vocabulary generalization.
                # Coverage differs per view (Function ~100%, Family 75.4%,
                # Location 92.6% of the base set) -- a protein's available
                # views are whichever of the given files actually contain its
                # protein_id, checked per-protein below, NOT assumed uniform.
                view_matrices = []
                view_id_to_idx = []
                view_ids_union = set()
                for fname in view_biobert_filenames:
                    with h5py.File(data_dir / fname, "r") as h5f:
                        view_matrices.append(torch.tensor(h5f["embeddings"][:], dtype=torch.float32))
                        ids = [pid.decode() for pid in h5f["protein_ids"][:]]
                    view_id_to_idx.append({pid: i for i, pid in enumerate(ids)})
                    view_ids_union |= set(ids)

                usable_proteins = (set(esm2_protein_ids) & view_ids_union) - held_out_protein_ids
                if restrict_to_protein_ids is not None:
                    usable_proteins = usable_proteins & restrict_to_protein_ids
                usable_proteins = sorted(usable_proteins)  # see 11y -- deterministic base order

                self.protein_esm2 = {pid: esm2_matrix[esm2_id_to_idx[pid]] for pid in usable_proteins}
                self.protein_text_views = {
                    pid: [matrix[id_to_idx[pid]] for matrix, id_to_idx in zip(view_matrices, view_id_to_idx)
                          if pid in id_to_idx]
                    for pid in usable_proteins
                }
                n_with_all_views = sum(1 for v in self.protein_text_views.values() if len(v) == len(view_biobert_filenames))
                print(f"Multi-view dataset: {len(usable_proteins):,} proteins, "
                      f"{n_with_all_views:,} ({n_with_all_views / len(usable_proteins) * 100:.1f}%) "
                      f"have all {len(view_biobert_filenames)} views available")

                if self.multi_positive:
                    # One row per (protein, view) instead of one row per
                    # protein with a random draw -- every available view
                    # becomes its own training example, so a protein with
                    # all 3 views contributes 3 rows in this epoch, not 1.
                    # protein_id_to_idx gives each protein a stable integer
                    # id (sorted order, same determinism precedent as
                    # `usable_proteins = sorted(...)` above) -- used by the
                    # multi-positive loss to tell which rows in a batch
                    # belong to the same protein. GroupedBatchSampler (see
                    # below) is what actually gets a protein's rows to
                    # co-occur in one batch -- plain shuffling would not.
                    self.protein_id_to_idx = {pid: i for i, pid in enumerate(usable_proteins)}
                    self.pairs = [
                        (pid, view_idx)
                        for pid in usable_proteins
                        for view_idx in range(len(self.protein_text_views[pid]))
                    ]
                    print(f"Multi-positive dataset: {len(self.pairs):,} (protein, view) rows "
                          f"from {len(usable_proteins):,} unique proteins")
                    return

                self._view_rng = random.Random(MULTI_VIEW_SEED)
                self.pairs = list(self.protein_esm2.keys())
                return

            if live_text:
                # BioBERT-unfreezing experiment: text must be encoded live
                # (with gradients) rather than looked up from a precomputed
                # .h5 file, so build the same raw text strings that file
                # would have been embedded from instead.
                protein_text_map = build_live_protein_texts(**(text_flags or {}))
                biobert_protein_ids = list(protein_text_map.keys())
            else:
                with h5py.File(data_dir / biobert_filename, "r") as h5f:
                    biobert_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
                    biobert_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]
                biobert_id_to_idx = {pid: i for i, pid in enumerate(biobert_protein_ids)}

            usable_proteins = (set(esm2_protein_ids) & set(biobert_protein_ids)) - held_out_protein_ids
            if restrict_to_protein_ids is not None:
                usable_proteins = usable_proteins & restrict_to_protein_ids
            # Sorted, not iterated as a raw set: Python randomizes string hash
            # values per process (PYTHONHASHSEED unset), so raw set iteration
            # order -- and therefore self.pairs' base order, upstream of the
            # seeded DataLoader shuffle -- silently differed between separate
            # training runs even with identical torch/shuffle seeds. This
            # was a real, undetected source of run-to-run training variance
            # (confirmed: cosine similarity ~0.97 between two "identical"
            # runs' trained weights, not the ~0.9999+ pure float noise would
            # produce). Sorting fixes the base order deterministically,
            # independent of hash seed; doesn't change which proteins are
            # used, only pins down the order before shuffling.
            usable_proteins = sorted(usable_proteins)

            self.protein_esm2 = {pid: esm2_matrix[esm2_id_to_idx[pid]] for pid in usable_proteins}
            if live_text:
                self.protein_text = {pid: protein_text_map[pid] for pid in usable_proteins}
            else:
                self.protein_text = {pid: biobert_matrix[biobert_id_to_idx[pid]] for pid in usable_proteins}
            self.pairs = list(self.protein_esm2.keys())
            return

        # (cohort_idx, patient_id, protein_id, abundance) — one row per
        # (patient, protein) measurement; kept per-cohort since abundance
        # is cohort/patient-specific. Ablation-baseline path only (section
        # 1-10) -- unchanged.
        self.pairs = []
        self.cohort_data = []  # per-cohort lookup structures

        for cohort_idx, cohort in enumerate(cohorts):
            data_dir = Path(f"data/cptac/{cohort}")

            abundance_df = pd.read_csv(
                data_dir / "rank_normalized_proteomics.csv",
                header=[0, 1], index_col=0
            )

            with h5py.File(data_dir / "esm2_embeddings.h5", "r") as h5f:
                esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
                esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

            with h5py.File(data_dir / biobert_filename, "r") as h5f:
                biobert_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
                biobert_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

            esm2_id_to_idx = {pid: i for i, pid in enumerate(esm2_protein_ids)}
            biobert_protein_id_to_idx = {pid: i for i, pid in enumerate(biobert_protein_ids)}

            # Text no longer varies per patient (clinical description dropped), but we still
            # restrict training to the same patient pool as before (patients with a valid
            # clinical description) so this retrain isolates text content as the only variable.
            descriptions_path = data_dir / "descriptions.csv"
            if not descriptions_path.exists():
                descriptions_path = data_dir / "clinical_descriptions.csv"
            description_patient_ids = set(pd.read_csv(descriptions_path, index_col=0).index)

            abundance_protein_ids = [col[1] for col in abundance_df.columns]
            abundance_patient_ids = list(abundance_df.index)

            shared_proteins = (
                set(esm2_protein_ids) & set(biobert_protein_ids) & set(abundance_protein_ids)
            ) - held_out_protein_ids
            if restrict_to_protein_ids is not None:
                shared_proteins = shared_proteins & restrict_to_protein_ids
            shared_patients = (
                description_patient_ids & set(abundance_patient_ids)
            ) - test_patient_ids

            protein_id_to_col = {pid: i for i, pid in enumerate(abundance_protein_ids)}
            abundance_array = abundance_df.values
            patient_id_to_row = {pid: i for i, pid in enumerate(abundance_patient_ids)}

            for patient_id in shared_patients:
                patient_row = patient_id_to_row[patient_id]
                for protein_id in shared_proteins:
                    val = abundance_array[patient_row, protein_id_to_col[protein_id]]
                    if not np.isnan(val):
                        self.pairs.append((cohort_idx, patient_id, protein_id, float(val)))

            self.cohort_data.append({
                "esm2_matrix": esm2_matrix,
                "biobert_matrix": biobert_matrix,
                "esm2_id_to_idx": esm2_id_to_idx,
                "biobert_protein_id_to_idx": biobert_protein_id_to_idx,
            })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.use_abundance:
            cohort_idx, patient_id, protein_id, abundance = self.pairs[idx]
            cd = self.cohort_data[cohort_idx]
            esm2_vec = cd["esm2_matrix"][cd["esm2_id_to_idx"][protein_id]]
            protein_emb = torch.cat([esm2_vec, torch.tensor([abundance], dtype=torch.float32)])
            text_emb = cd["biobert_matrix"][cd["biobert_protein_id_to_idx"][protein_id]]
        elif self.multi_positive:
            protein_id, view_idx = self.pairs[idx]
            protein_emb = self.protein_esm2[protein_id]
            text_emb = self.protein_text_views[protein_id][view_idx]
            # Returned alongside the two embeddings (unlike every other path
            # in this dataset) so the multi-positive loss can tell which
            # rows in a batch share a protein -- plain (protein_emb, text_emb)
            # carries no such signal on its own.
            protein_idx = self.protein_id_to_idx[protein_id]
            return protein_emb, text_emb, protein_idx
        elif self.multi_view:
            protein_id = self.pairs[idx]
            protein_emb = self.protein_esm2[protein_id]
            views = self.protein_text_views[protein_id]
            # Re-drawn on every call (not fixed once per protein/epoch) --
            # the same protein can show a different view across epochs and
            # across duplicate draws within an epoch.
            text_emb = self._view_rng.choice(views)
        else:
            protein_id = self.pairs[idx]
            protein_emb = self.protein_esm2[protein_id]
            text_emb = self.protein_text[protein_id]

        return protein_emb, text_emb


class GroupedBatchSampler:
    """
    Used only with ProteomicsDataset(multi_positive=True). Plain random
    shuffling of the (protein, view) rows would almost never put the same
    protein's 2-3 rows in the same batch (a batch covers only ~1.6% of the
    ~36k rows) -- and the multi-positive loss has nothing to pull together
    if a protein's rows never co-occur. This sampler instead shuffles
    PROTEINS (not rows), then fills each batch by pulling in a whole
    protein's row-group at a time, so every row belonging to a protein that
    makes it into a batch is guaranteed to land in that same batch together.

    Batches are built greedily and are NOT exactly `batch_size` -- a
    protein's group (1-3 rows) is never split across batches, so the last
    group added to a batch may push it slightly over `batch_size`, and the
    final batch of an epoch is typically smaller. This is deliberately
    tolerated rather than padded/truncated -- both InfoNCE-style losses
    here already handle variable batch sizes fine (the loss and the
    in-batch recall diagnostic both read the actual batch size off the
    tensors, nothing assumes a fixed 512).

    Seeded (MULTI_VIEW_SEED, same convention as the rest of this file) so
    batch composition is reproducible across runs, matching the
    determinism precedent set for the DataLoader shuffle order elsewhere
    in this project (see train_protein_level.py's `shuffle_generator`).
    """
    def __init__(self, dataset: "ProteomicsDataset", batch_size: int, seed: int = MULTI_VIEW_SEED):
        if not getattr(dataset, "multi_positive", False):
            raise ValueError("GroupedBatchSampler requires a ProteomicsDataset built with "
                              "multi_positive=True -- dataset.pairs must be (protein_id, view_idx) "
                              "rows for the protein-grouping logic below to apply")
        self.batch_size = batch_size
        self._rng = random.Random(seed)

        # groups: protein_id -> list of row-indices (into dataset.pairs)
        groups: dict = {}
        for row_idx, (protein_id, _view_idx) in enumerate(dataset.pairs):
            groups.setdefault(protein_id, []).append(row_idx)
        # Sorted protein order first (deterministic base order, same
        # rationale as `usable_proteins = sorted(...)` elsewhere in this
        # file), THEN shuffled per-epoch via self._rng -- so the *set* of
        # groups is fixed and only their draw order varies.
        self._protein_ids = sorted(groups.keys())
        self._groups = groups
        self._n_batches = None  # computed lazily on first __len__/__iter__ call

    def __iter__(self):
        protein_order = list(self._protein_ids)
        self._rng.shuffle(protein_order)

        batch: list[int] = []
        for protein_id in protein_order:
            batch.extend(self._groups[protein_id])
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        if self._n_batches is None:
            self._n_batches = sum(1 for _ in self)
        return self._n_batches
