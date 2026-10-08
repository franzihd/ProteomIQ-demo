# ProteomIQ — demo

Natural-language search over the proteins measured in a single proteomics sample.

This repository only contains what the interactive app needs to run (code, trained
projection heads, precomputed ESM2 embeddings, two example patients). It is generated
from the main ProteomIQ repository — do not edit it by hand.

## How to try it

1. Click **Use example dataset** in the sidebar to use a real CPTAC lung adenocarcinoma
   patient (11LU013), or upload your own CSV (see format below).
2. Type a biological question, e.g. *"receptor tyrosine kinase signaling"*.
3. The app ranks the sample's proteins by semantic relevance to your query and shows
   each protein's within-sample abundance percentile.
4. Optionally click **Interpret results** for a short natural-language reading (by Claude)
   of the top hits.

**Upload format:** a long-format CSV with one row per protein, a `gene` column
(gene symbol) and an `abundance` column (raw value, any scale — it is rank-normalized
within the sample). Two example files are included under `data/test_abundance_matrix/`
(patients 11LU013 and 11LU016).

**Note:** results are intended for hypothesis generation, not diagnosis. Abundance is
a within-sample rank, not up-/down-regulation relative to a reference.

Model: frozen ESM2 (150M) + frozen BioBERT, contrastively trained projection heads
on the reviewed human UniProt proteome (multiview checkpoint).
