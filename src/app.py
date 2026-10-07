import base64
import math
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import torch
import pandas as pd
import anthropic
import matplotlib
matplotlib.use("Agg")  # headless -- this runs inside the Streamlit server process, no display needed
import matplotlib.pyplot as plt
from adjustText import adjust_text

from src.model.retrieve_and_annotate import load_model, load_esm2_lookup, build_gene_lookup
from src.model.interpret import (
    build_annotation_lookup,
    retrieve_within_sample,
    build_chat_system_prompt,
    extract_gene_mentions,
    lookup_gene,
    stage3_chat,
    DEFAULT_TOP_K,
)
from src.preprocessing.normalize_uploaded_sample import normalize_uploaded_sample, build_gene_to_ensp_lookup

EXAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "test_abundance_matrix" / "11LU013.csv"
EXAMPLE_LABEL = "LUAD patient 11LU013 (from the CPTAC study data)"
REQUIRED_FILES = [
    Path("data/full_proteome/esm2_embeddings.h5"),
    Path("data/full_proteome/protein_annotations.csv"),
    Path("data/runs/full_proteome/multiview_experiment/multiview_best.pt"),
]

# --- Branding assets --------------------------------------------------------
# Not in REQUIRED_FILES / the hard-stop check above: missing branding is a
# cosmetic problem (falls back to the plain emoji title below), not a reason
# to refuse to start the way a missing checkpoint or embeddings file is.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSETS_DIR / "icon.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"  # pre-resized 256x256 copy of ICON_PATH -- see _load_favicon
LOGO_PATH = ASSETS_DIR / "logo.jpg"

# Brand palette (single source of truth, also used by the CSS :root vars
# below, so the matplotlib rank plot doesn't clash with the rest of the
# page). Semantic convention: blue = protein/proteomics, purple = language/
# query/semantic-search, navy = neutral structure (headings, main UI).
BRAND_NAVY = "#14213D"
BRAND_BLUE = "#2A7BCB"
BRAND_VIOLET = "#7B5CD6"


def _load_favicon():
    """Path to the pre-resized favicon (see FAVICON_PATH) -- deliberately a
    plain file path string, not a loaded PIL Image. Confirmed via Streamlit's
    own source (commands/page_config.py): a PIL Image is forced through
    image_to_url with channels="RGB", which mishandles our icon's actual
    RGBA/transparency and silently produces an empty favicon URL (no
    exception raised -- it just falls back to Streamlit's own default,
    which is what was actually showing up). A plain path string takes a
    much simpler code path (raw file bytes, no channel/PIL handling at all),
    which sidesteps the issue entirely. Falls back to None (Streamlit's own
    default icon) rather than an emoji if the asset is missing -- no emoji
    anywhere in the UI, including edge-case fallbacks."""
    if FAVICON_PATH.exists():
        return str(FAVICON_PATH)
    return None


def _load_logo_b64():
    """Base64-embedded wordmark for the header. Streamlit has no built-in
    static-asset serving, so embedding as a data URI inside custom HTML is
    the standard way to show a local image without a separate file server.
    Returns None (not an empty string) if the asset is missing, so callers
    can fall back to a plain-text title instead of a broken <img> tag."""
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_brand_header():
    """Logo above the hero text -- shared by the normal header and the
    early missing-data-files error path so both stay visually consistent.
    Sits directly on the page background with just margin around it
    (no bordered/shadowed card), deliberately modest in size so it never
    competes with the main product headline. Falls back to a plain text
    title if the logo asset isn't present -- no icon/emoji, per the
    no-decorative-icons design rule."""
    logo_b64 = _load_logo_b64()
    if logo_b64:
        st.markdown(
            f'<div class="logo-simple"><img class="logo-img" '
            f'src="data:image/jpeg;base64,{logo_b64}" alt="ProteomIQ"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="hero-fallback-title">ProteomIQ</div>', unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = ""):
    """One consistent section-header pattern (bold title + small muted
    subtitle) used for every content section -- Protein Search, Results,
    Chat -- instead of the old numbered step-badge treatment, so the page
    reads as one coherent application rather than a linear wizard."""
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-header"><div class="section-title">{title}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="ProteomIQ", page_icon=_load_favicon(), layout="wide")

# ---------------------------------------------------------------------------
# Styling. Streamlit's default look is intentionally neutral -- this gives
# the tool its own identity (a calm science/biology palette, card-based
# results instead of a raw dataframe, numbered steps) without needing a
# different framework.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
/* Reinforce Inter on native form controls specifically -- inputs/buttons/
   selects/textareas often keep the browser's own UI font by default even
   when a page-wide font-family is set, since form-control typography isn't
   always inherited the same way as regular text. */
input, textarea, button, select,
[data-testid="stSidebar"], [data-testid="stExpander"], [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', sans-serif;
}

/* Brand palette. Semantic convention (kept consistent throughout): blue =
   protein/proteomics, purple = language/query/semantic-search, navy =
   neutral structure (headings, main UI). Gradients are deliberately used
   sparingly -- small accents (buttons, thin bars) only, not large dominant
   boxes -- so most of the interface stays white / very light gray. */
:root {
    --navy-900: #14213D;
    --blue-600: #2A7BCB;
    --violet-600: #7B5CD6;
    --brand-100: #F5F7FA;
    --ink: #14213D;
    --muted: #5B6477;
    --card-bg: #ffffff;
    --card-border: #E4E7ED;
    --chat-user-bg: #F3EEFC;
    --chat-user-border: #E7DDF8;
    --chat-assistant-bg: #F5F8FC;
    --chat-assistant-border: #E3E8F0;
}

/* Reduce Streamlit's own default top padding. */
[data-testid="stMainBlockContainer"] { padding-top: 1.75rem !important; }

/* Logo -- a compact brand mark, deliberately modest in size so it never
   competes with the main product headline below it (req: "logo must not
   compete with the main product headline"). */
.logo-simple { margin: 0 0 0.75rem 0; }
.logo-img { height: 70px; width: auto; display: block; }
.hero-fallback-title { font-size: 1.1rem; font-weight: 700; color: var(--ink); margin: 0 0 0.5rem 0; }

/* Hero -- the entry point of a search application: one large, confident
   headline (this IS the product, not marketing copy), a small muted
   supporting line, and a thin blue->purple accent line underneath -- the
   only gradient/decorative element on the page. */
.hero { margin: 0 0 1.25rem 0; max-width: 40rem; }
.hero h1 {
    margin: 0 0 0.4rem 0; font-size: 2.3rem; font-weight: 700;
    color: var(--ink); line-height: 1.15; letter-spacing: -0.01em;
}
.hero p { margin: 0; font-size: 0.98rem; color: var(--muted); line-height: 1.5; }
.hero-accent {
    width: 2.75rem; height: 3px; border-radius: 2px; margin-top: 0.85rem;
    background: linear-gradient(90deg, var(--blue-600), var(--violet-600));
}

.minimal-empty-title { font-size: 0.95rem; font-weight: 700; color: var(--ink); margin: 1.1rem 0 0.15rem 0; }
.minimal-empty-hint { font-size: 0.85rem; color: var(--muted); }

/* Reusable section header -- one consistent pattern for every content
   section (Protein Search, Results, Chat) instead of numbered step
   badges, so the page reads as one coherent application rather than a
   linear onboarding wizard. */
.section-header { margin: 0 0 0.85rem 0; }
.section-title { font-size: 1.1rem; font-weight: 700; color: var(--ink); margin: 0 0 0.15rem 0; }
.section-sub { font-size: 0.83rem; color: var(--muted); }

/* Results -- structured rows with a subtle bottom-border separator, not
   individual bordered/padded cards around every protein. */
.result-row-item {
    display: flex; align-items: center; gap: 0.9rem;
    padding: 0.6rem 0.05rem;
    border-bottom: 1px solid var(--card-border);
}
.result-rank { font-size: 0.78rem; color: var(--muted); width: 1.6rem; flex-shrink: 0; font-variant-numeric: tabular-nums; }
.result-gene { font-weight: 700; font-size: 0.95rem; color: var(--ink); min-width: 5rem; flex-shrink: 0; }
.result-metric { flex: 1; min-width: 8rem; }
.result-metric-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
.result-metric-bar-track { background: #eef0f6; border-radius: 4px; height: 0.4rem; overflow: hidden; margin-top: 0.12rem; }
.result-metric-bar-fill { height: 100%; border-radius: 4px; }
.result-metric-value { font-size: 0.72rem; color: var(--muted); margin-top: 0.1rem; }

.callout {
    background: var(--brand-100); border-left: 3px solid var(--violet-600);
    border-radius: 8px; padding: 1rem 1.2rem; color: var(--ink); line-height: 1.55;
}

/* --- Sidebar: a compact control panel -- small uppercase section labels
   (SAMPLE, RESULTS) instead of icon-heavy headings. --- */
.sidebar-section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    color: var(--muted); text-transform: uppercase;
    margin: 1rem 0 0.35rem 0;
}
.dataset-status { margin: 0.4rem 0 0.6rem 0; }
.dataset-status-label { font-size: 0.85rem; font-weight: 600; color: var(--ink); }
.dataset-status-count { font-size: 0.78rem; color: var(--muted); }

[data-testid="stSidebar"] {
    background: var(--brand-100);
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] [data-testid="stMainBlockContainer"] { padding-top: 1.5rem !important; }

[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
    padding: 0.6rem !important;
}

[data-testid="stSidebar"] button, [data-testid="stButton"] button {
    border-radius: 8px !important;
    border: 1px solid var(--card-border) !important;
    color: var(--ink) !important;
    transition: border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stSidebar"] button:hover, [data-testid="stButton"] button:hover {
    border-color: var(--violet-600) !important;
    color: var(--violet-600) !important;
}

[data-testid="stAlertContainer"] {
    background: var(--brand-100) !important;
    border-left: 3px solid var(--blue-600) !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* Main search box -- scoped to just this input via Streamlit's st-key-*
   class (st.text_input(..., key="main_query")), so the sidebar upload /
   new-search-in-chat inputs are untouched. Larger type, brand-colored
   focus ring -- meant to be the visually dominant control on the page. */
.st-key-main_query [data-testid="stTextInput"] input {
    font-size: 1.02rem !important;
    padding: 0.8rem 1rem !important;
    border-radius: 8px !important;
    border: 1px solid var(--card-border) !important;
}
.st-key-main_query [data-testid="stTextInput"] input:focus {
    border-color: var(--violet-600) !important;
    box-shadow: 0 0 0 1px var(--violet-600) !important;
}

/* Chat panel -- a bordered section beside the results, restrained (thin
   neutral border, no gradient fill) rather than a heavy colored box.
   Targets the real st.container(key="chat_panel") wrapper (the
   `st-key-chat_panel` class Streamlit adds for a keyed container) -- NOT a
   raw-HTML div, which doesn't actually wrap subsequent Streamlit elements
   the way it looks like it should (see the comment at the container's
   Python call site for why). */
.st-key-chat_panel {
    background: #fbfbfe;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 1rem 1.2rem 0.6rem 1.2rem;
}
/* Scoped to just the chat panel's own heading -- Protein Search / Results
   keep the normal .section-title size untouched. */
.st-key-chat_panel .section-title { font-size: 19px; font-weight: 600; }

/* ============================================================
   CHAT -- one consolidated section (bubbles + typography + input).
   ============================================================
   Root cause of the earlier font-size fixes not rendering, confirmed by
   reading Streamlit's actual frontend source (static/js/index.*.js): the
   real text (p/li/strong/...) does NOT live directly inside
   [data-testid="stChatMessageContent"] -- it's one level deeper, inside a
   nested [data-testid="stMarkdownContainer"]. That component takes an
   `inheritFont` prop and, when it's not passed (a plain st.markdown() call
   never passes it), sets its own EXPLICIT font-size from Streamlit's theme
   (`fontSize: inheritFont ? 'inherit' : theme.fontSizes.md`) -- literally
   the string 'inherit' only in the true branch. Since our chat renders via
   `with st.chat_message(...): st.markdown(...)`, that inner container gets
   its own hard-coded ~16px, which is not a specificity fight -- a direct
   declaration on stMarkdownContainer simply overrides whatever inherited
   value arrives from its stChatMessageContent ancestor, regardless of
   !important placed up there. Fix: target stMarkdownContainer and its
   children directly, not just the outer wrapper. */

/* --- Bubble container: alignment, width, color, padding (this part WAS
   correctly targeted before -- stChatMessage is a real box-level element,
   not nested typography, so no inheritance issue here). --- */
[data-testid="stChatMessage"] {
    display: flex !important;
    align-items: flex-start !important; /* NOT center -- center is what vertically clips/misaligns a multi-line bubble against its background */
    margin-bottom: 10px;
    padding: 8px 12px;
    box-sizing: border-box;
    height: auto !important;
    min-height: unset !important;
    max-height: none !important;
    overflow: visible !important; /* was overflow:hidden -- that's the actual clipping bug, removed outright, not just "unless needed" */
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: fit-content !important;
    max-width: 72%;
    min-height: 36px !important;
    height: auto !important;
    padding: 8px 14px !important;
    box-sizing: border-box !important;
    overflow: visible !important;
    background: var(--chat-user-bg);
    border: 1px solid var(--chat-user-border);
    border-radius: 14px 14px 4px 14px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    margin-right: auto;
    width: fit-content !important;
    max-width: 94%;
    height: auto !important;
    background: var(--chat-assistant-bg);
    border: 1px solid var(--chat-assistant-border);
    border-radius: 14px 14px 14px 4px;
}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}

/* --- The actual root cause of this round's bug, confirmed by reading
   Streamlit's frontend source directly: stChatMessageContent's OWN default
   style is `margin: auto` (all four sides), as a flex item that also has
   flex-grow:1 -- a flex item simultaneously told to "grow to fill space"
   AND "auto-distribute margin into any extra space" is a genuine conflict,
   and every earlier round left this completely untouched (min-width was
   overridden, margin never was). Neutralizing it, plus forcing block-flow
   children so the actual text lays out in normal document flow instead of
   flex-distributed space, is the real fix -- not another blanket
   height/overflow override. --- */
[data-testid="stChatMessageContent"] {
    margin: 0 !important;
    padding: 0 !important;
    min-width: 0 !important;
    max-width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    overflow: visible !important;
    line-height: 1.35 !important;
    display: block !important;
    box-sizing: border-box;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
    font-size: 13px !important;
    line-height: 1.35 !important;
    color: var(--ink) !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    height: auto !important;
    min-height: 0 !important;
    display: block !important;
    overflow: visible !important;
    box-sizing: border-box;
    overflow-wrap: anywhere !important;
    word-break: normal !important;
    white-space: normal !important;
}
/* Scoped specifically to the user bubble's own paragraph, so this doesn't
   touch the assistant's multi-paragraph spacing (kept exactly as-is via
   the shared p/li rules further below). */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.35 !important;
    height: auto !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] span,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] div,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] em {
    font-size: 13px !important;
    line-height: 1.38 !important;
    max-width: 100% !important;
    box-sizing: border-box;
    overflow-wrap: anywhere !important;
    word-break: normal !important;
    white-space: normal !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] strong {
    font-weight: 700 !important; /* emphasis via weight only -- never larger than surrounding text */
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p {
    margin-top: 0 !important;
    margin-bottom: 5px !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0 !important; }
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] ul,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] ol {
    margin-top: 4px !important;
    margin-bottom: 5px !important;
    padding-left: 18px !important;
    max-width: 100% !important;
    box-sizing: border-box;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] li {
    margin-bottom: 2px !important;
    max-width: 100% !important;
    box-sizing: border-box;
    overflow-wrap: anywhere !important;
}
/* If a reply ever uses "## Heading" markdown rather than just "**bold**",
   pull it down to the same 13px scale so it reads as an inline label, not
   an article heading -- the single biggest thing that would otherwise
   blow past all of the above regardless of body font-size. */
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] h4 {
    font-size: 13px !important; font-weight: 700 !important; line-height: 1.3 !important;
    margin: 6px 0 2px 0 !important; color: var(--ink) !important;
}

/* --- Chat input -- kept as Streamlit's native st.chat_input (its built-in
   submit/clear handling is exactly the "state handling" left untouched),
   just restyled to sit flush against the panel and match the 13px scale.
   stChatInputTextArea is the actual <textarea> the theme-driven font-size
   issue also affects, hence the explicit !important here too. --- */
.st-key-chat_panel [data-testid="stChatInput"] {
    border: 1px solid var(--card-border) !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}
.st-key-chat_panel [data-testid="stChatInput"]:focus-within { border-color: var(--violet-600) !important; }
.st-key-chat_panel [data-testid="stChatInputTextArea"] { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


def render_result_row(rank: int, gene: str, cos_sim: float, percentile: float, annotation: str | None):
    """One structured row per protein -- rank, gene identifier, relevance,
    abundance -- separated by a thin bottom border, not an individually
    bordered/padded card. Protein description stays available via the
    expander (annotation) without cluttering the row itself."""
    # cos_sim is a cosine similarity in roughly [-1, 1]; clamp to [0, 1] for
    # the bar width so a negative (irrelevant) score doesn't render as a
    # negative-width bar.
    relevance_pct = max(0.0, min(1.0, (cos_sim + 1) / 2)) * 100
    abundance_pct = percentile * 100
    st.markdown(f"""
    <div class="result-row-item">
      <span class="result-rank">{rank:02d}</span>
      <span class="result-gene">{gene}</span>
      <div class="result-metric">
        <div class="result-metric-label">Relevance</div>
        <div class="result-metric-bar-track"><div class="result-metric-bar-fill" style="width:{relevance_pct:.1f}%; background:var(--violet-600);"></div></div>
        <div class="result-metric-value">{cos_sim:.3f} cosine similarity</div>
      </div>
      <div class="result-metric">
        <div class="result-metric-label">Abundance</div>
        <div class="result-metric-bar-track"><div class="result-metric-bar-fill" style="width:{abundance_pct:.1f}%; background:var(--blue-600);"></div></div>
        <div class="result-metric-value">{abundance_pct:.0f}th percentile</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if annotation:
        with st.expander(f"{gene} — protein description", expanded=False):
            st.caption(annotation)


def render_rank_plot(raw_abundance: dict, top_results, gene_lookup: dict, max_labels: int = 15):
    """Protein rank plot: raw abundance (log scale) vs. rank, the standard
    single-sample proteomics visualization (Schessner et al. 2022, PROTEOMICS
    -- "A practical guide to interpreting and generating bottom-up proteomics
    data visualizations"). Unlike the abundance percentile shown in each
    result card, this plots RAW abundance, not percentile -- percentile is
    derived from rank, so a percentile-vs-rank plot would just be a straight
    diagonal line and wouldn't show the field's actual point: proteomes have
    an enormous dynamic range (up to ~7 orders of magnitude), so a handful of
    proteins dominate and most sit in a long low-abundance tail.

    The query's retrieved proteins are highlighted on top of the full curve,
    making visible at a glance whether a semantically relevant protein is
    backed by real expression or is a low-abundance long-shot -- the same
    trade-off Stage 2's percentile was built to surface, shown instead of
    only tabulated.

    Log-scaling the y-axis is only valid for linear intensity values (always
    positive). CPTAC's own abundance values, and plausibly a researcher's own
    upload too, are already log2-ratio-to-reference values (can be zero or
    negative) -- log-transforming those a second time would be meaningless
    and would silently drop every non-positive protein from the plot. So the
    axis only switches to log scale when every measured value is positive;
    otherwise the values are plotted as given, on a linear axis (which is the
    correct view if they're already log-scaled).

    Returns None if there's no abundance data to plot at all.
    """
    values = {pid: v for pid, v in raw_abundance.items() if v is not None and not math.isnan(v)}
    if not values:
        return None

    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {pid: i + 1 for i, (pid, _) in enumerate(ordered)}
    x_all = list(range(1, len(ordered) + 1))
    y_all = [v for _, v in ordered]
    use_log_scale = all(v > 0 for v in y_all)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(x_all, y_all, color="#c9d3d0", linewidth=1.5, zorder=1, label="all measured proteins")

    hits = []
    for pid, _cos_sim, _pct in top_results:
        if pid not in rank_of:
            continue  # not among the plotted proteins (e.g. missing/NaN abundance)
        hits.append((rank_of[pid], values[pid], gene_lookup.get(pid, pid)))
    # hits is still in RELEVANCE order here (top_results' own order) -- cap
    # which proteins get labeled by relevance, before sorting by rank below.
    # Capping by rank instead would silently drop labels for the *lowest*-
    # abundance results specifically -- exactly the relevant-but-low-abundance
    # "long shots" this plot exists to surface -- whenever there are more
    # than max_labels results shown.
    to_label = sorted(hits[:max_labels], key=lambda h: h[0])

    if hits:
        ax.scatter([h[0] for h in hits], [h[1] for h in hits],
                    color=BRAND_VIOLET, s=32, zorder=2, label="retrieved for this query")

        # Points close together in RANK are automatically close together in
        # abundance too (abundance is monotonic in rank, by construction of
        # this plot) -- so a tight group of retrieved proteins would just get
        # individual labels stacked on top of each other and become
        # unreadable. Merge each such group into one label first (this alone
        # isn't enough once there are many groups packed into one region --
        # a fixed above/below alternation still collides between neighboring
        # groups, e.g. with top_k=20 -- so adjustText's iterative repulsion
        # does the actual layout: it looks at every label's real rendered
        # bounding box and nudges them apart from each other AND away from
        # the data points, not just from their own anchor).
        total_span = max(x_all) - min(x_all) if len(x_all) > 1 else 1
        cluster_gap = max(total_span * 0.02, 40)
        clusters = []
        for x, y, label in to_label:
            if clusters and (x - clusters[-1]["xs"][-1]) <= cluster_gap:
                clusters[-1]["xs"].append(x)
                clusters[-1]["ys"].append(y)
                clusters[-1]["labels"].append(label)
            else:
                clusters.append({"xs": [x], "ys": [y], "labels": [label]})

        texts = []
        for c in clusters:
            cx, cy = sum(c["xs"]) / len(c["xs"]), sum(c["ys"]) / len(c["ys"])
            texts.append(ax.text(cx, cy, " / ".join(c["labels"]), fontsize=7.5, color=BRAND_NAVY))
        if texts:
            adjust_text(
                texts, ax=ax,
                x=[x for c in clusters for x in c["xs"]], y=[y for c in clusters for y in c["ys"]],
                arrowprops=dict(arrowstyle="-", color=BRAND_NAVY, lw=0.6, alpha=0.6),
                expand=(1.3, 1.6),
            )

    if use_log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Abundance (log scale)", fontsize=9)
    else:
        ax.set_ylabel("Abundance (as provided)", fontsize=9)
    ax.set_xlabel("Protein rank (most → least abundant in this sample)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, use_log_scale


# ---------------------------------------------------------------------------
# Startup checks -- fail with a clear, actionable message instead of a raw
# traceback if the large data files this tool needs aren't present (they're
# not shipped via git; see the setup note in the sidebar).
# ---------------------------------------------------------------------------
missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
if missing:
    render_brand_header()
    st.markdown('<div class="hero"><p>Query your proteomics sample with natural language.</p></div>',
                unsafe_allow_html=True)
    st.error(
        "Required data files are missing, so the tool can't start:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n\nThese are large files not stored in the git repository. "
          "Ask whoever set up this project to share them with you (they should be copied "
          "into the same relative paths shown above, inside the ProteomIQ project folder)."
    )
    st.stop()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model(_device):
    """Trained projection heads (checkpoint) + frozen BioBERT for live query
    encoding -- reused from retrieve_and_annotate.py, not reimplemented."""
    return load_model(_device)


@st.cache_resource
def get_esm2_lookup():
    return load_esm2_lookup()


@st.cache_resource
def get_gene_lookup():
    return build_gene_lookup()


@st.cache_resource
def get_annotation_lookup():
    return build_annotation_lookup()


@st.cache_resource
def get_gene_to_ensp_lookup():
    return build_gene_to_ensp_lookup()


@st.cache_resource
def get_claude_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic()


with st.spinner("Loading model..."):
    protein_head, text_head, tokenizer, biobert_model = get_model(DEVICE)
    esm2_lookup = get_esm2_lookup()
    gene_lookup = get_gene_lookup()
    annotation_lookup = get_annotation_lookup()
    gene_to_ensp = get_gene_to_ensp_lookup()
    client = get_claude_client()

# ---------------------------------------------------------------------------
# Sidebar, part A: sample input controls -- has to happen before
# active_source can be resolved, which in turn determines how much intro
# content the header shows below (full onboarding vs. a compact logo-only
# header once a sample is already loaded).
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section-label" style="margin-top:0;">Sample</div>', unsafe_allow_html=True)
    st.caption("Proteomics data")
    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    use_example = st.button("Use example dataset", use_container_width=True)

# resolve which file to process: an upload takes priority; otherwise the
# example button (sticky via session_state so it survives the rerun the
# button click itself triggers)
if use_example:
    st.session_state["use_example"] = True
if uploaded is not None:
    st.session_state["use_example"] = False

active_source = None
if uploaded is not None:
    active_source = uploaded
    source_id = (uploaded.name, uploaded.size)
elif st.session_state.get("use_example"):
    active_source = EXAMPLE_CSV
    source_id = ("__example__", EXAMPLE_LABEL)

# ---------------------------------------------------------------------------
# Process the sample (data logic unchanged -- moved earlier in execution
# order than before, purely so the sidebar's dataset-status line and the
# header's full-vs-compact decision can both use the result below).
# ---------------------------------------------------------------------------
if active_source is not None:
    if st.session_state.get("file_id") != source_id:
        with st.spinner("Processing sample..."):
            try:
                result_df, unmatched, n_ambiguous = normalize_uploaded_sample(active_source, "gene", "abundance")
            except Exception as e:
                st.error(
                    f"Couldn't read this file as a proteomics sample: {e}\n\n"
                    "Make sure it's a CSV with a `gene` column and an `abundance` column, one row per protein."
                )
                st.stop()
            measured_proteins = dict(zip(result_df["ENSP_ID"], result_df["percentile"]))
            raw_abundance = dict(zip(result_df["ENSP_ID"], result_df["raw_abundance"]))

        st.session_state["file_id"] = source_id
        st.session_state["measured_proteins"] = measured_proteins
        st.session_state["raw_abundance"] = raw_abundance
        st.session_state["match_info"] = (len(result_df), n_ambiguous, len(unmatched))
        st.session_state["last_query"] = None
        st.session_state.pop("scored", None)
        st.session_state["chat_messages"] = []
        st.session_state["chat_searches"] = []
        st.session_state["chat_lookups"] = {}
        st.session_state.pop("chat_system_prompt", None)

    measured_proteins = st.session_state["measured_proteins"]
    raw_abundance = st.session_state["raw_abundance"]
    n_matched, n_ambiguous, n_unmatched = st.session_state["match_info"]
    sample_label = source_id[1] if isinstance(active_source, Path) else source_id[0]

# ---------------------------------------------------------------------------
# Sidebar, part B: dataset status (once loaded) + results controls + CSV
# format reference. A script can re-enter `with st.sidebar:` more than
# once -- Streamlit just appends each block in the order it's written, so
# this lands below the upload controls from part A above.
# ---------------------------------------------------------------------------
with st.sidebar:
    if active_source is not None:
        st.markdown(
            f'<div class="dataset-status"><div class="dataset-status-label">{sample_label}</div>'
            f'<div class="dataset-status-count">{n_matched:,} proteins</div></div>',
            unsafe_allow_html=True,
        )
        if n_ambiguous or n_unmatched:
            st.caption(
                f"{n_ambiguous} dropped as ambiguous (gene name appeared more than once in the file), "
                f"{n_unmatched} not in our reference protein set and skipped."
            )
        st.markdown("---")

    st.markdown('<div class="sidebar-section-label">Results</div>', unsafe_allow_html=True)
    st.caption("Number of proteins")
    top_k = st.slider(
        "Number of results to show", min_value=5, max_value=50, value=DEFAULT_TOP_K,
        label_visibility="collapsed",
    )

    with st.expander("Expected CSV format", expanded=False):
        st.markdown(
            "```\n"
            "gene,abundance\n"
            "EGFR,1.23\n"
            "KRAS,0.45\n"
            "MET,2.67\n"
            "```\n"
            "One row per protein: gene name + raw abundance value from your own measurement."
        )
    if client is None:
        st.warning(
            "No Claude API key configured — Stage 3 interpretation is disabled. "
            "Set the `ANTHROPIC_API_KEY` environment variable and restart to enable it. "
            "Retrieval and abundance results still work without it."
        )

# ---------------------------------------------------------------------------
# Header -- full onboarding (logo, headline, explainer) only before a
# sample is loaded. Once one is loaded, the intro collapses to just the
# logo, so the search/results stay the dominant part of the screen and
# nobody has to scroll past onboarding content on every search.
# ---------------------------------------------------------------------------
render_brand_header()

if active_source is None:
    st.markdown("""
    <div class="hero">
      <h1>Explore proteins through natural language</h1>
      <p>Search proteins measured in your proteomics sample using semantic similarity
      between protein and text embeddings.</p>
      <div class="hero-accent"></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How does ProteomIQ work?"):
        st.markdown("""
- **Semantic relevance** (purple bar) — how closely a protein's known biological function matches
  your question, measured by a model trained to connect protein sequences with natural-language
  descriptions of protein function. Score range: **cosine similarity**, roughly -1 (unrelated) to 1
  (closely related, this model was seen to be around 0.3–0.4 for a strong match).
- **Abundance** (blue bar) — how highly this protein was measured relative to every other protein
  in *this specific sample*, as a percentile. This is relative within the sample, not an absolute
  concentration, and not a statement about whether the protein is "overexpressed" — a naturally
  abundant protein (e.g. albumin) will always rank high regardless of regulation.
- **Ranking** is by semantic relevance only — abundance is shown for context, not used to reorder
  or filter results, so a highly relevant but low-abundance protein still shows up.
- Only proteins that were actually measured in the uploaded sample are searched — not the whole
  human proteome — so every result is something a researcher can actually follow up on in this
  patient.

**What this tool is, and isn't.** ProteomIQ explores *one sample at a time*, interactively, in
natural language — closer to how a clinician reviews a single patient's profile than to a
population-level study. It is **not** a substitute for group-level differential expression
analysis (tools like GSEA or limma, which compare many samples across conditions with proper
statistical testing) — those remain the right choice when comparing cohorts or claiming a protein
is significantly up- or down-regulated. Treat ProteomIQ's output as a starting point for
hypothesis generation, not a validated finding.
        """)

    st.markdown(
        '<div class="minimal-empty-title">Upload a proteomics sample to begin</div>'
        '<div class="minimal-empty-hint">Select a CSV file from the sidebar or use the example dataset.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Steps 2-4, side by side: query+results on the left as the main content,
# the chat as a persistent panel on the right -- CellWhisperer-style layout
# (a primary view + a dedicated "chat protocol" panel beside it), rather
# than the chat stacked inline below everything else.
# ---------------------------------------------------------------------------
main_col, chat_col = st.columns([1, 1], gap="large")

with main_col:
    # Search -- the main focus of the page once a sample is loaded. Styled
    # via the .st-key-main_query CSS rule above (larger type, brand focus
    # ring) so this is the visually dominant control on the page. The
    # "Search" button is an additional, purely visual trigger -- clicking
    # it just causes Streamlit's normal rerun, which re-evaluates the same
    # last_query comparison below exactly as pressing Enter already does;
    # no new state-handling logic, so the existing behavior is unchanged.
    render_section_header("Protein Search", "Find proteins in this sample using natural language.")
    search_input_col, search_button_col = st.columns([5, 1])
    with search_input_col:
        query = st.text_input(
            "Question", label_visibility="collapsed", key="main_query",
            placeholder="Describe a biological function, process, pathway or phenotype...",
        )
    with search_button_col:
        st.button("Search", use_container_width=True)
    st.caption("Examples: receptor tyrosine kinase signaling · immune evasion in lung cancer · DNA damage repair")

    if not query:
        st.stop()

    if st.session_state.get("last_query") != query:
        with st.spinner("Searching this sample's proteins..."):
            scored = retrieve_within_sample(
                query, measured_proteins, esm2_lookup,
                protein_head, text_head, tokenizer, biobert_model, DEVICE,
            )
        st.session_state["scored"] = scored
        st.session_state["last_query"] = query
        st.session_state["chat_messages"] = []
        st.session_state["chat_searches"] = []
        st.session_state["chat_lookups"] = {}
        st.session_state.pop("chat_system_prompt", None)
    else:
        scored = st.session_state["scored"]

    top = scored[:top_k]
    render_section_header("Results", f"{len(top)} proteins ranked by semantic similarity")

    for i, (pid, cos_sim, pct) in enumerate(top, start=1):
        gene = gene_lookup.get(pid, pid)
        annotation = annotation_lookup.get(pid)
        render_result_row(i, gene, cos_sim, pct, annotation)

    st.markdown("##### Where these proteins sit in the sample's abundance profile")
    rank_plot_result = render_rank_plot(raw_abundance, top, gene_lookup)
    if rank_plot_result is not None:
        rank_fig, used_log_scale = rank_plot_result
        st.pyplot(rank_fig, use_container_width=True)
        if used_log_scale:
            scale_clause = " on a log scale, since proteomes span several orders of magnitude,"
        else:
            scale_clause = ""
        st.caption(
            f"The standard single-sample proteomics view: every measured protein ranked by "
            f"abundance{scale_clause} most abundant on the left. Highlighted points are the "
            "proteins shown above, so you can see whether a relevant protein is backed by real "
            "expression or is a low-abundance long-shot."
        )

# Step 4: interpretation -- a real conversation with Claude about this
# sample, rendered as a persistent panel beside the results (not stacked
# below them) so the retrieved proteins stay in view while chatting about
# them. Stage 1/2 retrieval logic is unchanged; only this layer is
# multi-turn AND can trigger additional retrievals mid-conversation via the
# search box below, so a follow-up that needs different data doesn't have to
# restart the whole chat. Deliberately NOT agentic -- Claude never decides on
# its own to run a new search; the researcher always explicitly triggers one
# (same "researcher brings the question" principle Step 2 already follows,
# just extended to every search in the session, not only the first).
# The very first turn still requires an explicit click (Claude API calls
# cost money and take a few seconds, so this shouldn't fire on every rerun).
with chat_col:
    # A real st.container(key=...), not the raw-HTML "open a <div>, close it
    # in a later st.markdown call" hack this used before. That hack doesn't
    # actually work the way it looks like it should: each st.markdown() call
    # renders its own isolated HTML fragment, so an unclosed <div> in one
    # call doesn't stay open across the separate elements rendered after it
    # -- it just gets auto-closed by the browser immediately, as an empty
    # div, which is exactly the stray rounded empty box that was appearing
    # above the Chat heading. st.container(key=...) genuinely nests
    # everything inside it in one real DOM container, and gets a stable
    # `st-key-chat_panel` class we can style (see CSS above), so this both
    # removes the bug and makes the bordered panel actually work correctly
    # for the first time.
    with st.container(key="chat_panel"):
        render_section_header("Chat", "Ask follow-up questions about these results, or search for something new.")

        if client is None:
            st.caption("Unavailable — no Claude API key configured (see sidebar).")
        elif not st.session_state.get("chat_messages"):
            if st.button("Interpret with Claude", type="primary"):
                chat_searches = [{"query": query, "selected": scored[:top_k]}]
                system_prompt = build_chat_system_prompt(chat_searches, gene_lookup, annotation_lookup)
                opening_question = "What's your interpretation of these results?"
                with st.spinner("Asking Claude..."):
                    try:
                        reply = stage3_chat(client, system_prompt, [{"role": "user", "content": opening_question}])
                        st.session_state["chat_searches"] = chat_searches
                        st.session_state["chat_system_prompt"] = system_prompt
                        st.session_state["chat_messages"] = [
                            {"role": "user", "content": opening_question},
                            {"role": "assistant", "content": reply},
                        ]
                    except Exception as e:
                        st.error(f"Claude request failed: {e}")
        else:
            chat_history = st.container(height=600)
            with chat_history:
                for msg in st.session_state["chat_messages"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            follow_up = st.chat_input("Ask about these results...")
            if follow_up:
                st.session_state["chat_messages"].append({"role": "user", "content": follow_up})

                # Auto-look-up any gene the researcher explicitly named in their own
                # message (e.g. "resistant to EGFR inhibitors?") -- deterministic
                # string matching, no LLM judgment call, so a protein missing from
                # the retrieved set no longer means Claude has to say "I have no
                # data" when it could just check whether that named protein was
                # actually measured in this sample.
                mentioned = extract_gene_mentions(follow_up, gene_to_ensp)
                new_mentions = [g for g in mentioned if g not in st.session_state["chat_lookups"]]
                if new_mentions:
                    for gene in new_mentions:
                        st.session_state["chat_lookups"][gene] = lookup_gene(
                            gene, gene_to_ensp, measured_proteins, annotation_lookup
                        )
                    st.session_state["chat_system_prompt"] = build_chat_system_prompt(
                        st.session_state["chat_searches"], gene_lookup, annotation_lookup,
                        lookups=st.session_state["chat_lookups"],
                    )

                with st.spinner("Asking Claude..."):
                    try:
                        reply = stage3_chat(client, st.session_state["chat_system_prompt"], st.session_state["chat_messages"])
                        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.session_state["chat_messages"].pop()  # drop the unanswered turn so a retry doesn't break role alternation
                        st.error(f"Claude request failed: {e}")
                st.rerun()

            with st.expander("Search for something else in this sample"):
                with st.form("new_search_form", clear_on_submit=True):
                    new_search_query = st.text_input(
                        "New search query", label_visibility="collapsed",
                        placeholder="e.g. cell cycle regulation",
                    )
                    new_search_submitted = st.form_submit_button("Search")

                if new_search_submitted and new_search_query:
                    with st.spinner(f'Searching for "{new_search_query}"...'):
                        new_scored = retrieve_within_sample(
                            new_search_query, measured_proteins, esm2_lookup,
                            protein_head, text_head, tokenizer, biobert_model, DEVICE,
                        )
                    st.session_state["chat_searches"].append(
                        {"query": new_search_query, "selected": new_scored[:top_k]}
                    )
                    st.session_state["chat_system_prompt"] = build_chat_system_prompt(
                        st.session_state["chat_searches"], gene_lookup, annotation_lookup,
                        lookups=st.session_state["chat_lookups"],
                    )
                    st.session_state["chat_messages"].append(
                        {"role": "user", "content": f'New search: "{new_search_query}"'}
                    )
                    with st.spinner("Asking Claude..."):
                        try:
                            reply = stage3_chat(
                                client, st.session_state["chat_system_prompt"], st.session_state["chat_messages"]
                            )
                            st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.session_state["chat_messages"].pop()
                            st.error(f"Claude request failed: {e}")
                    st.rerun()

st.markdown("---")
st.caption(
    "ProteomIQ is a research prototype for hypothesis generation, not a diagnostic tool. "
    "See the accompanying thesis for methodology, evaluation, and known limitations."
)
