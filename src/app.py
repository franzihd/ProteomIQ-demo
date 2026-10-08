import base64
import html
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
FINAL_LOGO_PATH = ASSETS_DIR / "final_logo.png"  # the ProteomIQ logo (magnifier + wordmark + tagline)
# final_logo.png with its large transparent margins cropped off (and
# downscaled to 1200px wide), so the header can show the logo large without
# wasting vertical space. Regenerate from final_logo.png if the logo changes.
LOGO_HEADER_PATH = ASSETS_DIR / "final_logo_header.png"

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


def _load_logo_data_uri():
    """Data-URI-embedded wordmark for the header. Streamlit has no built-in
    static-asset serving, so embedding as a data URI inside custom HTML is
    the standard way to show a local image without a separate file server.
    Prefers the cropped header PNG, then the uncropped final_logo.png, then
    the old logo.jpg.
    Returns None if neither asset exists, so callers can fall back to a
    plain-text title instead of a broken <img> tag."""
    for path, mime in ((LOGO_HEADER_PATH, "image/png"), (FINAL_LOGO_PATH, "image/png"), (LOGO_PATH, "image/jpeg")):
        try:
            with open(path, "rb") as f:
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
        except Exception:
            continue
    return None


WORKFLOW_STEPS = ("Load sample", "Search", "Inspect results", "Interpret")


def render_brand_header(step_states=None):
    """Application header: the ProteomIQ wordmark as the visual anchor on the
    left and, once the app is running, a compact 4-step workflow indicator
    on the right (load -> search -> inspect -> interpret). Shared by the
    normal header and the early missing-data-files error path (which passes
    no step_states, so only the logo shows). Falls back to a plain text
    title if no logo asset is present -- no icon/emoji."""
    logo_uri = _load_logo_data_uri()
    if logo_uri:
        brand = f'<img class="brand-wordmark" src="{logo_uri}" alt="ProteomIQ — proteins meet language">'
    else:
        brand = '<div class="brand-fallback">ProteomIQ</div>'

    steps_html = ""
    if step_states:
        items = []
        for i, (label, state) in enumerate(zip(WORKFLOW_STEPS, step_states), start=1):
            num = "✓" if state == "done" else str(i)
            optional = '<span class="step-optional">optional</span>' if i == len(WORKFLOW_STEPS) else ""
            items.append(
                f'<li class="step step-{state}"><span class="step-num">{num}</span>'
                f'<span class="step-label">{label}{optional}</span></li>'
            )
        steps_html = f'<ol class="workflow" aria-label="Workflow">{"".join(items)}</ol>'

    st.markdown(f'<div class="app-header">{brand}{steps_html}</div>', unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "", eyebrow: str = ""):
    """One consistent section-header pattern (optional small uppercase
    eyebrow, bold title, small muted subtitle) used for every content
    section, so the page reads as one coherent application."""
    eyebrow_html = f'<div class="section-eyebrow">{eyebrow}</div>' if eyebrow else ""
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-header">{eyebrow_html}<div class="section-title">{title}</div>{sub_html}</div>',
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
   when a page-wide font-family is set. */
input, textarea, button, select,
[data-testid="stSidebar"], [data-testid="stExpander"], [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', sans-serif;
}

/* Brand palette + design tokens. Semantic convention (kept consistent
   throughout): blue = protein / sample abundance / biological data,
   purple = language / semantic similarity / AI, navy = structure,
   headings, primary text, light gray = backgrounds / secondary UI.
   One radius scale, one border color, one shadow -- reused everywhere. */
:root {
    --navy-900: #14213D;
    --blue-600: #2A7BCB;
    --blue-50: #EEF5FC;
    --violet-600: #7B5CD6;
    --violet-700: #6847C4;
    --violet-50: #F4F0FD;
    --brand-100: #F5F7FA;
    --ink: #14213D;
    --muted: #5B6477;
    --subtle: #8A93A6;
    --card-bg: #ffffff;
    --card-border: #E4E7ED;
    --border-strong: #CDD3DE;
    --track: #EDF0F5;
    --radius-sm: 6px;
    --radius: 10px;
    --shadow-sm: 0 1px 2px rgba(20, 33, 61, 0.05);
    --chat-user-bg: #F3EEFC;
    --chat-user-border: #E7DDF8;
    --chat-assistant-bg: #F5F8FC;
    --chat-assistant-border: #E3E8F0;
}

/* Main workspace: capped width so very wide monitors don't stretch rows
   into unreadable lines, centered. Top padding must clear Streamlit's
   fixed top bar (on Streamlit Community Cloud it carries Share/GitHub
   icons and is opaque) -- less than ~3.5rem clips the top of the logo. */
[data-testid="stMainBlockContainer"] {
    padding-top: 3.75rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1560px;
    margin: 0 auto;
}

/* ---------- Header: wordmark (visual anchor) + workflow indicator ---------- */
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 1rem 2rem;
    padding: 0.35rem 0 1.15rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--card-border);
}
.brand-wordmark { height: clamp(56px, 5.6vw, 80px); width: auto; display: block; }
.brand-fallback { font-size: 2rem; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }

.workflow {
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem;
    list-style: none; margin: 0; padding: 0;
}
.workflow .step {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.8rem; color: var(--subtle); white-space: nowrap;
}
.workflow .step + .step::before {
    content: ""; width: 1.4rem; height: 1px; background: var(--border-strong);
    margin-right: 0.35rem;
}
.workflow .step-num {
    width: 1.35rem; height: 1.35rem; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    border: 1px solid var(--border-strong); color: var(--subtle); background: #fff;
}
.workflow .step-done { color: var(--ink); }
.workflow .step-done .step-num { background: var(--navy-900); border-color: var(--navy-900); color: #fff; }
.workflow .step-current { color: var(--ink); font-weight: 600; }
.workflow .step-current .step-num { border-color: var(--violet-600); color: var(--violet-600); background: var(--violet-50); }
.workflow .step-optional { margin-left: 0.3rem; font-size: 0.68rem; font-weight: 400; color: var(--subtle); }

/* ---------- Empty state (no sample loaded yet) ---------- */
.hero { margin: 0.5rem 0 1.5rem 0; max-width: 44rem; }
.hero h1 {
    font-family: 'Inter', sans-serif !important;
    margin: 0 0 0.5rem 0; padding: 0 !important; font-size: 1.9rem; font-weight: 700;
    color: var(--ink); line-height: 1.2; letter-spacing: -0.015em;
}
[data-testid="stMarkdownContainer"] .hero p { margin: 0; font-size: 1rem; color: var(--muted); line-height: 1.55; }
.start-card {
    max-width: 44rem; margin: 0 0 1rem 0; padding: 1.1rem 1.25rem;
    background: var(--brand-100); border: 1px solid var(--card-border);
    border-left: 3px solid var(--blue-600); border-radius: var(--radius);
}
.start-title { font-size: 0.98rem; font-weight: 700; color: var(--ink); margin-bottom: 0.2rem; }
.start-hint { font-size: 0.87rem; color: var(--muted); line-height: 1.5; }
.start-hint code { font-size: 0.82rem; background: #fff; border: 1px solid var(--card-border); border-radius: 4px; padding: 0 0.3rem; color: var(--ink); }

/* ---------- Section headers ---------- */
.section-header { margin: 0 0 0.8rem 0; }
.section-eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--violet-600); margin-bottom: 0.2rem;
}
.section-title { font-size: 1.15rem; font-weight: 700; color: var(--ink); margin: 0 0 0.15rem 0; letter-spacing: -0.005em; }
.section-sub { font-size: 0.85rem; color: var(--muted); line-height: 1.45; }

/* ---------- Search bar: [ large input ][ Search ] ---------- */
.st-key-search_bar { gap: 0.6rem !important; }
.st-key-main_query [data-baseweb="input"] {
    min-height: 48px;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-strong) !important;
    background: #fff !important;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.st-key-main_query [data-baseweb="input"] > div { background: transparent !important; }
.st-key-main_query [data-baseweb="input"]:focus-within {
    border-color: var(--violet-600) !important;
    box-shadow: 0 0 0 3px rgba(123, 92, 214, 0.15) !important;
}
.st-key-main_query input {
    font-size: 1rem !important;
    padding: 0 1rem !important;
    color: var(--ink) !important;
}
.st-key-search_btn button {
    min-height: 48px; min-width: 7.5rem;
    border-radius: var(--radius) !important;
}
.search-examples {
    margin: 0.55rem 0 0 0; font-size: 0.78rem; color: var(--subtle); line-height: 1.6;
}
.search-examples .ex { color: var(--muted); }
.search-examples .ex + .ex::before { content: "·"; margin: 0 0.45rem; color: var(--border-strong); }
.search-spacer { height: 1.6rem; }

/* ---------- Results header ---------- */
.results-head { display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap; margin: 0 0 0.15rem 0; }
.results-head .section-title { margin: 0; }
.count-badge {
    font-size: 0.75rem; font-weight: 600; color: var(--violet-700);
    background: var(--violet-50); border: 1px solid #E4DBF8;
    border-radius: 999px; padding: 0.08rem 0.55rem;
}
.results-sub { font-size: 0.85rem; color: var(--muted); margin: 0 0 0.75rem 0; }
.results-sub .q { color: var(--ink); font-weight: 600; }

/* ---------- Result list: one coherent row per protein ----------
   Each protein is a single <details> element: the summary row holds rank,
   gene, relevance and abundance; opening it reveals the UniProt function
   text inside the same bordered row, so the description visibly belongs
   to its protein. The list is a container-query context so the row grid
   can reflow when the center column gets narrow, independent of the
   overall viewport width. */
.result-list {
    container-type: inline-size;
    border: 1px solid var(--card-border); border-radius: var(--radius);
    background: var(--card-bg); box-shadow: var(--shadow-sm);
    overflow: hidden;
}
.result-grid {
    display: grid;
    grid-template-columns: 2rem minmax(5.5rem, 0.8fr) minmax(9rem, 1.35fr) minmax(8rem, 1fr) 1rem;
    align-items: center; column-gap: 1.1rem;
}
.result-colhead {
    padding: 0.55rem 1rem; background: var(--brand-100);
    border-bottom: 1px solid var(--card-border);
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    color: var(--subtle);
}
.result-colhead .h-rel { color: var(--violet-600); }
.result-colhead .h-ab { color: var(--blue-600); }
.result-colhead .h-unit { text-transform: none; letter-spacing: 0; font-weight: 400; color: var(--subtle); }

.result { border-bottom: 1px solid var(--card-border); }
.result:last-child { border-bottom: none; }
.result > summary {
    list-style: none; cursor: pointer;
    padding: 0.62rem 1rem;
    transition: background 0.12s ease;
}
.result > summary::-webkit-details-marker { display: none; }
.result > summary:hover { background: #FAFBFD; }
.result[open] > summary { background: var(--brand-100); }
.result.no-detail > summary { cursor: default; }
.result.no-detail > summary:hover { background: transparent; }

.r-rank { font-size: 0.78rem; color: var(--subtle); font-variant-numeric: tabular-nums; }
.r-gene { font-weight: 700; font-size: 0.97rem; color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.r-metric { display: flex; align-items: center; gap: 0.6rem; min-width: 0; }
.r-bar { flex: 1; height: 6px; border-radius: 999px; background: var(--track); overflow: hidden; min-width: 2.5rem; }
.r-fill { display: block; height: 100%; border-radius: 999px; }
.r-rel .r-fill { background: var(--violet-600); }
.r-ab .r-bar { height: 4px; }
.r-ab .r-fill { background: var(--blue-600); }
.r-val { font-size: 0.8rem; font-variant-numeric: tabular-nums; white-space: nowrap; min-width: 3.1rem; text-align: right; }
.r-rel .r-val { font-weight: 600; color: var(--violet-700); }
.r-ab .r-val { color: var(--muted); }
.r-inline-label { display: none; }
.r-chev {
    width: 0.5rem; height: 0.5rem; justify-self: end;
    border-right: 1.5px solid var(--subtle); border-bottom: 1.5px solid var(--subtle);
    transform: rotate(-45deg); transition: transform 0.15s ease;
}
.result[open] .r-chev { transform: rotate(45deg); }

.result-detail {
    padding: 0.1rem 1rem 0.85rem calc(1rem + 2rem + 1.1rem);
    background: var(--brand-100);
}
.detail-label {
    font-size: 0.66rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--blue-600); margin-bottom: 0.25rem;
}
.result-detail p { margin: 0; font-size: 0.84rem; line-height: 1.55; color: var(--ink); max-width: 62rem; }

/* Narrow center column: gene stays on the first line, the two metrics
   drop below it (each with its own small inline label). */
@container (max-width: 560px) {
    .result-grid { grid-template-columns: 1.6rem 1fr 1rem; row-gap: 0.4rem; }
    .result-colhead { display: none; }
    .r-rank { grid-row: 1; grid-column: 1; }
    .r-gene { grid-row: 1; grid-column: 2; }
    .r-chev { grid-row: 1; grid-column: 3; }
    .r-rel { grid-row: 2; grid-column: 2 / 4; }
    .r-ab { grid-row: 3; grid-column: 2 / 4; }
    .r-inline-label { display: inline; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--subtle); width: 4.8rem; flex-shrink: 0; }
    .result-detail { padding-left: calc(1rem + 1.6rem + 1.1rem); }
}

/* ---------- Abundance-profile plot card ---------- */
.st-key-plot_card {
    border: 1px solid var(--card-border); border-radius: var(--radius);
    background: #fff; box-shadow: var(--shadow-sm);
    padding: 1rem 1.1rem 0.6rem 1.1rem;
}
.section-gap { height: 1.75rem; }

/* ---------- AI interpretation: call-to-action card (before) ---------- */
.st-key-ai_cta {
    border: 1px solid var(--card-border); border-radius: var(--radius);
    background: linear-gradient(180deg, var(--violet-50) 0%, #ffffff 70%);
    box-shadow: var(--shadow-sm);
    padding: 1.15rem 1.2rem 1rem 1.2rem;
}
.ai-title { font-size: 1.08rem; font-weight: 700; color: var(--ink); margin: 0 0 0.35rem 0; }
[data-testid="stMarkdownContainer"] p.ai-text { font-size: 0.86rem; color: var(--muted); line-height: 1.55; margin: 0 0 0.35rem 0; }
.ai-points { margin: 0.2rem 0 0.4rem 0; padding: 0; list-style: none; }
[data-testid="stMarkdownContainer"] .ai-points li { margin: 0; font-size: 0.82rem; color: var(--ink); padding: 0.18rem 0 0.18rem 1.1rem; position: relative; line-height: 1.45; }
.ai-points li::before {
    content: ""; position: absolute; left: 0.1rem; top: 0.62rem;
    width: 0.38rem; height: 0.38rem; border-radius: 50%; background: var(--violet-600);
}
.ai-footnote { font-size: 0.74rem; color: var(--subtle); margin-top: 0.15rem; line-height: 1.45; }
.ai-badge {
    display: inline-block; vertical-align: middle; margin-left: 0.45rem;
    font-size: 0.68rem; font-weight: 600; color: var(--violet-700);
    background: var(--violet-50); border: 1px solid #E4DBF8; border-radius: 999px; padding: 0.05rem 0.5rem;
}

/* ---------- AI interpretation: chat panel (after) ---------- */
.st-key-chat_panel {
    background: #fff;
    border: 1px solid var(--card-border);
    border-top: 3px solid var(--violet-600);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    padding: 1rem 1.1rem 0.75rem 1.1rem;
}

/* ---------- Responsiveness of the results | AI two-column layout ----------
   Streamlit only stacks columns below a ~640px viewport; between that and
   a comfortable desktop width the AI column would get squeezed into an
   unreadably narrow strip. Giving both columns a minimum width and
   letting the row wrap moves the AI column below the results instead.
   Scoped via :has() to only the block that contains the AI column. */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-ai_cta),
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-chat_panel) {
    flex-wrap: wrap;
}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-ai_cta) > [data-testid="stColumn"]:first-child,
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-chat_panel) > [data-testid="stColumn"]:first-child {
    min-width: min(100%, 22rem);
}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-ai_cta) > [data-testid="stColumn"]:last-child,
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-chat_panel) > [data-testid="stColumn"]:last-child {
    min-width: min(100%, 16rem);
}

/* ---------- Sidebar: the analysis control panel ---------- */
[data-testid="stSidebar"] {
    background: var(--brand-100);
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
.sidebar-section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    color: var(--muted); text-transform: uppercase;
    margin: 1.1rem 0 0.4rem 0;
}
.sidebar-section-label.first { margin-top: 0; }
.sidebar-or {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.72rem; color: var(--subtle); margin: 0.1rem 0 0.1rem 0;
}
.sidebar-or::before, .sidebar-or::after { content: ""; flex: 1; height: 1px; background: var(--card-border); }

.dataset-card {
    background: #fff; border: 1px solid var(--card-border);
    border-left: 3px solid var(--blue-600);
    border-radius: var(--radius); padding: 0.75rem 0.85rem 0.8rem 0.85rem;
    box-shadow: var(--shadow-sm);
}
.dc-eyebrow {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: var(--blue-600);
}
.dc-eyebrow .dot { width: 0.42rem; height: 0.42rem; border-radius: 50%; background: var(--blue-600); }
.dc-name {
    font-size: 0.9rem; font-weight: 600; color: var(--ink); margin: 0.25rem 0 0.6rem 0;
    line-height: 1.35; overflow-wrap: anywhere;
}
.dc-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; }
.dc-stat.main { grid-column: 1 / -1; display: flex; align-items: baseline; gap: 0.4rem; }
.dc-stat.main b { font-size: 1.15rem; }
.dc-stat.main span { font-size: 0.75rem; }
.dc-stat { background: var(--brand-100); border-radius: var(--radius-sm); padding: 0.4rem 0.45rem; min-width: 0; }
.dc-stat b { display: block; font-size: 0.92rem; color: var(--ink); font-variant-numeric: tabular-nums; }
.dc-stat span { display: block; font-size: 0.68rem; color: var(--muted); line-height: 1.25; }
.dc-stat.main b { color: var(--blue-600); }
.dc-note { font-size: 0.7rem; color: var(--subtle); line-height: 1.45; margin-top: 0.55rem; }

[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: var(--radius) !important;
    padding: 0.6rem !important;
}

/* ---------- Native widget polish (buttons, expanders, alerts) ----------
   Scoped by Streamlit's per-kind data-testids, so primary and secondary
   buttons keep distinct looks. white-space: nowrap on every button label
   is what guarantees "Search" etc. never wrap onto two lines. */
[data-testid^="stBaseButton"] { white-space: nowrap; }
[data-testid^="stBaseButton"] p { white-space: nowrap; }
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
    border-radius: 8px !important;
    border: 1px solid var(--border-strong) !important;
    background: #fff !important;
    color: var(--ink) !important;
    font-weight: 500;
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    border-color: var(--violet-600) !important;
    color: var(--violet-700) !important;
    background: var(--violet-50) !important;
}
[data-testid="stBaseButton-primary"] {
    border-radius: 8px !important;
    background: var(--violet-600) !important;
    border: 1px solid var(--violet-600) !important;
    color: #fff !important;
    font-weight: 600;
    transition: background 0.15s ease, border-color 0.15s ease;
}
[data-testid="stBaseButton-primary"]:hover {
    background: var(--violet-700) !important;
    border-color: var(--violet-700) !important;
}
[data-testid="stBaseButton-primary"]:disabled {
    background: var(--track) !important; border-color: var(--card-border) !important; color: var(--subtle) !important;
}

[data-testid="stAlertContainer"] {
    border-radius: var(--radius) !important;
}

[data-testid="stExpander"] details {
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
    background: #fff;
    overflow: hidden;
}
[data-testid="stExpander"] summary { font-size: 0.86rem; font-weight: 500; color: var(--ink); }
[data-testid="stExpander"] summary:hover { color: var(--violet-700); }
[data-testid="stExpander"] summary:hover svg { fill: var(--violet-700); }
[data-testid="stSidebar"] [data-testid="stExpander"] p { font-size: 0.8rem; line-height: 1.45; color: var(--muted); }
[data-testid="stSidebar"] [data-testid="stExpander"] code { font-size: 0.78rem; }
.st-key-howto { max-width: 44rem; }

.app-footer {
    margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--card-border);
    font-size: 0.76rem; color: var(--subtle); line-height: 1.5;
}

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


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _result_item_html(rank: int, gene: str, cos_sim: float, percentile: float, annotation: str | None) -> str:
    """One protein as one coherent row: rank, gene identifier, semantic
    relevance (purple, primary), abundance (blue, secondary). Rendered as a
    native <details> element, so clicking the row reveals the UniProt
    function text inside the same row instead of a separate widget."""
    # cos_sim is a cosine similarity in roughly [-1, 1]; clamp to [0, 1] for
    # the bar width so a negative (irrelevant) score doesn't render as a
    # negative-width bar.
    relevance_pct = max(0.0, min(1.0, (cos_sim + 1) / 2)) * 100
    abundance_pct = percentile * 100
    gene_html = html.escape(gene)
    summary = (
        f'<summary class="result-grid">'
        f'<span class="r-rank">{rank:02d}</span>'
        f'<span class="r-gene" title="{gene_html}">{gene_html}</span>'
        f'<span class="r-metric r-rel" title="Cosine similarity between this protein and your query">'
        f'<span class="r-inline-label">Relevance</span>'
        f'<span class="r-bar"><span class="r-fill" style="width:{relevance_pct:.1f}%"></span></span>'
        f'<span class="r-val">{cos_sim:.3f}</span></span>'
        f'<span class="r-metric r-ab" title="Abundance percentile within this sample">'
        f'<span class="r-inline-label">Abundance</span>'
        f'<span class="r-bar"><span class="r-fill" style="width:{abundance_pct:.1f}%"></span></span>'
        f'<span class="r-val">{_ordinal(round(abundance_pct))} pct</span></span>'
        f'<span class="{"r-chev" if annotation else ""}"></span>'
        f'</summary>'
    )
    if not annotation:
        return f'<details class="result no-detail">{summary}</details>'
    # Annotations are stored as "GENE: function text" -- the gene is already
    # the row's own heading, so don't repeat it in the detail panel.
    text = annotation
    if text.startswith(f"{gene}:"):
        text = text[len(gene) + 1:].strip()
    return (
        f'<details class="result">{summary}'
        f'<div class="result-detail"><div class="detail-label">UniProt function</div>'
        f'<p>{html.escape(text)}</p></div></details>'
    )


def render_result_list(rows):
    """The ranked result list as one bordered card: a column-header row
    (which also serves as the legend for what purple vs. blue means),
    followed by one row per protein. rows = [(rank, gene, cos_sim,
    percentile, annotation), ...]. Rendered as a single HTML block, which is
    also much faster than one Streamlit element per row."""
    head = (
        '<div class="result-grid result-colhead">'
        '<span>#</span><span>Protein</span>'
        '<span class="h-rel">Semantic relevance <span class="h-unit">· cosine</span></span>'
        '<span class="h-ab">Abundance <span class="h-unit">· percentile</span></span>'
        '<span></span></div>'
    )
    items = "".join(_result_item_html(*row) for row in rows)
    st.markdown(f'<div class="result-list">{head}{items}</div>', unsafe_allow_html=True)


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
    ax.plot(x_all, y_all, color="#C5CCD8", linewidth=1.5, zorder=1, label="all measured proteins")

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
    st.markdown('<div class="sidebar-section-label first">1 · Proteomics sample</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    st.markdown('<div class="sidebar-or">or</div>', unsafe_allow_html=True)
    use_example = st.button("Use example dataset", width="stretch")
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
    # Filled in further down, once the sample has been processed -- keeps
    # the loaded-dataset summary directly under the upload controls.
    dataset_status_slot = st.empty()

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
        if isinstance(active_source, Path):
            card_eyebrow, card_name = "Example dataset loaded", "LUAD patient 11LU013 · CPTAC"
        else:
            card_eyebrow, card_name = "Sample loaded", sample_label
        dataset_status_slot.markdown(
            f'<div class="dataset-card">'
            f'<div class="dc-eyebrow"><span class="dot"></span>{card_eyebrow}</div>'
            f'<div class="dc-name" title="{html.escape(str(sample_label))}">{html.escape(str(card_name))}</div>'
            f'<div class="dc-stats">'
            f'<div class="dc-stat main" title="Proteins matched to the reference set and searchable">'
            f'<b>{n_matched:,}</b><span>matched proteins</span></div>'
            f'<div class="dc-stat" title="Dropped: gene name appeared more than once in the file">'
            f'<b>{n_ambiguous:,}</b><span>ambiguous</span></div>'
            f'<div class="dc-stat" title="Skipped: not in our reference protein set">'
            f'<b>{n_unmatched:,}</b><span>not in reference</span></div>'
            f'</div>'
            + (
                '<div class="dc-note">Ambiguous = gene name appeared more than once in the file. '
                'Not in reference = no annotated protein in our reference set. Both are skipped.</div>'
                if (n_ambiguous or n_unmatched) else ""
            )
            + '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section-label">2 · Results</div>', unsafe_allow_html=True)
    top_k = st.slider(
        "Proteins to show", min_value=5, max_value=50, value=DEFAULT_TOP_K,
    )

    if client is None:
        st.warning(
            "No Claude API key configured — Stage 3 interpretation is disabled. "
            "Set the `ANTHROPIC_API_KEY` environment variable and restart to enable it. "
            "Retrieval and abundance results still work without it."
        )

# ---------------------------------------------------------------------------
# Header -- the ProteomIQ wordmark plus a compact workflow indicator, always
# shown. Full onboarding (headline, explainer) only before a sample is
# loaded; afterwards the header stays compact so the search/results remain
# the dominant part of the screen.
# ---------------------------------------------------------------------------
_current_query = st.session_state.get("main_query") or ""
_searched = active_source is not None and bool(_current_query)
# chat_messages is only cleared further down when the query changes, so
# also require the query to be the one the conversation was about.
_interpreted = (
    _searched
    and bool(st.session_state.get("chat_messages"))
    and st.session_state.get("last_query") == _current_query
)
if active_source is None:
    _steps = ("current", "todo", "todo", "todo")
elif not _searched:
    _steps = ("done", "current", "todo", "todo")
elif not _interpreted:
    _steps = ("done", "done", "current", "todo")
else:
    _steps = ("done", "done", "done", "current")
render_brand_header(_steps)

if active_source is None:
    st.markdown("""
<div class="hero">
<h1>Explore proteins through natural language</h1>
<p>Search the proteins measured in your proteomics sample with a plain-language question &mdash;
ranked by semantic similarity between protein and text embeddings, with each protein's abundance
in your sample shown alongside.</p>
</div>
<div class="start-card">
<div class="start-title">Load a proteomics sample to begin</div>
<div class="start-hint">Upload a CSV with <code>gene</code> and <code>abundance</code> columns in the
sidebar, or use the example dataset (LUAD patient 11LU013 from CPTAC).</div>
</div>
""", unsafe_allow_html=True)

    with st.container(key="howto"), st.expander("How does ProteomIQ work?"):
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

    st.stop()

# ---------------------------------------------------------------------------
# Steps 2-4, side by side: search + results in the wider center column as
# the main workspace, AI interpretation in a narrower column on the right
# -- CellWhisperer-style layout (a primary view + a dedicated "chat
# protocol" panel beside it), rather than the chat stacked below everything.
# ---------------------------------------------------------------------------
main_col, chat_col = st.columns([1.75, 1], gap="large")


def render_ai_cta(state: str, n_proteins: int = 0) -> bool:
    """Right-column call-to-action shown BEFORE any interpretation exists:
    a compact optional-feature card, not an (empty) chat. state is one of
    "no_query" (nothing to interpret yet -> disabled), "no_client" (no API
    key -> disabled) or "ready". Returns True when the button was clicked.
    Once a conversation exists, the caller renders the chat panel instead."""
    with st.container(key="ai_cta"):
        if state == "ready":
            intro = (
                f"Get a short biological interpretation of the top {n_proteins} proteins, "
                "then ask follow-up questions about the results."
            )
        elif state == "no_client":
            intro = "Unavailable — no Claude API key is configured for this app (see sidebar)."
        else:
            intro = "Available once you have run a protein search."
        st.markdown(
            '<div class="section-eyebrow">AI interpretation · optional</div>'
            '<div class="ai-title">Interpret these results</div>'
            f'<p class="ai-text">{intro}</p>'
            '<ul class="ai-points">'
            '<li>Uses the retrieved proteins, their abundance in this sample and UniProt function text</li>'
            '<li>Follow-up questions and new searches within the same conversation</li>'
            '</ul>',
            unsafe_allow_html=True,
        )
        clicked = st.button(
            "Interpret results", type="primary", key="interpret_btn", width="stretch",
            disabled=state != "ready",
        )
        st.markdown(
            '<div class="ai-footnote">Interpretation by Claude (Anthropic) · for hypothesis '
            'generation, not diagnosis.</div>',
            unsafe_allow_html=True,
        )
    return clicked


with main_col:
    # Search -- the main focus of the page once a sample is loaded: one
    # horizontal bar, [ large input ][ Search ]. The "Search" button is an
    # additional, purely visual trigger -- clicking it just causes
    # Streamlit's normal rerun, which re-evaluates the same last_query
    # comparison below exactly as pressing Enter already does; no new
    # state-handling logic. Fixed pixel width + nowrap (CSS) so the label
    # can never wrap.
    render_section_header("Protein Search", "Find proteins in this sample using natural language.")
    with st.container(horizontal=True, vertical_alignment="center", key="search_bar"):
        query = st.text_input(
            "Question", label_visibility="collapsed", key="main_query",
            placeholder="Describe a biological function, process, pathway or phenotype...",
            width="stretch",
        )
        st.button("Search", key="search_btn", type="primary", width=128)
    st.markdown(
        '<div class="search-examples">Examples: '
        '<span class="ex">receptor tyrosine kinase signaling</span>'
        '<span class="ex">immune evasion in lung cancer</span>'
        '<span class="ex">DNA damage repair</span></div>',
        unsafe_allow_html=True,
    )

    if not query:
        with chat_col:
            render_ai_cta("no_query")
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
    st.markdown('<div class="search-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="results-head"><div class="section-title">Results</div>'
        f'<span class="count-badge">{len(top)} proteins</span></div>'
        f'<div class="results-sub">Ranked by semantic similarity to '
        f'<span class="q">“{html.escape(query)}”</span> · click a protein for its function</div>',
        unsafe_allow_html=True,
    )
    render_result_list([
        (i, gene_lookup.get(pid, pid), cos_sim, pct, annotation_lookup.get(pid))
        for i, (pid, cos_sim, pct) in enumerate(top, start=1)
    ])

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    with st.container(key="plot_card"):
        render_section_header(
            "Abundance profile",
            "Where these proteins sit among all proteins measured in this sample.",
        )
        rank_plot_result = render_rank_plot(raw_abundance, top, gene_lookup)
        if rank_plot_result is not None:
            rank_fig, used_log_scale = rank_plot_result
            st.pyplot(rank_fig, width="stretch")
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
# sample, beside the results (not stacked below them) so the retrieved
# proteins stay in view while chatting about them. Stage 1/2 retrieval
# logic is unchanged; only this layer is multi-turn AND can trigger
# additional retrievals mid-conversation via the search box below, so a
# follow-up that needs different data doesn't have to restart the whole
# chat. Deliberately NOT agentic -- Claude never decides on its own to run a
# new search; the researcher always explicitly triggers one.
# Before the first turn, only a compact call-to-action card is shown (no
# empty chat panel); the first turn requires an explicit click (Claude API
# calls cost money and take a few seconds, so this shouldn't fire on every
# rerun). After it succeeds, st.rerun() swaps the card for the chat panel.
with chat_col:
    if client is None or not st.session_state.get("chat_messages"):
        if render_ai_cta("ready" if client is not None else "no_client", n_proteins=len(top)):
            chat_searches = [{"query": query, "selected": scored[:top_k]}]
            system_prompt = build_chat_system_prompt(chat_searches, gene_lookup, annotation_lookup)
            opening_question = "What's your interpretation of these results?"
            with st.spinner("Interpreting the results..."):
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
                    st.rerun()
    else:
        # A real st.container(key=...) -- genuinely nests everything inside
        # it in one DOM container with a stable `st-key-chat_panel` class we
        # can style (see CSS above). (A raw-HTML "open a <div> in one
        # st.markdown call, close it in a later one" does not work: each
        # call is its own isolated fragment.)
        with st.container(key="chat_panel"):
            render_section_header(
                'Interpretation<span class="ai-badge">Claude</span>',
                "Ask follow-up questions about these results, or search for something new.",
                eyebrow="AI interpretation",
            )
            chat_history = st.container(height=560)
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

st.markdown(
    '<div class="app-footer">ProteomIQ is a research prototype for hypothesis generation, not a '
    'diagnostic tool. See the accompanying thesis for methodology, evaluation, and known limitations.</div>',
    unsafe_allow_html=True,
)
