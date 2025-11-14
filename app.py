# size_ratio_explorer_app.py
# Streamlit app to explore compute_option_size_ratios_v5 step by step
# with full transparency, colour effects, revenue hierarchy & option comparisons.

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from size_ratio_engine import compute_option_size_ratios_v5


# -------------------- CONFIG & CONSTANTS -------------------- #

st.set_page_config(
    page_title="Size Ratio Engine Explorer",
    page_icon="📏",
    layout="wide",
)

st.markdown(
    """
<style>
.main {
    background-color: #fafafa;
}
.step-card {
    padding: 1rem 1.5rem;
    border-radius: 0.75rem;
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}
.small-caption {
    font-size: 0.85rem;
    color: #666666;
}
.formula-box {
    padding: 0.5rem 0.75rem;
    border-radius: 0.35rem;
    background-color: #f7f7ff;
    border: 1px dashed #c3c3ff;
    margin-top: 0.5rem;
    margin-bottom: 0.75rem;
}

/* Make section headings visually consistent */
h2, h3, h4 {
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
}

/* ---------- Option overview layout ---------- */
.overview-header-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.25rem;
    margin-top: 0.75rem;
    margin-bottom: 0.75rem;
    align-items: flex-start;
}
.overview-tag {
    font-size: 0.8rem;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f3f4ff;
    border: 1px solid #d9ddff;
    color: #111827;
}
.overview-tag span.label {
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.7rem;
    color: #4b5563;
}
.overview-tag span.value {
    font-family: monospace;
    font-size: 0.8rem;
    padding: 0.1rem 0.3rem;
    background: #eef2ff;
    border-radius: 0.3rem;
}
.overview-tag.option   { border-color: rgba(129, 140, 248, 0.9); }
.overview-tag.cat2     { background: #ecfdf3; border-color: #bbf7d0; }
.overview-tag.cat3     { background: #fff7ed; border-color: #fed7aa; }
.overview-tag.color    { background: #eff6ff; border-color: #bfdbfe; }

.overview-metric-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 0.25rem;
}
.overview-metric-box {
    flex: 1 1 180px;
    min-width: 180px;
    max-width: 260px;
    padding: 0.7rem 0.9rem;
    border-radius: 0.65rem;
    border: 1px solid #e5e7eb;
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
}
.overview-metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
    margin-bottom: 0.15rem;
}
.overview-metric-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f172a;
}
.overview-metric-help {
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 0.15rem;
}

.overview-metric-box.revenue {
    flex: 2 1 280px;
    max-width: 420px;
}
.overview-rev-line {
    font-size: 0.8rem;
    margin-bottom: 0.15rem;
}
.overview-rev-line span.key {
    font-family: monospace;
    font-weight: 600;
}
.overview-rev-line span.value {
    font-weight: 600;
}
.overview-rev-line small {
    color: #6b7280;
}

/* ---------- Chip / pill styles (size mix snapshot) ---------- */
.snapshot-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 0.5rem;
}
.pill {
    display: inline-flex;
    align-items: center;
    padding: 0.6rem 1rem;
    border-radius: 999px;
    background: #f3f4ff;
    border: 1px solid #d9ddff;
    font-size: 0.85rem;
    color: #111827;
    gap: 0.45rem;
    min-width: 0;
    flex-shrink: 0;
    cursor: default;
}
.pill-secondary {
    background: #f5fbf7;
    border-color: #ccefd9;
}
.pill-tertiary {
    background: #fff7f1;
    border-color: #ffd9b8;
}
.pill-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4f46e5;
    flex-shrink: 0;
}
.pill-dot-secondary {
    background: #16a34a;
}
.pill-dot-tertiary {
    background: #ea580c;
}
.pill-text-block {
    display: flex;
    flex-direction: column;
}
.pill-label {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.7rem;
    color: #4b5563;
}
.pill-value {
    font-weight: 600;
    font-size: 0.9rem;
}
.pill-subvalue {
    font-size: 0.75rem;
    color: #6b7280;
}
</style>
""",
    unsafe_allow_html=True,
)

# Defaults aligned with your engine
ROS_COL_DEFAULT = "pred_ros_at_target_disc_wavg_boosted"
REV_COL_DEFAULT = "revenue_per_day_boosted"

OPTION_COL_DEFAULT = "optioncode"
SIZE_COL_DEFAULT = "size"
CAT_L3_COL_DEFAULT = "cat_l3"
CAT_L2_COL_DEFAULT = "cat_l2"
COLORGROUP_COL_DEFAULT = "colorgroup"
STATUS_COL_DEFAULT = "Status"
STOCK_COL_DEFAULT = "stockcode"

# Hyperparameters (for explanations & recomputations)
CAT3_STRENGTH_POWER = 0.5
OPTION_STRENGTH_POWER = 1.0

CORE_BIAS_MIN = 0.7
CORE_BIAS_MAX = 1.3
FRINGE_BIAS_MIN = 0.8
FRINGE_BIAS_MAX = 1.2
COLOR_CORE_TOP_N = 5

# Preferred size ordering
SIZE_ORDER_LIST = ["XXS", "XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL", "6XL"]


# -------------------- UTILS -------------------- #

def _fmt_pct(x: float) -> str:
    try:
        return f"{x * 100:.1f}%"
    except Exception:
        return ""


def _safe_sum(x: pd.Series) -> float:
    return float(x.sum()) if len(x) else 0.0


def order_sizes(raw_sizes) -> list:
    """
    Order sizes as XXS, XS, S, M, L, XL, 2XL, 3XL, 4XL, ... whenever they appear.
    Fallback: lexical order for anything not in the predefined list.
    """
    raw_sizes = [str(s) for s in raw_sizes]
    order_map = {s.upper(): i for i, s in enumerate(SIZE_ORDER_LIST)}
    return sorted(raw_sizes, key=lambda s: (order_map.get(s.upper(), 999), s))


# -------------------- SIDEBAR: DATA & SETTINGS -------------------- #

st.sidebar.title("📂 Data & Settings")

uploaded = st.sidebar.file_uploader(
    "Upload CSV or Parquet with ROS data",
    type=["csv", "parquet"],
)

st.sidebar.markdown(
    """
**Expected columns** (default names):

- `optioncode` (option id)  
- `stockcode`  
- `size`  
- `cat_l2`, `cat_l3`  
- `colorgroup` *(optional)*  
- `Status` *(for Dropped rows)*  
- `pred_ros_at_target_disc_wavg_boosted` *(ROS)*  
- `revenue_per_day_boosted` *(revenue per day, for shares)*  
"""
)

with st.sidebar.expander("Advanced: column name overrides", expanded=False):
    option_col = st.text_input("Option column", OPTION_COL_DEFAULT)
    stock_col = st.text_input("Stock column", STOCK_COL_DEFAULT)
    size_col = st.text_input("Size column", SIZE_COL_DEFAULT)
    cat_l3_col = st.text_input("Cat L3 column", CAT_L3_COL_DEFAULT)
    cat_l2_col = st.text_input("Cat L2 column", CAT_L2_COL_DEFAULT)
    colorgroup_col = st.text_input("Colorgroup column (optional)", COLORGROUP_COL_DEFAULT)
    status_col = st.text_input("Status column (optional)", STATUS_COL_DEFAULT)
    ros_col = st.text_input("ROS column", ROS_COL_DEFAULT)
    rev_col = st.text_input("Revenue column", REV_COL_DEFAULT)


# -------------------- LOAD DATA -------------------- #

data_source_msg = ""

if uploaded is not None:
    if uploaded.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_parquet(uploaded)
    data_source_msg = f"Loaded uploaded file `{uploaded.name}` with {len(df_raw):,} rows."
else:
    default_path = Path("final_ros.parquet")
    if default_path.exists():
        df_raw = pd.read_parquet(default_path)
        data_source_msg = (
            f"No file uploaded. Loaded default `final_ros.parquet` "
            f"({len(df_raw):,} rows)."
        )
    else:
        st.info(
            "⬅️ Upload a CSV/Parquet in the sidebar, or place `final_ros.parquet` "
            "in the same folder as this app."
        )
        st.stop()

st.sidebar.success("Data loaded ✅")
st.sidebar.caption(data_source_msg)

# Column checks
required_cols = [option_col, size_col, cat_l3_col, cat_l2_col, ros_col]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns in data: {missing}")
    st.stop()

# Mask same as engine default
if status_col in df_raw.columns:
    mask_status_ok = df_raw[status_col] != "Dropped"
else:
    mask_status_ok = pd.Series(True, index=df_raw.index)


# -------------------- RUN ENGINE -------------------- #

st.title("📏 Size Ratio Engine Explorer")
st.caption(
    "Interactive walkthrough of the hierarchical size-ratio logic for any option. "
    "Complete transparency, colour effects, and business shares."
)

with st.spinner("Running size-ratio engine on full dataset..."):
    df_result = compute_option_size_ratios_v5(
        df_boosted_cat=df_raw,
        mask_status_ok=mask_status_ok,
        ros_col=ros_col,
        option_col=option_col,
        stock_col=stock_col if stock_col in df_raw.columns else None,
        size_col=size_col,
        cat_l3_col=cat_l3_col,
        cat_l2_col=cat_l2_col,
        colorgroup_col=colorgroup_col if colorgroup_col in df_raw.columns else None,
        status_col=status_col if status_col in df_raw.columns else "Status",
    )

st.success("Engine run complete ✅")

# Download full engine output
csv_data = df_result.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="⬇️ Download engine output (CSV)",
    data=csv_data,
    file_name="size_ratio_engine_output.csv",
    mime="text/csv",
)


# -------------------- OPTION SELECTION -------------------- #

option_ids = df_result[option_col].dropna().unique()
option_ids = sorted(option_ids, key=lambda x: str(x))

selected_option = st.selectbox(
    "🔎 Pick an option to inspect",
    options=option_ids,
    index=0,
)

compare_option = st.sidebar.selectbox(
    "Compare with another option (Compare tab)",
    options=option_ids,
    index=min(1, len(option_ids) - 1) if len(option_ids) > 1 else 0,
    help="This second option will be used in the Compare Options tab.",
)

df_opt_res = df_result[df_result[option_col] == selected_option].copy()
df_opt_raw = df_raw[df_raw[option_col] == selected_option].copy()

if df_opt_res.empty:
    st.warning("No rows found for that option in the engine output.")
    st.stop()

# Cast size to string for all curves & ordering
df_opt_res["_size_str"] = df_opt_res[size_col].astype(str)
df_opt_raw["_size_str"] = df_opt_raw[size_col].astype(str)

sizes_ordered = order_sizes(df_opt_res["_size_str"].unique())

# Meta
cat_l3_val = df_opt_res[cat_l3_col].iloc[0]
cat_l2_val = df_opt_res[cat_l2_col].iloc[0]
color_val = (
    df_opt_res[colorgroup_col].iloc[0]
    if colorgroup_col in df_opt_res.columns
    else None
)

opt_total_ros = df_opt_res["opt_total_ros"].iloc[0]
option_strength = df_opt_res["option_strength"].iloc[0]
cat3_strength = df_opt_res["cat3_strength"].iloc[0]

# Revenue summary (if available)
if rev_col in df_raw.columns:
    df_valid_rev = df_raw.loc[mask_status_ok & df_raw[rev_col].notna()]

    rev_option = float(
        df_valid_rev.loc[df_valid_rev[option_col] == selected_option, rev_col].sum()
    )
    rev_cat3 = float(
        df_valid_rev.loc[df_valid_rev[cat_l3_col] == cat_l3_val, rev_col].sum()
    )
    rev_cat2 = float(
        df_valid_rev.loc[df_valid_rev[cat_l2_col] == cat_l2_val, rev_col].sum()
    )
    rev_total = float(df_valid_rev[rev_col].sum())

    share_opt_in_cat3 = rev_option / rev_cat3 if rev_cat3 > 0 else 0.0
    share_cat3_in_cat2 = rev_cat3 / rev_cat2 if rev_cat2 > 0 else 0.0
    share_cat2_in_total = rev_cat2 / rev_total if rev_total > 0 else 0.0
else:
    rev_option = rev_cat3 = rev_cat2 = rev_total = None
    share_opt_in_cat3 = share_cat3_in_cat2 = share_cat2_in_total = None


# -------------------- BUILD CURVES FOR VISUALISATION -------------------- #

opt_curve = pd.Series(
    df_opt_res["opt_ratio"].values, index=df_opt_res["_size_str"].values
)
cat3_curve = pd.Series(
    df_opt_res["cat_l3_ratio"].values, index=df_opt_res["_size_str"].values
)
cat2_curve = pd.Series(
    df_opt_res["cat_l2_ratio"].values, index=df_opt_res["_size_str"].values
)
final_curve = pd.Series(
    df_opt_res["final_ratio"].values, index=df_opt_res["_size_str"].values
)
size_type_series = pd.Series(
    df_opt_res["size_type"].values, index=df_opt_res["_size_str"].values
)


# ---------- OPTION OVERVIEW CARD (pure HTML inside step-card) ---------- #

if color_val is not None and str(color_val) != "nan":
    color_tag = (
        '<div class="overview-tag color">'
        '<span class="label">Colorgroup</span>'
        f'<span class="value">{color_val}</span>'
        '</div>'
    )
else:
    color_tag = ""

if rev_option is not None:
    rev_block = (
        '<div class="overview-metric-box revenue">'
        '<div class="overview-metric-label">Revenue &amp; shares</div>'
        '<div class="overview-rev-line">'
        '<span class="key">Option</span> : '
        f'<span class="value">{rev_option:,.0f} / day</span> '
        f'<small> ({_fmt_pct(share_opt_in_cat3)} within Cat L3)</small>'
        '</div>'
        '<div class="overview-rev-line">'
        '<span class="key">Cat L3</span> : '
        f'<span class="value">{rev_cat3:,.0f} / day</span> '
        f'<small> ({_fmt_pct(share_cat3_in_cat2)} within Cat L2)</small>'
        '</div>'
        '<div class="overview-rev-line">'
        '<span class="key">Cat L2</span> : '
        f'<span class="value">{rev_cat2:,.0f} / day</span> '
        f'<small> ({_fmt_pct(share_cat2_in_total)} within Total)</small>'
        '</div>'
        '</div>'
    )
else:
    rev_block = ""

overview_html = (
    '<div class="step-card">'
    '<h3>🔍 Option overview</h3>'
    '<div class="overview-header-row">'
    '<div class="overview-tag option">'
    '<span class="label">Option</span>'
    f'<span class="value">{selected_option}</span>'
    '</div>'
    '<div class="overview-tag cat2">'
    '<span class="label">Cat L2</span>'
    f'<span class="value">{cat_l2_val}</span>'
    '</div>'
    '<div class="overview-tag cat3">'
    '<span class="label">Cat L3</span>'
    f'<span class="value">{cat_l3_val}</span>'
    '</div>'
    f'{color_tag}'
    '</div>'  # header row
    '<div class="overview-metric-grid">'
    '<div class="overview-metric-box">'
    '<div class="overview-metric-label">Option total ROS</div>'
    f'<div class="overview-metric-value">{opt_total_ros:.1f}</div>'
    '<div class="overview-metric-help">Sum of ROS across all sizes for this option</div>'
    '</div>'
    '<div class="overview-metric-box">'
    '<div class="overview-metric-label">Option strength (vs Cat L3 peers)</div>'
    f'<div class="overview-metric-value">{option_strength:.3f}</div>'
    '<div class="overview-metric-help">Relative to options in the same Cat L3</div>'
    '</div>'
    '<div class="overview-metric-box">'
    '<div class="overview-metric-label">Cat L3 strength (inside Cat L2)</div>'
    f'<div class="overview-metric-value">{cat3_strength:.3f}</div>'
    '<div class="overview-metric-help">Share of this Cat L3 inside its Cat L2</div>'
    '</div>'
    f'{rev_block}'
    '</div>'  # metric grid
    '</div>'  # step-card
)

st.markdown(overview_html, unsafe_allow_html=True)


# ---------- TOP SUMMARY ROW (step-card with pills + hover tooltips) ---------- #

size_types_ordered = size_type_series.reindex(sizes_ordered)

core_mask_top = size_types_ordered == "core"
fringe_mask_top = size_types_ordered == "fringe"

# Extended sizes based on labels (3XL and above)
extended_labels = {"3XL", "4XL", "5XL", "6XL"}
extended_mask_top = pd.Series(
    [sz.upper() in extended_labels for sz in sizes_ordered],
    index=sizes_ordered,
)

core_share_top = float(final_curve.reindex(sizes_ordered)[core_mask_top].sum())
fringe_share_top = float(final_curve.reindex(sizes_ordered)[fringe_mask_top].sum())
extended_share_top = float(final_curve.reindex(sizes_ordered)[extended_mask_top].sum())

core_count_top = int(core_mask_top.sum())
fringe_count_top = int(fringe_mask_top.sum())
extended_count_top = int(extended_mask_top.sum())


def _tooltip_for_mask(mask: pd.Series, label: str) -> str:
    sizes = [sz for sz in mask.index if mask.get(sz, False)]
    if not sizes:
        return f"No {label.lower()} sizes present."
    parts = []
    for sz in sizes:
        parts.append(f"{sz} ({_fmt_pct(float(final_curve.get(sz, 0.0)))})")
    return f"{label} sizes: " + ", ".join(parts)


core_tt = _tooltip_for_mask(core_mask_top, "Core")
fringe_tt = _tooltip_for_mask(fringe_mask_top, "Fringe")
extended_tt = _tooltip_for_mask(extended_mask_top, "Extended")

snapshot_html = f"""
<div class="step-card">
<h4>📊 Size mix snapshot (for this option)</h4>
<div class="snapshot-row">
  <div class="pill" title="{core_tt}">
    <div class="pill-dot"></div>
    <div class="pill-text-block">
      <div class="pill-label">Core sizes</div>
      <div class="pill-value">{core_count_top} sizes</div>
      <div class="pill-subvalue">{_fmt_pct(core_share_top)} of demand</div>
    </div>
  </div>

  <div class="pill pill-secondary" title="{fringe_tt}">
    <div class="pill-dot pill-dot-secondary"></div>
    <div class="pill-text-block">
      <div class="pill-label">Fringe sizes</div>
      <div class="pill-value">{fringe_count_top} sizes</div>
      <div class="pill-subvalue">{_fmt_pct(fringe_share_top)} of demand</div>
    </div>
  </div>

  <div class="pill pill-tertiary" title="{extended_tt}">
    <div class="pill-dot pill-dot-tertiary"></div>
    <div class="pill-text-block">
      <div class="pill-label">Extended (2XL+)</div>
      <div class="pill-value">{extended_count_top} sizes</div>
      <div class="pill-subvalue">{_fmt_pct(extended_share_top)} of demand</div>
    </div>
  </div>
</div>
</div>
"""

st.markdown(snapshot_html, unsafe_allow_html=True)


def build_bar_chart_curves(
    sizes, opt, cat3, cat2, final
) -> go.Figure:
    fig = go.Figure()

    fig.add_bar(
        x=sizes,
        y=opt.reindex(sizes),
        name="Raw option curve",
        opacity=0.85,
    )
    fig.add_bar(
        x=sizes,
        y=cat3.reindex(sizes),
        name="Cat L3 baseline",
        opacity=0.7,
    )
    fig.add_bar(
        x=sizes,
        y=cat2.reindex(sizes),
        name="Cat L2 baseline",
        opacity=0.7,
    )
    fig.add_bar(
        x=sizes,
        y=final.reindex(sizes),
        name="Final curve (engine)",
        opacity=0.9,
    )

    fig.update_layout(
        barmode="group",
        title="Curves by size (raw vs baselines vs final)",
        xaxis_title="Size",
        yaxis_title="Share of demand",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450,
    )
    return fig


def build_core_fringe_chart(sizes, final, size_type) -> go.Figure:
    df_cf = pd.DataFrame(
        {
            "size": sizes,
            "final_ratio": final.reindex(sizes).values,
            "size_type": size_type.reindex(sizes).values,
        }
    )

    fig = go.Figure()
    for stype, sub in df_cf.groupby("size_type"):
        fig.add_bar(
            x=sub["size"],
            y=sub["final_ratio"],
            name=f"{stype}",
        )

    fig.update_layout(
        barmode="group",
        title="Final curve split by size_type (core / fringe / one_size)",
        xaxis_title="Size",
        yaxis_title="Share of demand",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig


# -------------------- TABS -------------------- #

(
    tab_logic,
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
    tab8,
) = st.tabs(
    [
        "📘 Logic Overview",
        "1️⃣ Data & Raw Option Curve",
        "2️⃣ Category Baselines",
        "3️⃣ Engine Output vs Baselines",
        "4️⃣ Core vs Fringe & Guardrails",
        "5️⃣ Colour Effects",
        "6️⃣ Detailed Calculations",
        "7️⃣ Compare Options",
        "8️⃣ Full Table View",
    ]
)


# ----- TAB 0: Logic Overview ----- #
with tab_logic:
    st.subheader("📘 Full Logic Overview")

    st.markdown("This page explains the complete logic behind the size ratio engine.")

    st.markdown("#### 1. Filter valid data")
    st.markdown(
        "- We drop rows with `Status = 'Dropped'` and keep all valid sales data.\n"
        "- All downstream curves are built only on this **clean base**."
    )

    st.markdown("#### 2. Build raw option size curve")
    st.markdown("For each option `o` and size `s`:")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r"opt\_total\_ros(o) = \sum_{s} ROS(o, s)")
    st.latex(
        r"opt\_ratio(o, s) = \frac{ROS(o, s)}{\sum_{s'} ROS(o, s')}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "- This curve reflects **how this option actually sold by size**.\n"
        "- It can be biased if some sizes were often out of stock."
    )

    st.markdown("#### 3. Build category baselines (Cat L3 and Cat L2)")

    st.markdown(
        "We aggregate ROS at sub-category (`Cat L3`) and category (`Cat L2`) level:"
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"cat\_l3\_ratio(c3, s) = "
        r"\frac{ROS(c3, s)}{\sum_{s'} ROS(c3, s')}"
    )
    st.latex(
        r"cat\_l2\_ratio(c2, s) = "
        r"\frac{ROS(c2, s)}{\sum_{s'} ROS(c2, s')}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "- These curves represent **portfolio behaviour** of that sub-category and category.\n"
        "- They are more stable than any single option."
    )

    st.markdown("#### 4. Category strength (Cat L3 inside Cat L2)")

    st.markdown(
        "We measure how important a sub-category is inside its parent category:"
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"share_{cat3} = \frac{TotalROS(cat3)}{TotalROS(cat2)}"
    )
    st.latex(
        r"cat3\_strength = share_{cat3}^{0.5}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "- If a Cat L3 is a big chunk of Cat L2, its curve is more trustworthy.\n"
        "- Strength is squashed with a square-root so that very large shares don't dominate too aggressively."
    )

    st.markdown("#### 5. Option strength (within its Cat L3)")

    st.markdown(
        "We compare the option to its peers in the same sub-category:"
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"T = ROS(option), \quad M = median\_ROS(\text{options in same Cat L3})"
    )
    st.latex(
        r"base = \begin{cases}"
        r"\dfrac{T}{T+M} & M > 0 \\"
        r"0.5 & M \le 0 \text{ and } T > 0 \\"
        r"0 & T = 0"
        r"\end{cases}"
    )
    st.latex(
        r"option\_strength = base^{1.0}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "- High ROS vs peers ⇒ strength close to 1.\n"
        "- Weak option ⇒ strength closer to 0.\n"
        "- This strength decides how much we trust the **option curve vs category curves**."
    )

    st.markdown("#### 6. Core vs fringe size tagging")

    st.markdown(
        "- At Cat L3 and Cat L2 level, we sort sizes by share and pick minimal set covering ~80% ROS (min 2, max 4 sizes). "
        "These are inferred **core sizes**.\n"
        "- Everything else becomes **fringe**.\n"
        "- For a given option, if it has good coverage on expected core sizes, we trust its option curve more."
    )

    st.markdown("#### 7. Blend option, Cat L3, and Cat L2 curves per size")

    st.markdown(
        "For each option and size `s`, we assign weights that depend on:\n"
        "- Option strength\n"
        "- Cat L3 strength\n"
        "- Whether `s` is core or fringe"
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"shape(s) = "
        r"w_{opt}(s)\,opt\_ratio(s) + "
        r"w_{c3}(s)\,cat\_l3\_ratio(s) + "
        r"w_{c2}(s)\,cat\_l2\_ratio(s)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "- For **core sizes**, we allow higher `w_opt` if option strength is high.\n"
        "- For **fringe sizes**, we lean more on Cat L3 / Cat L2 baselines.\n"
        "- If Cat L3/Cat L2 curves are missing, we fall back to global size curve."
    )

    st.markdown("#### 8. Colour behaviour & bias")

    st.markdown(
        "We look at how this colour behaves vs overall, Cat L2, and Cat L3:"
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"\text{bias}_{global}(s) = "
        r"\begin{cases}"
        r"\dfrac{global\_color\_ratio(s)}{global\_ratio(s)} & global\_ratio(s) > 0 \\"
        r"1 & \text{otherwise}"
        r"\end{cases}"
    )
    st.latex(
        r"bias(s) = w_g\,\text{bias}_{global}(s) + "
        r"w_2\,\text{bias}_{cat2}(s) + "
        r"w_3\,\text{bias}_{cat3}(s)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "- If this colour over-indexes in certain sizes, bias > 1 there.\n"
        "- If it under-indexes, bias < 1.\n"
        "- We clamp bias to **[0.7, 1.3] for core**, **[0.8, 1.2] for fringe** so colour never overpowers the business reality."
    )

    st.markdown("#### 9. Core share guardrails")

    st.markdown(
        "After applying colour, we check the final share of core sizes:"
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"core\_share = \sum_{s \in core} final\_ratio(s)"
    )
    st.latex(
        r"core\_share \in [0.70, 0.90]"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "- If core share is too low/high, we rescale core vs fringe to bring it into [70%, 90%] while keeping intra-core and intra-fringe shapes intact."
    )

    st.markdown("#### 10. Extended sizes floors")

    st.markdown(
        "- For 2XL, 3XL, 4XL, 5XL etc., we set a minimum floor relative to Cat L2 + colour curve.\n"
        "- This avoids allocating **zero** to extended sizes where the brand wants a minimum presence."
    )

    st.markdown("#### 11. Final normalization and output")

    st.markdown(
        "- After all adjustments, we re-normalize so that "
        r"$\sum_s final\_ratio(s) = 1$ for each option.\n"
        "- Output is **one row per (option, size)** with full diagnostics."
    )


# ----- TAB 1: Raw option curve ----- #
with tab1:
    st.subheader("1️⃣ Data & Raw Option Curve")

    st.markdown("We start by aggregating **ROS by size within the option**.")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r"opt\_total\_ros(o) = \sum_{s} ROS(o, s)")
    st.latex(
        r"opt\_ratio(o, s) = \frac{ROS(o, s)}{\sum_{s'} ROS(o, s')}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Recompute from raw data for this option
    mask_opt = mask_status_ok & (df_raw[option_col] == selected_option)
    df_step1_raw = (
        df_raw.loc[mask_opt]
        .assign(_size_str=lambda d: d[size_col].astype(str))
        .groupby("_size_str", dropna=False)[ros_col]
        .sum()
        .reset_index()
        .rename(columns={ros_col: "ros_from_data"})
    )

    total_ros_opt_ = _safe_sum(df_step1_raw["ros_from_data"])
    df_step1_raw["opt_ratio_from_data"] = np.where(
        total_ros_opt_ > 0, df_step1_raw["ros_from_data"] / total_ros_opt_, 0.0
    )

    df_step1 = df_step1_raw.merge(
        df_opt_res[["_size_str", "opt_ratio"]],
        on="_size_str",
        how="left",
    )
    df_step1["opt_ratio_from_data_pct"] = df_step1["opt_ratio_from_data"].apply(_fmt_pct)
    df_step1["opt_ratio_engine_pct"] = df_step1["opt_ratio"].apply(_fmt_pct)
    df_step1 = df_step1.set_index("_size_str").reindex(sizes_ordered)

    st.markdown("**Raw ROS and option curve (data vs engine):**")
    st.dataframe(df_step1, use_container_width=True)

    fig_step1 = go.Figure()
    fig_step1.add_bar(
        x=sizes_ordered,
        y=df_step1["opt_ratio"],
        name="Raw option curve (engine)",
    )
    fig_step1.add_bar(
        x=sizes_ordered,
        y=df_step1["opt_ratio_from_data"],
        name="Raw option curve (recomputed)",
        opacity=0.6,
    )
    fig_step1.update_layout(
        xaxis_title="Size",
        yaxis_title="Share of option ROS",
        title="Raw option curve by size",
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_step1, use_container_width=True)

    st.markdown(
        '<p class="small-caption">This curve reflects how this option actually sold, '
        "but can be biased if some sizes were out of stock.</p>",
        unsafe_allow_html=True,
    )


# ----- TAB 2: Category baselines ----- #
with tab2:
    st.subheader("2️⃣ Category baselines (Cat L3 & Cat L2)")

    st.markdown(
        f"""
For this option we use two baselines:

- **Cat L3 baseline**: all options in sub-category `{cat_l3_val}`  
- **Cat L2 baseline**: all options in category `{cat_l2_val}`  
"""
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"cat\_l3\_ratio(c3, s) = "
        r"\frac{ROS(c3, s)}{\sum_{s'} ROS(c3, s')}"
    )
    st.latex(
        r"cat\_l2\_ratio(c2, s) = "
        r"\frac{ROS(c2, s)}{\sum_{s'} ROS(c2, s')}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    df_valid = df_raw.loc[mask_status_ok].copy()
    df_valid["_size_str"] = df_valid[size_col].astype(str)

    # Cat L3 raw curve
    df_cat3_raw = (
        df_valid[df_valid[cat_l3_col] == cat_l3_val]
        .groupby("_size_str", dropna=False)[ros_col]
        .sum()
        .reset_index()
        .rename(columns={ros_col: "ros_cat3"})
    )
    total_cat3_ros = _safe_sum(df_cat3_raw["ros_cat3"])
    df_cat3_raw["cat_l3_ratio_computed"] = np.where(
        total_cat3_ros > 0, df_cat3_raw["ros_cat3"] / total_cat3_ros, 0.0
    )

    # Cat L2 raw curve
    df_cat2_raw = (
        df_valid[df_valid[cat_l2_col] == cat_l2_val]
        .groupby("_size_str", dropna=False)[ros_col]
        .sum()
        .reset_index()
        .rename(columns={ros_col: "ros_cat2"})
    )
    total_cat2_ros = _safe_sum(df_cat2_raw["ros_cat2"])
    df_cat2_raw["cat_l2_ratio_computed"] = np.where(
        total_cat2_ros > 0, df_cat2_raw["ros_cat2"] / total_cat2_ros, 0.0
    )

    df_step2 = (
        df_opt_res[["_size_str", "cat_l3_ratio", "cat_l2_ratio"]]
        .merge(df_cat3_raw[["_size_str", "cat_l3_ratio_computed"]], on="_size_str", how="left")
        .merge(df_cat2_raw[["_size_str", "cat_l2_ratio_computed"]], on="_size_str", how="left")
        .set_index("_size_str")
    ).reindex(sizes_ordered)

    for col in ["cat_l3_ratio", "cat_l3_ratio_computed", "cat_l2_ratio", "cat_l2_ratio_computed"]:
        df_step2[col + "_pct"] = df_step2[col].apply(_fmt_pct)

    st.markdown("**Baselines (engine vs recomputed from raw data):**")
    st.dataframe(df_step2, use_container_width=True)

    fig_step2 = go.Figure()
    fig_step2.add_bar(
        x=sizes_ordered,
        y=df_step2["cat_l3_ratio"],
        name="Cat L3 baseline (engine)",
    )
    fig_step2.add_bar(
        x=sizes_ordered,
        y=df_step2["cat_l2_ratio"],
        name="Cat L2 baseline (engine)",
    )
    fig_step2.update_layout(
        barmode="group",
        xaxis_title="Size",
        yaxis_title="Share of ROS",
        title="Category & sub-category size curves",
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_step2, use_container_width=True)

    st.markdown(
        '<p class="small-caption">'
        "These curves represent what we expect for the **overall business**, "
        "not just this single option.</p>",
        unsafe_allow_html=True,
    )


# ----- TAB 3: Engine output vs baselines ----- #
with tab3:
    st.subheader("3️⃣ Engine output vs baselines")

    st.markdown(
        "The engine blends the three curves **per size** with weights that depend on "
        "option strength, category strength, and whether the size is core or fringe."
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"shape(s) = "
        r"w_{opt}(s)\,opt\_ratio(s) + "
        r"w_{c3}(s)\,cat\_l3\_ratio(s) + "
        r"w_{c2}(s)\,cat\_l2\_ratio(s)"
    )
    st.latex(
        r"final\_ratio(s) = "
        r"\text{shape after colour bias, core guardrails, and tail floors}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    fig_all = build_bar_chart_curves(
        sizes_ordered, opt_curve, cat3_curve, cat2_curve, final_curve
    )
    st.plotly_chart(fig_all, use_container_width=True)

    df_step3 = pd.DataFrame(
        {
            "size": sizes_ordered,
            "opt_ratio": opt_curve.reindex(sizes_ordered).values,
            "cat_l3_ratio": cat3_curve.reindex(sizes_ordered).values,
            "cat_l2_ratio": cat2_curve.reindex(sizes_ordered).values,
            "final_ratio": final_curve.reindex(sizes_ordered).values,
        }
    ).set_index("size")

    for col in ["opt_ratio", "cat_l3_ratio", "cat_l2_ratio", "final_ratio"]:
        df_step3[col + "_pct"] = df_step3[col].apply(_fmt_pct)

    st.markdown("**Numeric comparison across curves:**")
    st.dataframe(df_step3, use_container_width=True)

    st.markdown(
        '<p class="small-caption">'
        "Final curve is usually close to the category curve, but gently tilted "
        "towards the option's actual behaviour where the data is strong.</p>",
        unsafe_allow_html=True,
    )


# ----- TAB 4: Core vs fringe & guardrails ----- #
with tab4:
    st.subheader("4️⃣ Core vs fringe & guardrails")

    st.markdown(
        """
The engine ensures:

- A healthy split between **core** and **fringe** sizes  
- Extended sizes get a **minimum floor** based on category + colour  
"""
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"core\_share = \sum_{s \in core} final\_ratio(s)"
    )
    st.latex(
        r"core\_share \in [core\_share\_{min}, core\_share\_{max}] = [0.70, 0.90]"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    fig_cf = build_core_fringe_chart(sizes_ordered, final_curve, size_type_series)
    st.plotly_chart(fig_cf, use_container_width=True)

    df_cf = pd.DataFrame(
        {
            "size": sizes_ordered,
            "final_ratio": final_curve.reindex(sizes_ordered).values,
            "final_ratio_pct": final_curve.reindex(sizes_ordered).apply(_fmt_pct).values,
            "size_type": size_type_series.reindex(sizes_ordered).values,
        }
    ).set_index("size")

    st.markdown("**Final ratios with size type:**")
    st.dataframe(df_cf, use_container_width=True)

    core_mask = size_type_series.reindex(sizes_ordered) == "core"
    core_share = float(final_curve.reindex(sizes_ordered)[core_mask].sum())

    st.markdown(
        f"- **Core share in final curve:** `{core_share:.3f}` "
        f"({_fmt_pct(core_share)})"
    )


# ----- TAB 5: Colour Effects ----- #
with tab5:
    st.subheader("5️⃣ Colour effects & bias")

    if colorgroup_col not in df_raw.columns or pd.isna(color_val):
        st.info("No colour group column or this option has no colour. Nothing to show here.")
    else:
        st.markdown(
            f"This option is in colorgroup **`{color_val}`**. "
            "We compare its colour behaviour vs global/category behaviour "
            "and see how the engine applies colour bias."
        )

        df_valid_color = df_raw.loc[mask_status_ok & df_raw[colorgroup_col].notna()].copy()
        df_valid_color["_size_str"] = df_valid_color[size_col].astype(str)

        # Global size curve
        global_size = (
            df_valid_color.groupby("_size_str")[ros_col]
            .sum()
            .reset_index()
            .rename(columns={ros_col: "ros_global"})
        )
        total_global_ros = _safe_sum(global_size["ros_global"])
        global_size["global_ratio"] = np.where(
            total_global_ros > 0, global_size["ros_global"] / total_global_ros, 0.0
        )

        # Global colour curve
        global_color = (
            df_valid_color.groupby([colorgroup_col, "_size_str"])[ros_col]
            .sum()
            .reset_index()
            .rename(columns={ros_col: "ros_global_color"})
        )
        global_color["global_color_total_ros"] = global_color.groupby(colorgroup_col)[
            "ros_global_color"
        ].transform("sum")
        global_color["global_color_ratio"] = np.where(
            global_color["global_color_total_ros"] > 0,
            global_color["ros_global_color"] / global_color["global_color_total_ros"],
            0.0,
        )

        # Cat2 size & colour
        cat2_raw = (
            df_valid_color.groupby([cat_l2_col, "_size_str"])[ros_col]
            .sum()
            .reset_index()
            .rename(columns={ros_col: "ros_cat2"})
        )
        cat2_raw["cat_l2_total_ros"] = cat2_raw.groupby(cat_l2_col)["ros_cat2"].transform("sum")
        cat2_raw["cat_l2_ratio_full"] = np.where(
            cat2_raw["cat_l2_total_ros"] > 0,
            cat2_raw["ros_cat2"] / cat2_raw["cat_l2_total_ros"],
            0.0,
        )

        cat2c = (
            df_valid_color.groupby([cat_l2_col, colorgroup_col, "_size_str"])[ros_col]
            .sum()
            .reset_index()
            .rename(columns={ros_col: "ros_cat2_color"})
        )
        cat2c["cat_l2_color_total_ros"] = cat2c.groupby([cat_l2_col, colorgroup_col])[
            "ros_cat2_color"
        ].transform("sum")
        cat2c["cat_l2_color_ratio"] = np.where(
            cat2c["cat_l2_color_total_ros"] > 0,
            cat2c["ros_cat2_color"] / cat2c["cat_l2_color_total_ros"],
            0.0,
        )

        # Cat3 size & colour
        cat3_raw = (
            df_valid_color.groupby([cat_l3_col, "_size_str"])[ros_col]
            .sum()
            .reset_index()
            .rename(columns={ros_col: "ros_cat3"})
        )
        cat3_raw["cat_l3_total_ros"] = cat3_raw.groupby(cat_l3_col)["ros_cat3"].transform("sum")
        cat3_raw["cat_l3_ratio_full"] = np.where(
            cat3_raw["cat_l3_total_ros"] > 0,
            cat3_raw["ros_cat3"] / cat3_raw["cat_l3_total_ros"],
            0.0,
        )

        cat3c = (
            df_valid_color.groupby([cat_l3_col, colorgroup_col, "_size_str"])[ros_col]
            .sum()
            .reset_index()
            .rename(columns={ros_col: "ros_cat3_color"})
        )
        cat3c["cat_l3_color_total_ros"] = cat3c.groupby([cat_l3_col, colorgroup_col])[
            "ros_cat3_color"
        ].transform("sum")
        cat3c["cat_l3_color_ratio"] = np.where(
            cat3c["cat_l3_color_total_ros"] > 0,
            cat3c["ros_cat3_color"] / cat3c["cat_l3_color_total_ros"],
            0.0,
        )

        # Volume maps for weights
        cat3_vol_map = (
            cat3_raw.groupby(cat_l3_col)["cat_l3_total_ros"].first().to_dict()
        )
        cat2_vol_map = (
            cat2_raw.groupby(cat_l2_col)["cat_l2_total_ros"].first().to_dict()
        )

        global_color_vol_map = (
            global_color.groupby(colorgroup_col)["global_color_total_ros"]
            .first()
            .to_dict()
        )
        cat2_color_vol_map = (
            cat2c.groupby([cat_l2_col, colorgroup_col])["cat_l2_color_total_ros"]
            .first()
            .to_dict()
        )
        cat3_color_vol_map = (
            cat3c.groupby([cat_l3_col, colorgroup_col])["cat_l3_color_total_ros"]
            .first()
            .to_dict()
        )

        # Core colours
        color_vol_series = pd.Series(global_color_vol_map)
        core_colors = set(
            color_vol_series.sort_values(ascending=False).head(COLOR_CORE_TOP_N).index
        )

        global_total_col_ros = sum(global_color_vol_map.values()) or 1.0
        T_cat3 = float(cat3_vol_map.get(cat_l3_val, 0.0))
        T_cat2 = float(cat2_vol_map.get(cat_l2_val, 0.0))
        T_cat3_color = float(cat3_color_vol_map.get((cat_l3_val, color_val), 0.0))
        T_cat2_color = float(cat2_color_vol_map.get((cat_l2_val, color_val), 0.0))
        T_global_color = float(global_color_vol_map.get(color_val, 0.0))

        share_global_col = T_global_color / global_total_col_ros if global_total_col_ros > 0 else 0.0
        share_cat2_col = T_cat2_color / T_cat2 if T_cat2 > 0 else 0.0
        share_cat3_col = T_cat3_color / T_cat3 if T_cat3 > 0 else 0.0

        def _color_score(share: float, is_core: bool) -> float:
            if share <= 0:
                return 0.0
            score = share
            if is_core:
                score *= 1.2
            return score

        is_core_color = color_val in core_colors
        score_global = _color_score(share_global_col, is_core_color)
        score_cat2 = _color_score(share_cat2_col, is_core_color)
        score_cat3 = _color_score(share_cat3_col, is_core_color)
        total_score = score_global + score_cat2 + score_cat3

        if total_score <= 0:
            w_col_global = 1.0
            w_col_cat2 = 0.0
            w_col_cat3 = 0.0
        else:
            w_col_global = score_global / total_score
            w_col_cat2 = score_cat2 / total_score
            w_col_cat3 = score_cat3 / total_score

        # Build per-size bias factors
        global_ratio_s = global_size.set_index("_size_str")["global_ratio"]
        global_color_ratio_s = (
            global_color[global_color[colorgroup_col] == color_val]
            .set_index("_size_str")["global_color_ratio"]
        )
        base_cat2_s = cat2_raw[cat2_raw[cat_l2_col] == cat_l2_val].set_index("_size_str")[
            "cat_l2_ratio_full"
        ]
        base_color_cat2_s = (
            cat2c[(cat2c[cat_l2_col] == cat_l2_val) & (cat2c[colorgroup_col] == color_val)]
            .set_index("_size_str")["cat_l2_color_ratio"]
        )
        base_cat3_s = cat3_raw[cat3_raw[cat_l3_col] == cat_l3_val].set_index("_size_str")[
            "cat_l3_ratio_full"
        ]
        base_color_cat3_s = (
            cat3c[(cat3c[cat_l3_col] == cat_l3_val) & (cat3c[colorgroup_col] == color_val)]
            .set_index("_size_str")["cat_l3_color_ratio"]
        )

        bias_rows = []

        for sz in sizes_ordered:
            stype = size_type_series.get(sz, "fringe")

            gr = float(global_ratio_s.get(sz, 0.0))
            gcr = float(global_color_ratio_s.get(sz, 0.0))
            c2 = float(base_cat2_s.get(sz, 0.0))
            c2c = float(base_color_cat2_s.get(sz, 0.0))
            c3 = float(base_cat3_s.get(sz, 0.0))
            c3c = float(base_color_cat3_s.get(sz, 0.0))

            bias_global = (gcr / gr) if gr > 0 else 1.0
            bias_cat2 = (c2c / c2) if c2 > 0 else 1.0
            bias_cat3 = (c3c / c3) if c3 > 0 else 1.0

            b = (
                w_col_global * bias_global
                + w_col_cat2 * bias_cat2
                + w_col_cat3 * bias_cat3
            )

            if stype in ["core", "one_size"]:
                lo, hi = CORE_BIAS_MIN, CORE_BIAS_MAX
            else:
                lo, hi = FRINGE_BIAS_MIN, FRINGE_BIAS_MAX

            b_clamped = max(lo, min(hi, float(b)))

            final_val = float(final_curve.get(sz, 0.0))
            shape_approx = final_val / b_clamped if b_clamped != 0 else final_val
            bias_rows.append(
                {
                    "size": sz,
                    "size_type": stype,
                    "global_ratio": gr,
                    "global_color_ratio": gcr,
                    "cat_l2_ratio": c2,
                    "cat_l2_color_ratio": c2c,
                    "cat_l3_ratio": c3,
                    "cat_l3_color_ratio": c3c,
                    "bias_global": bias_global,
                    "bias_cat2": bias_cat2,
                    "bias_cat3": bias_cat3,
                    "combined_bias_clamped": b_clamped,
                    "final_ratio": final_val,
                    "approx_shape_no_color": shape_approx,
                }
            )

        df_bias = pd.DataFrame(bias_rows).set_index("size")
        shape_sum = df_bias["approx_shape_no_color"].sum()
        if shape_sum > 0:
            df_bias["approx_shape_no_color_norm"] = (
                df_bias["approx_shape_no_color"] / shape_sum
            )
        else:
            df_bias["approx_shape_no_color_norm"] = df_bias["approx_shape_no_color"]

        st.markdown(
            """
**Colour curves vs overall curves** (for this colorgroup and category levels):

- We compute global, Cat L2, and Cat L3 curves *with* and *without* colour conditioning.  
- Then derive per-size **bias factors** which slightly tilt the shape towards the colour behaviour.
"""
        )

        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.latex(
            r"\text{bias}_{global}(s) = "
            r"\begin{cases}"
            r"\dfrac{global\_color\_ratio(s)}{global\_ratio(s)} & global\_ratio(s) > 0 \\"
            r"1 & \text{otherwise}"
            r"\end{cases}"
        )
        st.latex(r"\text{similarly for } \text{bias}_{cat2}(s), \text{bias}_{cat3}(s)")
        st.latex(
            r"bias(s) = w_g\,\text{bias}_{global}(s) + "
            r"w_2\,\text{bias}_{cat2}(s) + "
            r"w_3\,\text{bias}_{cat3}(s)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        fig_color = go.Figure()
        fig_color.add_bar(
            x=sizes_ordered,
            y=df_bias["approx_shape_no_color_norm"].reindex(sizes_ordered),
            name="Approx. curve without colour bias",
            opacity=0.7,
        )
        fig_color.add_bar(
            x=sizes_ordered,
            y=df_bias["final_ratio"].reindex(sizes_ordered),
            name="Final curve (with colour)",
            opacity=0.9,
        )
        fig_color.update_layout(
            barmode="group",
            xaxis_title="Size",
            yaxis_title="Share of demand",
            title="Effect of colour bias on final curve (approximation)",
            height=420,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_color, use_container_width=True)

        show_cols = [
            "size_type",
            "global_ratio",
            "global_color_ratio",
            "cat_l2_ratio",
            "cat_l2_color_ratio",
            "cat_l3_ratio",
            "cat_l3_color_ratio",
            "bias_global",
            "bias_cat2",
            "bias_cat3",
            "combined_bias_clamped",
            "final_ratio",
            "approx_shape_no_color_norm",
        ]
        df_bias_display = df_bias[show_cols].copy()
        for col in [
            "global_ratio",
            "global_color_ratio",
            "cat_l2_ratio",
            "cat_l2_color_ratio",
            "cat_l3_ratio",
            "cat_l3_color_ratio",
            "final_ratio",
            "approx_shape_no_color_norm",
        ]:
            df_bias_display[col + "_pct"] = df_bias_display[col].apply(_fmt_pct)

        st.markdown("**Per-size colour influence breakdown:**")
        st.dataframe(df_bias_display, use_container_width=True)

        st.markdown(
            '<p class="small-caption">'
            "Note: The `approx_shape_no_color_norm` curve is back-solved from the final curve "
            "and colour bias. It is meant as an illustrative view of "
            "“what the shape would look like without colour”, not an exact internal state."
            "</p>",
            unsafe_allow_html=True,
        )


# ----- TAB 6: Detailed calculations ----- #
with tab6:
    st.subheader("6️⃣ Detailed calculations & reasons")

    st.markdown(
        """
This view shows the **actual numbers** behind each key step for this option:

1. Option, Cat L3, and Cat L2 ROS totals  
2. Cat L3 strength inside Cat L2  
3. Option strength inside Cat L3  
4. How final ratios differ from each baseline  
"""
    )

    df_valid = df_raw.loc[mask_status_ok].copy()

    total_ros_option = opt_total_ros
    total_ros_cat3 = _safe_sum(
        df_valid.loc[df_valid[cat_l3_col] == cat_l3_val, ros_col]
    )
    total_ros_cat2 = _safe_sum(
        df_valid.loc[df_valid[cat_l2_col] == cat_l2_val, ros_col]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total ROS (Option)", f"{total_ros_option:,.1f}")
    c2.metric("Total ROS (Cat L3)", f"{total_ros_cat3:,.1f}")
    c3.metric("Total ROS (Cat L2)", f"{total_ros_cat2:,.1f}")

    # Cat3 strength
    share_cat3 = total_ros_cat3 / total_ros_cat2 if total_ros_cat2 > 0 else 0.0
    cat3_strength_computed = share_cat3 ** CAT3_STRENGTH_POWER

    st.markdown("**Cat L3 strength inside Cat L2**")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"share_{cat3} = \frac{TotalROS(cat3)}{TotalROS(cat2)}"
    )
    st.latex(
        r"cat3\_strength = share_{cat3}^{0.5}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.table(
        pd.DataFrame(
            {
                "metric": [
                    "share_cat3",
                    "cat3_strength (engine)",
                    "cat3_strength (recomputed)",
                ],
                "value": [
                    f"{share_cat3:.3f}",
                    f"{cat3_strength:.3f}",
                    f"{cat3_strength_computed:.3f}",
                ],
            }
        )
    )

    # Option strength vs median in cat3
    df_opt_level = (
        df_result.groupby([cat_l3_col, option_col])["opt_total_ros"]
        .first()
        .reset_index()
    )
    df_opt_level_cat3 = df_opt_level[df_opt_level[cat_l3_col] == cat_l3_val]
    T = float(total_ros_option)
    M = float(df_opt_level_cat3["opt_total_ros"].median()) if len(df_opt_level_cat3) else np.nan

    if np.isfinite(M) and M > 0:
        base = T / (T + M)
    elif T > 0:
        base = 0.5
    else:
        base = 0.0
    option_strength_computed = base ** OPTION_STRENGTH_POWER

    st.markdown("**Option strength vs peers in same Cat L3**")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"T = ROS(option), \quad M = median\_ROS(\text{options in same Cat L3})"
    )
    st.latex(
        r"base = \begin{cases}"
        r"\dfrac{T}{T+M} & M > 0 \\"
        r"0.5 & M \le 0 \text{ and } T > 0 \\"
        r"0 & T = 0"
        r"\end{cases}"
    )
    st.latex(
        r"option\_strength = base^{1.0}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.table(
        pd.DataFrame(
            {
                "metric": [
                    "T = option total ROS",
                    "M = median option ROS in same Cat L3",
                    "option_strength (engine)",
                    "option_strength (recomputed)",
                ],
                "value": [
                    f"{T:,.1f}",
                    f"{M:,.1f}" if np.isfinite(M) else "NA",
                    f"{option_strength:.3f}",
                    f"{option_strength_computed:.3f}",
                ],
            }
        )
    )

    # Per-size deltas
    df_detail = pd.DataFrame(
        {
            "size": sizes_ordered,
            "size_type": size_type_series.reindex(sizes_ordered).values,
            "opt_ratio": opt_curve.reindex(sizes_ordered).values,
            "cat_l3_ratio": cat3_curve.reindex(sizes_ordered).values,
            "cat_l2_ratio": cat2_curve.reindex(sizes_ordered).values,
            "final_ratio": final_curve.reindex(sizes_ordered).values,
        }
    ).set_index("size")

    for col in ["opt_ratio", "cat_l3_ratio", "cat_l2_ratio", "final_ratio"]:
        df_detail[col + "_pct"] = df_detail[col].apply(_fmt_pct)

    df_detail["Δ final - opt"] = df_detail["final_ratio"] - df_detail["opt_ratio"]
    df_detail["Δ final - cat_l3"] = df_detail["final_ratio"] - df_detail["cat_l3_ratio"]
    df_detail["Δ final - cat_l2"] = df_detail["final_ratio"] - df_detail["cat_l2_ratio"]

    st.markdown("**Per-size view: how final curve differs from each baseline**")
    st.dataframe(df_detail, use_container_width=True)

    st.markdown(
        '<p class="small-caption">Positive Δ means final curve is above that baseline, '
        'negative means below. This shows how the engine tilts the curve.</p>',
        unsafe_allow_html=True,
    )


# ----- TAB 7: Compare options ----- #
with tab7:
    st.subheader("7️⃣ Compare two options")

    st.markdown(
        f"Primary option: `{selected_option}` &nbsp;&nbsp; | &nbsp;&nbsp; "
        f"Comparison option: `{compare_option}`",
        unsafe_allow_html=True,
    )

    df_res_a = df_result[df_result[option_col] == selected_option].copy()
    df_res_b = df_result[df_result[option_col] == compare_option].copy()

    if df_res_b.empty:
        st.warning("No rows found for comparison option in engine output.")
    else:
        df_res_a["_size_str"] = df_res_a[size_col].astype(str)
        df_res_b["_size_str"] = df_res_b[size_col].astype(str)

        total_ros_a = df_res_a["opt_total_ros"].iloc[0]
        total_ros_b = df_res_b["opt_total_ros"].iloc[0]
        opt_strength_a = df_res_a["option_strength"].iloc[0]
        opt_strength_b = df_res_b["option_strength"].iloc[0]
        cat3_strength_a = df_res_a["cat3_strength"].iloc[0]
        cat3_strength_b = df_res_b["cat3_strength"].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Option `{selected_option}`**")
            st.table(
                pd.DataFrame(
                    {
                        "metric": ["total ROS", "option_strength", "cat3_strength"],
                        "value": [
                            f"{total_ros_a:,.1f}",
                            f"{opt_strength_a:.3f}",
                            f"{cat3_strength_a:.3f}",
                        ],
                    }
                )
            )
        with col_b:
            st.markdown(f"**Option `{compare_option}`**")
            st.table(
                pd.DataFrame(
                    {
                        "metric": ["total ROS", "option_strength", "cat3_strength"],
                        "value": [
                            f"{total_ros_b:,.1f}",
                            f"{opt_strength_b:.3f}",
                            f"{cat3_strength_b:.3f}",
                        ],
                    }
                )
            )

        # Align sizes
        sizes_union = order_sizes(
            set(df_res_a["_size_str"].unique()) | set(df_res_b["_size_str"].unique())
        )

        final_a = pd.Series(
            df_res_a["final_ratio"].values, index=df_res_a["_size_str"].values
        )
        final_b = pd.Series(
            df_res_b["final_ratio"].values, index=df_res_b["_size_str"].values
        )

        fig_cmp = go.Figure()
        fig_cmp.add_bar(
            x=sizes_union,
            y=final_a.reindex(sizes_union).fillna(0.0),
            name=f"Final curve – {selected_option}",
        )
        fig_cmp.add_bar(
            x=sizes_union,
            y=final_b.reindex(sizes_union).fillna(0.0),
            name=f"Final curve – {compare_option}",
        )
        fig_cmp.update_layout(
            barmode="group",
            xaxis_title="Size",
            yaxis_title="Share of demand",
            title="Final curves (side-by-side)",
            height=420,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        df_cmp = pd.DataFrame(
            {
                "size": sizes_union,
                f"final_{selected_option}": final_a.reindex(sizes_union).fillna(0.0).values,
                f"final_{compare_option}": final_b.reindex(sizes_union).fillna(0.0).values,
            }
        ).set_index("size")
        df_cmp[f"final_{selected_option}_pct"] = df_cmp[
            f"final_{selected_option}"
        ].apply(_fmt_pct)
        df_cmp[f"final_{compare_option}_pct"] = df_cmp[
            f"final_{compare_option}"
        ].apply(_fmt_pct)
        df_cmp["Δ (primary - comparison)"] = (
            df_cmp[f"final_{selected_option}"] - df_cmp[f"final_{compare_option}"]
        )

        st.markdown("**Per-size numeric comparison:**")
        st.dataframe(df_cmp, use_container_width=True)


# ----- TAB 8: Full engine table for this option ----- #
with tab8:
    st.subheader("8️⃣ Full engine output for this option")

    st.markdown(
        "Complete engine output rows for this option & all its sizes."
    )

    st.dataframe(df_opt_res.drop(columns=["_size_str"]), use_container_width=True)
