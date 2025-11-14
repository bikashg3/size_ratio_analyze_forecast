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
      <div class="pill-label">Extended (3XL+)</div>
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

    st.markdown(
        """
This tab explains the **complete logic** behind the size-ratio engine, step by step,
with equations and a numeric walk-through for one size of the currently selected option.

Think of the engine as answering:

> *“Given all my sales history and portfolio behaviour, what is the safest, most
>  business-sensible size curve for this option?”*
"""
    )

    # ---------------- 1. Input data & cleaning ---------------- #
    st.markdown("### 1️⃣ Input data & cleaning")

    st.markdown(
        """
We start from ROS at `(option, stockcode, size)` level.

- Drop rows where `Status = 'Dropped'`.
- All curves and diagnostics are computed only on this **clean base**.
"""
    )

    st.markdown("Mathematically we define:")
    st.latex(r"ROS(o,s) = \text{total ROS for option } o \text{ in size } s.")
    st.latex(
        r"""
ROS(o,s)
= \sum_{\text{stockcode } k} ROS(o,k,s)
"""
    )

    # ---------------- 2. Raw option size curve ---------------- #
    st.markdown("### 2️⃣ Raw option size curve (what the option itself says)")

    st.markdown("For each option \\(o\\) and size \\(s\\):")

    st.latex(
        r"""
opt\_total\_ros(o)
= \sum_{s} ROS(o,s)
"""
    )
    st.latex(
        r"""
opt\_ratio(o,s)
= \frac{ROS(o,s)}{\sum_{s'} ROS(o,s')}
= \frac{ROS(o,s)}{opt\_total\_ros(o)}
"""
    )

    st.markdown(
        """
Business meaning:
- This is **the option’s own raw preference by size**.
- If sizes were frequently out of stock, this curve can be biased or incomplete.
"""
    )

    st.markdown("**Edge cases:**")
    st.latex(
        r"""
opt\_ratio(o,s) =
\begin{cases}
0 & \text{if } opt\_total\_ros(o) = 0 \\
\frac{ROS(o,s)}{opt\_total\_ros(o)} & \text{otherwise}
\end{cases}
"""
    )

    # ---------------- 3. Category baselines ---------------- #
    st.markdown("### 3️⃣ Category baselines (Cat L3 & Cat L2 curves)")

    st.markdown(
        """
We build **portfolio size curves** at two levels:

- \\( c3 \\): Cat L3 (e.g., “Deo-Soft Supima Solid Trunk”)
- \\( c2 \\): Cat L2 (e.g., “Trunk”)
"""
    )

    st.latex(
        r"""
cat\_l3\_ratio(c3,s)
= \frac{ROS(c3,s)}{\sum_{s'} ROS(c3,s')}
"""
    )
    st.latex(
        r"""
cat\_l2\_ratio(c2,s)
= \frac{ROS(c2,s)}{\sum_{s'} ROS(c2,s')}
"""
    )

    st.markdown(
        """
Business meaning:
- **Cat L3 curve**: “How this sub-category sells by size overall.”
- **Cat L2 curve**: “How the full category sells by size.”
- More robust than any single option.
"""
    )

    st.markdown("**Edge cases:**")
    st.latex(
        r"""
cat\_l3\_ratio(c3,s) =
\begin{cases}
0 & \text{if } \sum_{s'} ROS(c3,s') = 0 \\
\frac{ROS(c3,s)}{\sum_{s'} ROS(c3,s')} & \text{otherwise}
\end{cases}
"""
    )
    st.latex(
        r"""
cat\_l2\_ratio(c2,s) =
\begin{cases}
0 & \text{if } \sum_{s'} ROS(c2,s') = 0 \\
\frac{ROS(c2,s)}{\sum_{s'} ROS(c2,s')} & \text{otherwise}
\end{cases}
"""
    )

    # ---------------- 4. Category strength ---------------- #
    st.markdown("### 4️⃣ Category strength (Cat L3 inside Cat L2)")

    st.markdown(
        """
We measure how important this Cat L3 is inside its parent Cat L2.
"""
    )

    st.latex(
        r"""
TotalROS(c3) = \sum_{s} ROS(c3,s), \qquad
TotalROS(c2) = \sum_{s} ROS(c2,s)
"""
    )
    st.latex(
        r"""
share_{cat3}
= \frac{TotalROS(c3)}{TotalROS(c2)}
"""
    )
    st.latex(
        r"""
cat3\_strength
= share_{cat3}^{0.5}
"""
    )

    st.markdown(
        """
- If a Cat L3 contributes a lot to Cat L2, \\( cat3\\_strength \\) is higher.
- The square root dampens very large categories so they don’t dominate everything.
"""
    )

    st.markdown(
        "**Edge case:** if \\( TotalROS(c2) = 0 \\Rightarrow share_{cat3} = 0 "
        "\\Rightarrow cat3\\_strength = 0.**"
    )

    # ---------------- 5. Option strength ---------------- #
    st.markdown("### 5️⃣ Option strength (how strong is this option vs peers)")

    st.markdown(
        """
We compare the option’s ROS versus the **median option ROS in the same Cat L3**.
"""
    )

    st.latex(
        r"""
T = opt\_total\_ros(o), \qquad
M = \operatorname{median}\{\;opt\_total\_ros(o')\;|\;o' \in \text{same Cat L3}\;\}
"""
    )
    st.latex(
        r"""
base =
\begin{cases}
\dfrac{T}{T + M} & M > 0 \\
0.5 & M \le 0 \text{ and } T > 0 \\
0 & T = 0
\end{cases}
"""
    )
    st.latex(
        r"""
option\_strength = base^{1.0} = base
"""
    )

    st.markdown(
        """
- High ROS option → \\( option\\_strength \\) closer to **1**.
- Weak option → \\( option\\_strength \\) closer to **0**.
"""
    )

    # ---------------- 6. Core vs fringe & core coverage ---------------- #
    st.markdown("### 6️⃣ Core vs fringe sizes & core coverage")

    st.markdown("**Step 6.1 – Infer portfolio core sizes (Cat L3 / Cat L2)**")

    st.markdown(
        """
For each Cat L3:

- Sort sizes by \\( cat\_l3\_ratio(c3,s) \\) descending.
- Pick minimal set of sizes whose cumulative share ≥ **80%**  
  (subject to a min/max number of sizes).
- Those sizes form **core** for that sub-category; rest are **fringe**.

If Cat L3 is too thin, we fall back to Cat L2 and/or a default {M, L, XL}.
"""
    )

    st.markdown("**Step 6.2 – Tag sizes per option**")

    st.markdown(
        """
- For each option, intersect available sizes with the **expected core set**.
- `size_type` per row: `core`, `fringe`, or `one_size` (if option has only one size).
"""
    )

    st.markdown("**Step 6.3 – Core coverage for the option**")

    st.markdown(
        """
Let:
- \\( C_{\\text{expected}} \\): set of expected core sizes for this Cat L3.
- \\( C_{\\text{present}} \\): core sizes actually present for this option.
"""
    )
    st.latex(
        r"""
core\_coverage =
\begin{cases}
\dfrac{|C_{\text{present}}|}{|C_{\text{expected}}|} & |C_{\text{expected}}| > 0 \\
1 & |C_{\text{expected}}| = 0
\end{cases}
"""
    )

    st.markdown(
        """
If the option is missing key core sizes, \\( core\\_coverage < 1 \\) and the engine
penalises option influence:
"""
    )
    st.latex(
        r"""
option\_strength\_used
= \min\left(1,\; \max\left(0,\; option\_strength \times core\_coverage\right)\right)
"""
    )

    # ---------------- 7. Weight formulas: w_opt(s), w_cat3(s), w_cat2(s) ---------------- #
    st.markdown("### 7️⃣ Size-level weights: \\(w_{opt}(s), w_{cat3}(s), w_{cat2}(s)\\)")

    st.markdown(
        """
Weights depend on:

- `size_type` (core vs fringe vs one_size),
- \\( option\\_strength\\_used \\),
- \\( cat3\\_strength \\),
- and hyperparameters:  
  \\( opt\\_core\\_min = 0.20, opt\\_core\\_max = 0.60 \\)  
  \\( opt\\_fringe\\_min = 0.05, opt\\_fringe\\_max = 0.25 \\)
"""
    )

    st.markdown("**For core sizes (and one_size treated as core-ish):**")
    st.latex(
        r"""
w_{opt}^{core}(s)
= opt\_core\_min
+ \left(opt\_core\_max - opt\_core\_min\right) \cdot option\_strength\_used
"""
    )
    st.latex(
        r"""
w_{cat\_total}^{core}(s)
= 1 - w_{opt}^{core}(s)
"""
    )
    st.latex(
        r"""
f_{cat3}^{core}
= \operatorname{clip}\bigl(0.50 + 0.10 \cdot cat3\_strength,\; 0,\; 1\bigr)
"""
    )
    st.latex(
        r"""
w_{cat3}^{core}(s) = w_{cat\_total}^{core}(s) \cdot f_{cat3}^{core},
\qquad
w_{cat2}^{core}(s) = w_{cat\_total}^{core}(s) \cdot \bigl(1 - f_{cat3}^{core}\bigr)
"""
    )

    st.markdown("**For fringe sizes:**")
    st.latex(
        r"""
w_{opt}^{fringe}(s)
= opt\_fringe\_min
+ \left(opt\_fringe\_max - opt\_fringe\_min\right) \cdot option\_strength\_used
"""
    )
    st.latex(
        r"""
w_{cat\_total}^{fringe}(s) = 1 - w_{opt}^{fringe}(s)
"""
    )
    st.latex(
        r"""
f_{cat3}^{fringe}
= \operatorname{clip}\bigl(0.45 + 0.05 \cdot cat3\_strength,\; 0,\; 1\bigr)
"""
    )
    st.latex(
        r"""
w_{cat3}^{fringe}(s) = w_{cat\_total}^{fringe}(s) \cdot f_{cat3}^{fringe},
\qquad
w_{cat2}^{fringe}(s) = w_{cat\_total}^{fringe}(s) \cdot \bigl(1 - f_{cat3}^{fringe}\bigr)
"""
    )

    st.markdown(
        """
In code:

- `w_opt_size`, `w_cat3_size`, `w_cat2_size` are exactly these \\(w_{opt}(s), w_{cat3}(s), w_{cat2}(s)\\)
  applied per size row, using `size_type` to pick core vs fringe formula.
- `one_size` uses the **core** formula.
"""
    )

    # ---------------- 8. Shape before colour ---------------- #
    st.markdown("### 8️⃣ Blended shape before colour")

    st.markdown(
        """
Using the weights we blend **option**, **Cat L3** and **Cat L2** curves.
"""
    )

    st.latex(
        r"""
shape\_raw(s)
= w_{opt}(s)\,opt\_ratio(o,s)
+ w_{cat3}(s)\,cat\_l3\_ratio(c3,s)
+ w_{cat2}(s)\,cat\_l2\_ratio(c2,s)
"""
    )
    st.latex(
        r"""
shape\_norm\_no\_color(s)
= \frac{shape\_raw(s)}{\sum_{s'} shape\_raw(s')}
"""
    )

    st.markdown(
        """
Business meaning:
- For **strong options in core sizes**, \\(w_{opt}(s)\\) is higher.
- For **weak options / fringe sizes**, more weight flows to Cat L3 & Cat L2.
"""
    )

    st.markdown(
        "**Edge case:** if \\( \sum_{s'} shape\_raw(s') \le 0 \\) we use a uniform split "
        "across sizes."
    )

    # ---------------- 9. Colour behaviour ---------------- #
    st.markdown("### 9️⃣ Colour behaviour & multiplicative bias")

    st.markdown(
        """
When colour is available, we compare colour-specific size curves vs the base curves.
"""
    )

    st.latex(
        r"""
global\_ratio(s),\; global\_color\_ratio(s)
"""
    )
    st.latex(
        r"""
bias_{global}(s)
=
\begin{cases}
\dfrac{global\_color\_ratio(s)}{global\_ratio(s)} & global\_ratio(s) > 0 \\
1 & global\_ratio(s) = 0
\end{cases}
"""
    )
    st.latex(
        r"""
bias_{cat2}(s)
=
\begin{cases}
\dfrac{cat\_l2\_color\_ratio(c2,s)}{cat\_l2\_ratio(c2,s)} & cat\_l2\_ratio(c2,s) > 0 \\
1 & cat\_l2\_ratio(c2,s) = 0
\end{cases}
"""
    )
    st.latex(
        r"""
bias_{cat3}(s)
=
\begin{cases}
\dfrac{cat\_l3\_color\_ratio(c3,s)}{cat\_l3\_ratio(c3,s)} & cat\_l3\_ratio(c3,s) > 0 \\
1 & cat\_l3\_ratio(c3,s) = 0
\end{cases}
"""
    )

    st.markdown(
        """
We compute **colour weights** from volume shares. The exact scoring is piecewise,
but in simplified form:
"""
    )
    st.latex(
        r"""
w\_{col,global}
= \frac{score_{global}}{score_{global} + score_{cat2} + score_{cat3}},\quad
w\_{col,cat2}
= \frac{score_{cat2}}{score_{global} + score_{cat2} + score_{cat3}},\quad
w\_{col,cat3}
= \frac{score_{cat3}}{score_{global} + score_{cat2} + score_{cat3}}
"""
    )

    st.latex(
        r"""
bias\_raw(s)
= w\_{col,global}\,bias_{global}(s)
+ w\_{col,cat2}\,bias_{cat2}(s)
+ w\_{col,cat3}\,bias_{cat3}(s)
"""
    )

    st.markdown("We clamp the bias differently for core vs fringe:")
    st.latex(
        r"""
bias\_{clamped}(s) =
\begin{cases}
\operatorname{clip}\bigl(bias\_raw(s),\; 0.7,\; 1.3\bigr) & s \text{ is core or one\_size} \\
\operatorname{clip}\bigl(bias\_raw(s),\; 0.8,\; 1.2\bigr) & s \text{ is fringe}
\end{cases}
"""
    )

    st.latex(
        r"""
shape\_with\_color\_raw(s)
= shape\_norm\_no\_color(s) \times bias\_{clamped}(s)
"""
    )
    st.latex(
        r"""
shape\_with\_color\_norm(s)
= \frac{shape\_with\_color\_raw(s)}{\sum_{s'} shape\_with\_color\_raw(s')}
"""
    )

    st.markdown(
        """
Business meaning:
- Colour can nudge sizes up/down by **±20–30%**, but cannot explode or kill a size.
"""
    )

    # ---------------- 10. Core share guardrails ---------------- #
    st.markdown("### 🔟 Core share guardrails")

    st.markdown(
        """
We ensure total share on core sizes stays in a safe band:

- \\( core\_share\_{min} = 0.70 \\)
- \\( core\_share\_{max} = 0.85 \\)
"""
    )

    st.latex(
        r"""
core\_share
= \sum_{s \in core} shape\_with\_color\_norm(s)
"""
    )

    st.latex(
        r"""
\text{target} =
\begin{cases}
core\_share\_{min} & \text{if } core\_share < core\_share\_{min} \\
core\_share\_{max} & \text{if } core\_share > core\_share\_{max} \\
core\_share & \text{otherwise}
\end{cases}
"""
    )

    st.markdown(
        """
We rescale core vs fringe blocks while keeping **within-core** and **within-fringe**
shape unchanged.
"""
    )

    st.latex(
        r"""
\sum_{s \in core} \alpha\,c(s) = \text{target}, \qquad
\sum_{s \notin core} \beta\,f(s) = 1 - \text{target}
"""
    )
    st.latex(
        r"""
final\_after\_core(s) =
\begin{cases}
\alpha\,c(s) & s \in core \\
\beta\,f(s) & s \notin core
\end{cases}
"""
    )

    # ---------------- 11. Extended size floors ---------------- #
    st.markdown("### 1️⃣1️⃣ Extended size floors (3XL and above)")

    st.markdown(
        """
For extended sizes (e.g., 3XL, 4XL, 5XL, 6XL), we set a **minimum floor** vs Cat L2+colour.
"""
    )

    st.latex(
        r"""
floor(s)
= tail\_floor\_factor \times cat2\_color\_baseline(s),
\qquad tail\_floor\_factor = 0.7
"""
    )
    st.latex(
        r"""
final\_after\_tail(s)
= \max\left(final\_after\_core(s),\ floor(s)\right)
"""
    )

    st.markdown(
        """
Then we renormalise over all sizes.

Business meaning:
- Extended sizes never go to **exactly zero** if there is any reasonable Cat L2 evidence.
"""
    )

    # ---------------- 12. Final normalisation ---------------- #
    st.markdown("### 1️⃣2️⃣ Final normalisation & output")

    st.latex(
        r"""
final\_ratio(o,s)
= \frac{final\_after\_tail(s)}{\sum_{s'} final\_after\_tail(s')}
"""
    )

    st.markdown(
        """
The engine outputs one row per `(option, size)` with:

- `final_ratio(o,s)`
- `ratio_source` (what dominated the final shape)
- all intermediate diagnostics (strengths, weights, biases, guardrails, floors).
"""
    )

        # ---------------- 13. Worked example with user-chosen size ---------------- #
    st.markdown("---")
    st.markdown("### 1️⃣3️⃣ Worked example: choose a size and see all steps")

    with st.expander(
        "🔍 Click to see a numeric walk-through for a specific size of this option",
        expanded=False,
    ):
        try:
            if df_opt_res.empty:
                st.info("No diagnostics available for this option.")
            else:
                # All rows for this option
                full_opt = df_opt_res.copy()
                full_opt = full_opt.sort_values("final_ratio", ascending=False)

                # Let user choose size
                size_choices = list(full_opt["_size_str"].unique())
                size_choices_sorted = order_sizes(size_choices)
                default_idx = 0

                chosen_size = st.selectbox(
                    "Choose the size for which you want to see the working:",
                    options=size_choices_sorted,
                    index=default_idx,
                )

                # Row for chosen size (fallback to best if missing)
                df_size = full_opt[full_opt["_size_str"] == chosen_size]
                if df_size.empty:
                    ex_row = full_opt.iloc[0]
                    ex_size = str(ex_row["_size_str"])
                else:
                    ex_row = df_size.iloc[0]
                    ex_size = chosen_size

                size_type_ex = ex_row["size_type"]

                st.markdown(
                    f"**Example:** option `{selected_option}`, "
                    f"size `{ex_size}` (tagged as **{size_type_ex}**)."
                )

                # ---------- Step A: raw ROS & option curve ----------
                mask_ex = (
                    mask_status_ok
                    & (df_raw[option_col] == selected_option)
                    & (df_raw[size_col].astype(str) == ex_size)
                )
                ros_ex = float(df_raw.loc[mask_ex, ros_col].sum())
                opt_total = float(ex_row["opt_total_ros"])
                opt_ratio_ex = float(ex_row["opt_ratio"])

                st.markdown("**Step 1 – Raw option curve for this size**")
                st.latex(
                    r"opt\_total\_ros(o) = \sum_{s} ROS(o,s)"
                )
                st.latex(
                    r"opt\_ratio(o,s) = \frac{ROS(o,s)}{opt\_total\_ros(o)}"
                )
                st.latex(
                    r"opt\_ratio(o,s) = \frac{%.2f}{%.2f} \approx %.3f"
                    % (ros_ex, opt_total, opt_ratio_ex)
                )
                st.markdown(
                    f"- This size contributes **{_fmt_pct(opt_ratio_ex)}** of ROS for this option."
                )

                # ---------- Step B: category baselines ----------
                cat3_total = float(ex_row["cat_l3_total_ros"])
                cat3_ratio_ex = float(ex_row["cat_l3_ratio"])
                cat3_ros_ex = cat3_total * cat3_ratio_ex

                cat2_total = float(ex_row["cat_l2_total_ros"])
                cat2_ratio_ex = float(ex_row["cat_l2_ratio"])
                cat2_ros_ex = cat2_total * cat2_ratio_ex

                st.markdown("**Step 2 – Category baselines for this size**")
                st.latex(
                    r"cat\_l3\_ratio(c3,s) = \frac{ROS(c3,s)}{TotalROS(c3)}"
                )
                st.latex(
                    r"cat\_l3\_ratio(c3,s) = \frac{%.2f}{%.2f} \approx %.3f"
                    % (cat3_ros_ex, cat3_total, cat3_ratio_ex)
                )
                st.latex(
                    r"cat\_l2\_ratio(c2,s) = \frac{ROS(c2,s)}{TotalROS(c2)}"
                )
                st.latex(
                    r"cat\_l2\_ratio(c2,s) = \frac{%.2f}{%.2f} \approx %.3f"
                    % (cat2_ros_ex, cat2_total, cat2_ratio_ex)
                )

                # ---------- Step C: strengths ----------
                T_ex = opt_total
                M_ex = float(ex_row["median_opt_ros_in_cat3"])
                option_strength_ex = float(ex_row["option_strength"])
                cat3_strength_ex = float(ex_row["cat3_strength"])
                core_coverage_ex = float(ex_row["core_coverage"])
                option_strength_used_ex = float(ex_row["option_strength_used"])

                st.markdown("**Step 3 – Option & Cat L3 strengths**")
                st.latex(
                    r"""
base =
\begin{cases}
\dfrac{T}{T + M} & M > 0 \\
0.5 & M \le 0 \text{ and } T > 0 \\
0 & T = 0
\end{cases}
"""
                )
                st.latex(
                    "T = %.2f, \\quad M = %.2f" % (T_ex, M_ex)
                )
                st.latex(
                    r"option\_strength \approx %.3f" % option_strength_ex
                )
                st.latex(
                    r"core\_coverage \approx %.3f" % core_coverage_ex
                )
                st.latex(
                    r"option\_strength\_used = option\_strength \times core\_coverage \approx %.3f"
                    % option_strength_used_ex
                )
                st.latex(
                    r"cat3\_strength \approx %.3f" % cat3_strength_ex
                )

                # ---------- Step D: size-level weights ----------
                w_opt_ex = float(ex_row["w_opt_size"])
                w_c3_ex = float(ex_row["w_cat3_size"])
                w_c2_ex = float(ex_row["w_cat2_size"])

                st.markdown("**Step 4 – Weights for this size**")
                st.latex(
                    r"""
shape\_raw(s)
= w_{opt}(s)\,opt\_ratio(o,s)
+ w_{cat3}(s)\,cat\_l3\_ratio(c3,s)
+ w_{cat2}(s)\,cat\_l2\_ratio(c2,s)
"""
                )
                st.latex(
                    r"w_{opt}(s) \approx %.3f,\quad w_{cat3}(s) \approx %.3f,\quad w_{cat2}(s) \approx %.3f"
                    % (w_opt_ex, w_c3_ex, w_c2_ex)
                )

                shape_raw_ex = float(ex_row["shape_raw"])
                shape_norm_no_color_ex = float(ex_row["shape_norm_no_color"])

                st.latex(
                    r"shape\_raw(s) \approx (%.3f)(%.3f) + (%.3f)(%.3f) + (%.3f)(%.3f) \approx %.4f"
                    % (
                        w_opt_ex,
                        opt_ratio_ex,
                        w_c3_ex,
                        cat3_ratio_ex,
                        w_c2_ex,
                        cat2_ratio_ex,
                        shape_raw_ex,
                    )
                )
                st.latex(
                    r"shape\_norm\_no\_color(s) = \frac{shape\_raw(s)}{\sum_{s'} shape\_raw(s')} \approx %.4f"
                    % shape_norm_no_color_ex
                )
                st.markdown(
                    f"- Before colour, this size gets about "
                    f"**{_fmt_pct(shape_norm_no_color_ex)}** of the mix."
                )

                # ---------- Step E: colour bias ----------
                global_ratio_ex = float(ex_row["global_ratio"])
                global_color_ratio_ex = float(ex_row["global_color_ratio"])
                cat2_color_ratio_ex = float(ex_row["cat_l2_color_ratio"])
                cat3_color_ratio_ex = float(ex_row["cat_l3_color_ratio"])

                if global_ratio_ex > 0:
                    bias_g_calc = global_color_ratio_ex / global_ratio_ex
                else:
                    bias_g_calc = 1.0

                if cat2_ratio_ex > 0:
                    bias_2_calc = cat2_color_ratio_ex / cat2_ratio_ex
                else:
                    bias_2_calc = 1.0

                if cat3_ratio_ex > 0:
                    bias_3_calc = cat3_color_ratio_ex / cat3_ratio_ex
                else:
                    bias_3_calc = 1.0

                w_col_g = float(ex_row["w_color_global"])
                w_col_2 = float(ex_row["w_color_cat2"])
                w_col_3 = float(ex_row["w_color_cat3"])
                bias_total_clamped_ex = float(ex_row["bias_total_clamped"])

                st.markdown("**Step 5 – Colour bias for this size**")
                st.latex(
                    r"""
bias\_raw(s)
= w\_{col,global}\,bias_{global}(s)
+ w\_{col,cat2}\,bias_{cat2}(s)
+ w\_{col,cat3}\,bias_{cat3}(s)
"""
                )
                st.latex(
                    r"bias_{global}(s) \approx %.3f,\quad bias_{cat2}(s) \approx %.3f,\quad bias_{cat3}(s) \approx %.3f"
                    % (bias_g_calc, bias_2_calc, bias_3_calc)
                )
                st.latex(
                    r"w\_{col,global} \approx %.3f,\quad w\_{col,cat2} \approx %.3f,\quad w\_{col,cat3} \approx %.3f"
                    % (w_col_g, w_col_2, w_col_3)
                )
                st.latex(
                    r"bias\_{clamped}(s) \approx %.3f" % bias_total_clamped_ex
                )

                shape_with_color_raw_ex = float(ex_row["shape_with_color_raw"])
                shape_with_color_norm_ex = float(ex_row["shape_with_color_norm"])

                st.markdown("**Step 6 – Apply colour to the blended shape**")
                st.latex(
                    r"shape\_with\_color\_raw(s) = shape\_norm\_no\_color(s) \times bias\_{clamped}(s)"
                )
                st.latex(
                    r"shape\_with\_color\_raw(s) \approx (%.4f)(%.3f) \approx %.4f"
                    % (
                        shape_norm_no_color_ex,
                        bias_total_clamped_ex,
                        shape_with_color_raw_ex,
                    )
                )
                st.latex(
                    r"shape\_with\_color\_norm(s) \approx %.4f"
                    % shape_with_color_norm_ex
                )

                # ---------- Step G: core guardrails ----------
                df_ex = full_opt  # all sizes, this option
                core_share_before = float(
                    df_ex["core_share_before_guardrail"].iloc[0]
                )
                core_share_after = float(
                    df_ex["core_share_after_guardrail"].iloc[0]
                )
                before_core_ex = float(ex_row["final_before_core_guardrail"])
                after_core_ex = float(ex_row["final_after_core_guardrail"])

                st.markdown("**Step 7 – Core guardrails on this option**")
                st.latex(
                    r"core\_share\_{\text{before}} \approx %.3f,\quad core\_share\_{\text{after}} \approx %.3f"
                    % (core_share_before, core_share_after)
                )

                if abs(core_share_before - core_share_after) > 1e-6:
                    st.markdown(
                        f"- Guardrail adjusted core share from "
                        f"**{_fmt_pct(core_share_before)}** to "
                        f"**{_fmt_pct(core_share_after)}**."
                    )
                    st.latex(
                        r"final\_after\_core(s):\; %.4f \rightarrow %.4f"
                        % (before_core_ex, after_core_ex)
                    )
                else:
                    st.markdown(
                        "- Core share already within [70%, 85%], so no rescaling was needed."
                    )

                # ---------- Step H: tail floors & final ratio ----------
                tail_floor_ex = float(ex_row["tail_floor"])
                after_tail_ex = float(ex_row["final_after_tail_floors"])
                final_ratio_ex = float(ex_row["final_ratio"])

                st.markdown("**Step 8 – Extended size floor & final ratio**")
                if tail_floor_ex > 0 and ex_size.upper() in {"3XL", "4XL", "5XL", "6XL"}:
                    st.markdown(
                        f"- `{ex_size}` is an extended size with a floor of "
                        f"{_fmt_pct(tail_floor_ex)} based on Cat L2+colour."
                    )
                    st.latex(
                        r"final\_after\_tail(s) = \max(%.4f,\ %.4f) = %.4f"
                        % (after_core_ex, tail_floor_ex, after_tail_ex)
                    )
                else:
                    st.markdown(
                        "- For this size, the extended-size floor is not binding."
                    )

                st.latex(
                    r"final\_ratio(s) \approx %.4f" % final_ratio_ex
                )
                st.markdown(
                    f"👉 **Final recommendation:** allocate about "
                    f"**{_fmt_pct(final_ratio_ex)}** of this option’s buy to size `{ex_size}`."
                )

        except Exception as e:
            st.warning(
                "Could not build the worked example for this option due to missing or "
                f"incompatible diagnostics.\n\nError: {e}"
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
        r"core\_share \in [core\_share\_{min}, core\_share\_{max}] = [0.70, 0.85]"
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
# ----- TAB 6: Detailed calculations ----- #
with tab6:
    st.subheader("6️⃣ Detailed calculations & reasons")

    st.markdown(
        """
This tab exposes the **actual internal numbers** that the engine used for this option:

1. ROS totals and strengths  
2. How we compute **option_strength** and **cat3_strength**  
3. Per-size blending weights between *option / Cat L3 / Cat L2*  
4. Per-size **colour bias** factors and colour weights  
5. Core-share guardrails (before vs after)  
6. Extended size floors and the final curve  
"""
    )

    # ----- 6.1 Total ROS and strengths -----
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

    # Cat3 strength from ROS share
    share_cat3 = total_ros_cat3 / total_ros_cat2 if total_ros_cat2 > 0 else 0.0
    cat3_strength_computed = share_cat3 ** CAT3_STRENGTH_POWER
    cat3_strength_engine = float(df_opt_res["cat3_strength"].iloc[0])

    st.markdown("### 6.1.1 Cat L3 strength inside Cat L2")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r"share_{cat3} = \frac{TotalROS(cat3)}{TotalROS(cat2)}")
    st.latex(r"cat3\_strength = share_{cat3}^{0.5}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.table(
        pd.DataFrame(
            {
                "metric": [
                    "share_cat3 = TotalROS(cat3)/TotalROS(cat2)",
                    "cat3_strength (engine)",
                    "cat3_strength (recomputed)",
                ],
                "value": [
                    f"{share_cat3:.3f}",
                    f"{cat3_strength_engine:.3f}",
                    f"{cat3_strength_computed:.3f}",
                ],
            }
        )
    )

    # Option strength vs peers in same Cat L3
    df_opt_level_all = (
        df_result.groupby([cat_l3_col, option_col])["opt_total_ros"]
        .first()
        .reset_index()
    )
    df_opt_level_cat3 = df_opt_level_all[df_opt_level_all[cat_l3_col] == cat_l3_val]

    T = float(total_ros_option)
    M = (
        float(df_opt_level_cat3["opt_total_ros"].median())
        if len(df_opt_level_cat3)
        else np.nan
    )

    if np.isfinite(M) and M > 0:
        base = T / (T + M)
    elif T > 0:
        base = 0.5
    else:
        base = 0.0
    option_strength_computed = base ** OPTION_STRENGTH_POWER
    option_strength_engine = float(df_opt_res["option_strength"].iloc[0])
    option_strength_used = float(df_opt_res["option_strength_used"].iloc[0])
    core_coverage = float(df_opt_res["core_coverage"].iloc[0])

    st.markdown("### 6.1.2 Option strength vs peers in same Cat L3")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r"T = ROS(option), \quad M = median\_ROS(\text{options in same Cat L3})")
    st.latex(
        r"""
base =
\begin{cases}
\dfrac{T}{T+M} & M > 0 \\
0.5            & M \le 0 \text{ and } T > 0 \\
0              & T = 0
\end{cases}
"""
    )
    st.latex(r"option\_strength = base^{1.0}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.table(
        pd.DataFrame(
            {
                "metric": [
                    "T = option total ROS",
                    "M = median option ROS in same Cat L3",
                    "base = T / (T + M)",
                    "option_strength (engine)",
                    "option_strength (recomputed)",
                    "core_coverage (actual vs expected core sizes)",
                    "option_strength_used = option_strength × core_coverage",
                ],
                "value": [
                    f"{T:,.1f}",
                    f"{M:,.1f}" if np.isfinite(M) else "NA",
                    f"{base:.3f}",
                    f"{option_strength_engine:.3f}",
                    f"{option_strength_computed:.3f}",
                    f"{core_coverage:.3f}",
                    f"{option_strength_used:.3f}",
                ],
            }
        )
    )

    st.markdown("---")

    # ----- 6.2 Per-size blending: option vs Cat L3 vs Cat L2 -----
    st.markdown("### 6.2 Per-size blending: option vs Cat L3 vs Cat L2")

    st.markdown(
        """
For each size we compute weights between **option**, **Cat L3**, and **Cat L2** curves,
then blend them into a base shape **before colour**.
"""
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"shape_{\text{no\_color}}(s) = "
        r"w_{opt}(s)\,opt\_ratio(s) + "
        r"w_{c3}(s)\,cat\_l3\_ratio(s) + "
        r"w_{c2}(s)\,cat\_l2\_ratio(s)"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    cols_blend = [
        "_size_str",
        "size_type",
        "opt_ratio",
        "cat_l3_ratio",
        "cat_l2_ratio",
        "global_ratio",
        "w_opt_size",
        "w_cat3_size",
        "w_cat2_size",
        "shape_raw",
        "shape_norm_no_color",
    ]

    df_step_blend = (
        df_opt_res[cols_blend]
        .set_index("_size_str")
        .reindex(sizes_ordered)
    )
    df_step_blend.index.name = "size"

    # % versions for the ratios / shapes
    for col in [
        "opt_ratio",
        "cat_l3_ratio",
        "cat_l2_ratio",
        "global_ratio",
        "shape_raw",
        "shape_norm_no_color",
    ]:
        df_step_blend[col + "_pct"] = df_step_blend[col].apply(_fmt_pct)

    df_step_blend = df_step_blend.rename(
        columns={
            "size_type": "size_type",
            "w_opt_size": "w_opt",
            "w_cat3_size": "w_cat3",
            "w_cat2_size": "w_cat2",
        }
    )

    st.markdown("**Per-size blend before colour:**")
    st.dataframe(df_step_blend, use_container_width=True)

    st.markdown(
        '<p class="small-caption">'
        "`shape_raw` is the un-normalised blend, `shape_norm_no_color` is the "
        "normalised curve used as the base before applying colour bias.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ----- 6.3 Colour bias (per size) -----
    st.markdown("### 6.3 Colour bias: how the colour tilts the curve")

    has_color = colorgroup_col in df_raw.columns and not pd.isna(color_val)
    if not has_color:
        st.info(
            "This option has no colour group assigned. Colour bias is neutral "
            "(all bias factors = 1, final shape = base shape)."
        )

    w_col_g = float(df_opt_res["w_color_global"].iloc[0])
    w_col_2 = float(df_opt_res["w_color_cat2"].iloc[0])
    w_col_3 = float(df_opt_res["w_color_cat3"].iloc[0])

    st.markdown(
        f"""
**Colour weights across levels** (same for all sizes in this option):

- Weight to **Global colour behaviour**: `{w_col_g:.3f}`  
- Weight to **Cat L2 colour behaviour**: `{w_col_2:.3f}`  
- Weight to **Cat L3 colour behaviour**: `{w_col_3:.3f}`  
"""
    )

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(
        r"\text{bias}_{global}(s) = \frac{global\_color\_ratio(s)}{global\_ratio(s)}"
    )
    st.latex(
        r"\text{bias}_{cat2}(s) = \frac{cat2\_color\_ratio(s)}{cat2\_ratio(s)},\quad"
        r"\text{bias}_{cat3}(s) = \frac{cat3\_color\_ratio(s)}{cat3\_ratio(s)}"
    )
    st.latex(
        r"bias_{\text{combined}}(s) ="
        r" w_g\,\text{bias}_{global}(s) +"
        r" w_2\,\text{bias}_{cat2}(s) +"
        r" w_3\,\text{bias}_{cat3}(s)"
    )
    st.latex(
        r"bias_{\text{clamped}}(s) \in"
        r" [0.7, 1.3] \text{ (core)},\ [0.8, 1.2] \text{ (fringe)}"
    )
    st.latex(
        r"shape_{\text{with\_color}}(s) \propto"
        r" shape_{\text{no\_color}}(s) \times bias_{\text{clamped}}(s)"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    cols_color = [
        "_size_str",
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
        "bias_total_clamped",
        "shape_norm_no_color",
        "shape_with_color_norm",
    ]
    df_step_color = (
        df_opt_res[cols_color]
        .set_index("_size_str")
        .reindex(sizes_ordered)
    )
    df_step_color.index.name = "size"

    for col in [
        "global_ratio",
        "global_color_ratio",
        "cat_l2_ratio",
        "cat_l2_color_ratio",
        "cat_l3_ratio",
        "cat_l3_color_ratio",
        "shape_norm_no_color",
        "shape_with_color_norm",
    ]:
        df_step_color[col + "_pct"] = df_step_color[col].apply(_fmt_pct)

    st.markdown("**Per-size colour bias breakdown:**")
    st.dataframe(df_step_color, use_container_width=True)

    st.markdown(
        '<p class="small-caption">'
        "`bias_total_clamped` is the actual multiplicative factor applied on the base "
        "shape for that size, after mixing global/Cat L2/Cat L3 colour behaviour and "
        "clamping to safe ranges.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ----- 6.4 Core guardrails & tail floors -----
    st.markdown("### 6.4 Core guardrails & extended-size floors")

    core_share_before = float(df_opt_res["core_share_before_guardrail"].iloc[0])
    core_share_after = float(df_opt_res["core_share_after_guardrail"].iloc[0])

    st.markdown("#### 6.4.1 Core-share guardrails")

    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r"core\_share = \sum_{s \in core} final\_ratio(s)")
    st.latex(r"core\_share \in [0.70,\ 0.85]")
    st.markdown("</div>", unsafe_allow_html=True)

    st.table(
        pd.DataFrame(
            {
                "metric": [
                    "core_share BEFORE guardrail",
                    "core_share AFTER guardrail",
                ],
                "value": [
                    f"{core_share_before:.3f} ({_fmt_pct(core_share_before)})",
                    f"{core_share_after:.3f} ({_fmt_pct(core_share_after)})",
                ],
            }
        )
    )

    st.markdown("#### 6.4.2 Tail floors for extended sizes (3XL+)")

    df_guard = (
        df_opt_res[
            [
                "_size_str",
                "size_type",
                "final_before_core_guardrail",
                "final_after_core_guardrail",
                "final_before_tail_floors",
                "tail_floor",
                "final_ratio",
            ]
        ]
        .set_index("_size_str")
        .reindex(sizes_ordered)
    )
    df_guard.index.name = "size"

    for col in [
        "final_before_core_guardrail",
        "final_after_core_guardrail",
        "final_before_tail_floors",
        "tail_floor",
        "final_ratio",
    ]:
        df_guard[col + "_pct"] = df_guard[col].apply(_fmt_pct)

    st.markdown(
        """
**Per-size guardrail & floor view:**

- `final_before_core_guardrail`: curve after colour, before core share adjustment  
- `final_after_core_guardrail`: after rebalancing core vs fringe  
- `final_before_tail_floors`: after core guardrail, before extended-size floors  
- `tail_floor`: minimum floor applied vs Cat L2+colour curve (non-zero only for extended sizes)  
- `final_ratio`: final engine curve used everywhere else  
"""
    )

    st.dataframe(df_guard, use_container_width=True)

    st.markdown(
        '<p class="small-caption">'
        "This table shows exactly how each size moves from raw blended shape, to "
        "colour-adjusted curve, to core-guardrailed curve, and finally to the curve "
        "with extended-size floors applied.</p>",
        unsafe_allow_html=True,
    )

        # ----- 6.5 Per-size step explainer & dominant driver -----
    st.markdown("---")
    st.markdown("### 6.5 Per-size step explainer & dominant driver")

    # Treat |Δ| < 0.2 percentage points as "no meaningful change"
    EPS = 0.002

    # One-size-only flag (from engine diagnostics)
    is_one_size_only_flag = bool(
        df_opt_res["is_one_size_only"].iloc[0]
    ) if "is_one_size_only" in df_opt_res.columns else False

    def _movement_label(delta: float, eps: float = EPS) -> str:
        if delta > eps:
            return "up"
        elif delta < -eps:
            return "down"
        return "flat"

    def _build_reason_tag(driver_key: str, movement: str) -> str:
        if driver_key == "one_size":
            return "One-size-only"
        if driver_key == "neutral":
            return "Neutral (tiny change)"

        if driver_key == "shape":
            base = "Shape"
        elif driver_key == "colour":
            base = "Colour"
        elif driver_key == "guardrail":
            base = "Core guardrail"
        elif driver_key == "floor":
            base = "Extended floor"
        else:
            base = "Mixed"

        if movement == "up":
            return f"{base} uplift"
        elif movement == "down":
            return f"{base} trim"
        else:
            return f"{base} neutral"

    def _build_step_explainer(row) -> str:
        """
        Business-language explanation of why this size moved vs the raw option curve,
        based on which step had the biggest effect.
        """
        if is_one_size_only_flag or row["dominant_driver_key"] == "one_size":
            return (
                "One-size-only option: 100% of demand is forced into this size, "
                "so shape / colour / guardrails / floors are effectively neutral."
            )

        driver = row["dominant_driver_key"]
        movement = row["movement"]
        size_type = row["size_type"]
        size_bucket = "core" if size_type == "core" else "fringe / extended"

        # Shape-led movement
        if driver == "shape":
            if movement == "up":
                return (
                    f"Shape step pushes this {size_bucket} size **up vs raw option curve** "
                    f"because Cat L3 / Cat L2 baselines concentrate more demand here and the "
                    f"option is strong enough to support it."
                )
            elif movement == "down":
                return (
                    f"Shape step pulls this {size_bucket} size **down vs raw option curve** "
                    f"because Cat L3 / Cat L2 baselines are weaker here and the option is "
                    f"not strong enough to keep that extra share."
                )
            else:
                return (
                    f"Shape step keeps this {size_bucket} size broadly in line with the raw "
                    f"option curve; category curves confirm the same story."
                )

        # Colour-led movement
        if driver == "colour":
            if movement == "up":
                return (
                    "Colour behaviour for this shade **over-indexes in this size** at category "
                    "level, so the engine nudges the share up vs the neutral shape."
                )
            elif movement == "down":
                return (
                    "Colour for this shade **under-indexes in this size**, so the engine trims "
                    "the share slightly vs the neutral shape."
                )
            else:
                return (
                    "Colour behaviour is broadly neutral here, so it doesn’t materially move "
                    "this size vs the neutral shape."
                )

        # Core guardrail-led movement
        if driver == "guardrail":
            if movement == "up":
                if size_type == "core":
                    return (
                        "Core-share guardrail **lifts this core size** so overall core share "
                        "lands inside the 70–85% band without distorting intra-core shape."
                    )
                else:
                    return (
                        "Core-share guardrail **pulls more volume into core sizes**; this "
                        "fringe size benefits slightly as part of the rebalancing."
                    )
            elif movement == "down":
                if size_type == "core":
                    return (
                        "Core-share guardrail **trims this core size** a bit to avoid "
                        "over-concentration in core and keep total core share inside the "
                        "70–85% band."
                    )
                else:
                    return (
                        "Core-share guardrail **takes some share away from this fringe size** "
                        "so that core sizes can be boosted back into the target band."
                    )
            else:
                return (
                    "Core-share guardrail does not materially move this size; core share was "
                    "already within the desired band."
                )

        # Extended-size floor-led movement
        if driver == "floor":
            if movement == "up":
                return (
                    "Extended-size floor **protects this size** by giving it a minimum share "
                    "vs the Cat L2 + colour baseline, so it cannot collapse to zero."
                )
            elif movement == "down":
                return (
                    "Extended-size floors raise some other extended sizes, so this size is "
                    "slightly trimmed when the curve is renormalised."
                )
            else:
                return (
                    "Extended-size floors do not materially move this size; its share was "
                    "already at or above the floor."
                )

        # Neutral / mixed case
        return (
            "Net effect across shape, colour, guardrails and floors is tiny here; this size "
            "stays broadly in line with the raw option curve."
        )

    # Build per-size diagnostics based on engine columns
    diag_cols = [
        "_size_str",
        "size_type",
        "opt_ratio",
        "shape_norm_no_color",
        "shape_with_color_norm",
        "final_before_core_guardrail",
        "final_after_core_guardrail",
        "final_before_tail_floors",
        "final_after_tail_floors",
        "final_ratio",
    ]
    df_diag = (
        df_opt_res[diag_cols]
        .set_index("_size_str")
        .reindex(sizes_ordered)
    )
    df_diag.index.name = "size"

    # Core deltas by step
    df_diag["delta_vs_option"] = df_diag["final_ratio"] - df_diag["opt_ratio"]
    df_diag["delta_shape"] = df_diag["shape_norm_no_color"] - df_diag["opt_ratio"]
    df_diag["delta_colour"] = (
        df_diag["shape_with_color_norm"] - df_diag["shape_norm_no_color"]
    )
    df_diag["delta_guardrail"] = (
        df_diag["final_after_core_guardrail"] - df_diag["shape_with_color_norm"]
    )
    df_diag["delta_floor"] = (
        df_diag["final_after_tail_floors"] - df_diag["final_before_tail_floors"]
    )

    # Dominant driver per size
    delta_cols = ["delta_shape", "delta_colour", "delta_guardrail", "delta_floor"]
    abs_deltas = df_diag[delta_cols].abs()
    dominant_keys = abs_deltas.idxmax(axis=1)  # which step moved the most
    dominant_magnitude = abs_deltas.max(axis=1)

    driver_label_map = {
        "delta_shape": "Shape (option vs category)",
        "delta_colour": "Colour trend",
        "delta_guardrail": "Core-share guardrail",
        "delta_floor": "Extended size floor",
    }
    driver_key_map = {
        "delta_shape": "shape",
        "delta_colour": "colour",
        "delta_guardrail": "guardrail",
        "delta_floor": "floor",
    }

    df_diag["dominant_driver_key"] = dominant_keys.map(driver_key_map)
    df_diag["dominant_driver"] = dominant_keys.map(driver_label_map)

    # Neutral overrides for very tiny movements
    df_diag.loc[dominant_magnitude < EPS, "dominant_driver_key"] = "neutral"
    df_diag.loc[dominant_magnitude < EPS, "dominant_driver"] = "Neutral / tiny movement"

    # One-size-only override
    if is_one_size_only_flag:
        df_diag["dominant_driver_key"] = "one_size"
        df_diag["dominant_driver"] = "One-size-only (forced curve)"

    # Movement vs raw option curve
    df_diag["movement"] = df_diag["delta_vs_option"].apply(_movement_label)

    # Δ vs option in p.p. for display & chips
    df_diag["delta_vs_option_pp"] = df_diag["delta_vs_option"] * 100.0

    # Reason tags & full explainers
    df_diag["reason_tag"] = df_diag.apply(
        lambda r: _build_reason_tag(r["dominant_driver_key"], r["movement"]),
        axis=1,
    )
    df_diag["step_explainer"] = df_diag.apply(_build_step_explainer, axis=1)

    # ----- 6.5.1 Business summary -----
    st.markdown("#### 6.5.1 Business summary (what really moved this curve)")

    if is_one_size_only_flag:
        st.info(
            "This option is **one-size-only**. The engine simply sends 100% of demand "
            "into that size; shape / colour / guardrails / floors are neutral by design."
        )
    else:
        n_sizes = len(df_diag.index)
        up_count = int((df_diag["movement"] == "up").sum())
        down_count = int((df_diag["movement"] == "down").sum())
        flat_count = n_sizes - up_count - down_count

        driver_counts = df_diag["dominant_driver_key"].value_counts()
        shape_n = int(driver_counts.get("shape", 0))
        colour_n = int(driver_counts.get("colour", 0))
        guardrail_n = int(driver_counts.get("guardrail", 0))
        floor_n = int(driver_counts.get("floor", 0))
        neutral_n = int(driver_counts.get("neutral", 0))

        st.markdown(
            f"- **Sizes vs raw option curve:** {up_count} sizes moved up, "
            f"{down_count} moved down, {flat_count} stayed broadly similar "
            f"( |Δ| < 0.2 percentage points )."
        )
        st.markdown(
            "- **Dominant drivers by count of sizes:** "
            f"{shape_n}× Shape, {colour_n}× Colour, "
            f"{guardrail_n}× Core guardrail, {floor_n}× Extended floors, "
            f"{neutral_n}× Neutral."
        )

        # Identify main driver (ignoring neutral)
        main_driver_key = None
        for key in ["shape", "colour", "guardrail", "floor"]:
            if driver_counts.get(key, 0) > 0:
                if (
                    main_driver_key is None
                    or driver_counts.get(key, 0)
                    > driver_counts.get(main_driver_key, 0)
                ):
                    main_driver_key = key

        if main_driver_key == "shape":
            st.markdown(
                "Overall, the final curve is **shape-led**: we lean into Cat L3 / Cat L2 "
                "size behaviour where the option is not strong enough to fully dictate "
                "its own size curve."
            )
        elif main_driver_key == "colour":
            st.markdown(
                "Overall, the final curve is **colour-led**: movements are driven by how "
                "this colour over- or under-indexes by size at category level."
            )
        elif main_driver_key == "guardrail":
            st.markdown(
                "Overall, **core-share guardrails** are doing most of the work: the engine "
                "rebases core vs fringe to keep core share inside the 70–85% band while "
                "preserving shapes inside each bucket."
            )
        elif main_driver_key == "floor":
            st.markdown(
                "Overall, **extended-size floors** play an important role: extended sizes "
                "get a minimum share vs Cat L2+colour, and the rest of the curve adjusts "
                "around that."
            )

    # ----- 6.5.2 Step explainer chips (hover tooltips) -----
    st.markdown("#### 6.5.2 Step explainer chips (hover for details)")

    step_explainer_css = """
    <style>
    .step-explainer-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .step-explainer-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        gap: 0.35rem;
        cursor: default;
    }
    .step-explainer-pill .size-badge {
        font-weight: 600;
        font-family: monospace;
        padding: 0.1rem 0.35rem;
        border-radius: 0.6rem;
        background: #f3f4f6;
    }
    .step-explainer-pill .reason-tag {
        font-weight: 500;
    }
    .step-explainer-pill .delta-label {
        font-size: 0.75rem;
        color: #4b5563;
    }
    .driver-shape {
        background: #e0edff;
        border-color: #bfdbfe;
    }
    .driver-colour {
        background: #fff4e5;
        border-color: #fed7aa;
    }
    .driver-guardrail {
        background: #ede9fe;
        border-color: #c4b5fd;
    }
    .driver-floor {
        background: #e6f4ea;
        border-color: #bbf7d0;
    }
    .driver-neutral {
        background: #f3f4f6;
        border-color: #e5e7eb;
    }
    .driver-one_size {
        background: #fef3c7;
        border-color: #facc15;
    }
    </style>
    """
    st.markdown(step_explainer_css, unsafe_allow_html=True)

    chips = [
        '<div class="step-card"><h4>💡 Per-size step explainer (hover on chips)</h4>',
        '<div class="step-explainer-row">',
    ]

    for size, row in df_diag.iterrows():
        driver_key = row["dominant_driver_key"] or "neutral"
        if driver_key not in {"shape", "colour", "guardrail", "floor", "neutral", "one_size"}:
            driver_key = "neutral"

        reason_tag = row["reason_tag"]
        explainer = str(row["step_explainer"]).replace("\n", " ")
        delta_pp = float(row["delta_vs_option_pp"])
        movement = row["movement"]

        if movement == "up":
            arrow = "⬆️"
        elif movement == "down":
            arrow = "⬇️"
        else:
            arrow = "⟷"

        subtitle = f"{arrow} {delta_pp:+.1f} pp vs option"

        chips.append(
            f'<div class="step-explainer-pill driver-{driver_key}" title="{explainer}">'
            f'<span class="size-badge">{size}</span>'
            f'<span class="reason-tag">{reason_tag}</span>'
            f'<span class="delta-label">{subtitle}</span>'
            "</div>"
        )

    chips.append("</div></div>")
    chips_html = "\n".join(chips)
    st.markdown(chips_html, unsafe_allow_html=True)

    # ----- 6.5.3 Detailed table with dominant driver, colours & reasons -----
    st.markdown("#### 6.5.3 Detailed per-size table with dominant driver & reasons")

    df_display = df_diag.copy()
    df_display["option_curve_pct"] = df_display["opt_ratio"].apply(_fmt_pct)
    df_display["final_ratio_pct"] = df_display["final_ratio"].apply(_fmt_pct)
    df_display["delta_vs_option_pp_label"] = df_display["delta_vs_option_pp"].map(
        lambda v: f"{v:+.1f} pp"
    )

    df_display = df_display[
        [
            "size_type",
            "option_curve_pct",
            "final_ratio_pct",
            "delta_vs_option_pp_label",
            "dominant_driver",
            "reason_tag",
            "step_explainer",
        ]
    ]
    df_display.columns = [
        "size_type",
        "Option curve",
        "Final curve (engine)",
        "Δ vs option (pp)",
        "Dominant driver",
        "Reason tag",
        "Business-friendly explainer",
    ]

    def _style_driver_cell(val: str) -> str:
        if isinstance(val, str):
            if val.startswith("Shape"):
                return "background-color: #e0edff;"
            if val.startswith("Colour"):
                return "background-color: #fff4e5;"
            if val.startswith("Core-share"):
                return "background-color: #ede9fe;"
            if val.startswith("Extended"):
                return "background-color: #e6f4ea;"
            if val.startswith("One-size"):
                return "background-color: #fef3c7;"
            if val.startswith("Neutral"):
                return "background-color: #f3f4f6;"
        return ""

    styled = df_display.style.applymap(_style_driver_cell, subset=["Dominant driver"])
    styled = styled.set_properties(**{"font-size": "0.85rem"})

    st.dataframe(styled, use_container_width=True)




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
