import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Tuple


def compute_option_size_ratios_v5(
    df_boosted_cat: pd.DataFrame,
    mask_status_ok: Optional[pd.Series] = None,
    ros_col: str = "pred_ros_at_target_disc_wavg_boosted",
    option_col: str = "optioncode",
    stock_col: str = "stockcode",
    size_col: str = "size",
    cat_l3_col: str = "cat_l3",
    cat_l2_col: str = "cat_l2",
    colorgroup_col: str = "colorgroup",
    status_col: str = "Status",
    # optional pre-defined size family config:
    # { (cat_l2, cat_l3): {"core": [...], "fringe": [...]}, ... }
    size_config: Optional[Dict[Tuple[str, str], Dict[str, Iterable[str]]]] = None,
    # hyper-parameters
    core_share_min: float = 0.70,
    core_share_max: float = 0.85,
    auto_core_target_share: float = 0.80,
    # option weight ranges (core vs fringe)
    opt_core_min: float = 0.20,
    opt_core_max: float = 0.60,
    opt_fringe_min: float = 0.05,
    opt_fringe_max: float = 0.25,
    # extended size floor vs cat_l2+colour
    tail_floor_factor: float = 0.7,
    # strength transforms
    cat3_strength_power: float = 0.5,   # sqrt(cat3_share)
    option_strength_power: float = 1.0, # linear
    # colour bias caps
    core_bias_min: float = 0.7,
    core_bias_max: float = 1.3,
    fringe_bias_min: float = 0.8,
    fringe_bias_max: float = 1.2,
    # top-N colours by volume treated as “core” colours
    color_core_top_n: int = 5,
) -> pd.DataFrame:
    """
    Hierarchical size-ratio engine for menswear (India).

    Shape:
        option vs cat_l3 vs cat_l2 with dynamic weights depending on:
        - option ROS vs cat_l3 (option_strength),
        - cat_l3 ROS vs cat_l2 (cat3_strength),
        - core vs fringe sizes.

    Colour:
        global / cat_l2 / cat_l3 colour behaviour learned “one level up”
        and applied as multiplicative biases with caps.

    Extended sizes:
        2XL / 3XL / 4XL / 5XL get a floor vs cat_l2+colour curve.

    Returns one row per (option, size) with **full diagnostics**:
        - raw curves & totals
        - option & cat3 strengths
        - blending weights w_opt / w_cat3 / w_cat2
        - pre- and post-colour shapes
        - colour bias components & weights
        - core share before/after guardrails
        - tail floors and final ratios
    """

    df = df_boosted_cat.copy()

    # 0) Mask valid rows
    if mask_status_ok is None:
        if status_col in df.columns:
            mask_status_ok = df[status_col] != "Dropped"
        else:
            mask_status_ok = pd.Series(True, index=df.index)

    use_color = colorgroup_col is not None and colorgroup_col in df.columns

    # 1) Option x size aggregation (“child”)
    group_keys = [stock_col, option_col, size_col]
    child = (
        df.loc[mask_status_ok]
        .groupby(group_keys, dropna=False)[ros_col]
        .sum()
        .reset_index()
    )

    # Attach meta (cat_l2, cat_l3, colorgroup)
    meta_cols = [stock_col, option_col, size_col, cat_l2_col, cat_l3_col]
    if use_color:
        meta_cols.append(colorgroup_col)
    meta = df[meta_cols].drop_duplicates()
    child = child.merge(meta, on=[stock_col, option_col, size_col], how="left")

    # Option totals & raw option size curve (naive)
    child["opt_total_ros"] = child.groupby(option_col)[ros_col].transform("sum")
    child["opt_ratio"] = np.where(
        child["opt_total_ros"] > 0,
        child[ros_col] / child["opt_total_ros"],
        0.0,
    )

    # 2) Global, cat_l2, cat_l3 size curves
    # Global
    global_size = (
        child.groupby(size_col, dropna=False)[ros_col]
        .sum()
        .reset_index()
    )
    global_total_ros = float(global_size[ros_col].sum())
    global_size["global_ratio"] = np.where(
        global_total_ros > 0,
        global_size[ros_col] / global_total_ros,
        0.0,
    )
    global_size = global_size[[size_col, "global_ratio"]]

    # cat_l3
    cat3 = (
        child.groupby([cat_l3_col, size_col], dropna=False)[ros_col]
        .sum()
        .reset_index()
    )
    cat3["cat_l3_total_ros"] = cat3.groupby(cat_l3_col)[ros_col].transform("sum")
    cat3["cat_l3_ratio"] = np.where(
        cat3["cat_l3_total_ros"] > 0,
        cat3[ros_col] / cat3["cat_l3_total_ros"],
        0.0,
    )
    cat3 = cat3[[cat_l3_col, size_col, "cat_l3_ratio", "cat_l3_total_ros"]]

    # cat_l2
    cat2 = (
        child.groupby([cat_l2_col, size_col], dropna=False)[ros_col]
        .sum()
        .reset_index()
    )
    cat2["cat_l2_total_ros"] = cat2.groupby(cat_l2_col)[ros_col].transform("sum")
    cat2["cat_l2_ratio"] = np.where(
        cat2["cat_l2_total_ros"] > 0,
        cat2[ros_col] / cat2["cat_l2_total_ros"],
        0.0,
    )
    cat2 = cat2[[cat_l2_col, size_col, "cat_l2_ratio", "cat_l2_total_ros"]]

    # 3) Colour curves (global / cat2 / cat3)
    if use_color:
        # global colour
        global_color = (
            child.groupby([colorgroup_col, size_col], dropna=False)[ros_col]
            .sum()
            .reset_index()
        )
        global_color["global_color_total_ros"] = global_color.groupby(colorgroup_col)[
            ros_col
        ].transform("sum")
        global_color["global_color_ratio"] = np.where(
            global_color["global_color_total_ros"] > 0,
            global_color[ros_col] / global_color["global_color_total_ros"],
            0.0,
        )
        global_color = global_color[
            [colorgroup_col, size_col, "global_color_ratio", "global_color_total_ros"]
        ]
        # colour volumes for “core” colours
        color_vol = global_color.groupby(colorgroup_col)[
            "global_color_total_ros"
        ].first()
        core_colors = set(
            color_vol.sort_values(ascending=False).head(color_core_top_n).index
        )
        global_color_vol_map: Dict[str, float] = (
            global_color.groupby(colorgroup_col)["global_color_total_ros"]
            .first()
            .to_dict()
        )
    else:
        global_color = None
        core_colors = set()
        global_color_vol_map = {}

    if use_color:
        cat3c = (
            child.groupby([cat_l3_col, colorgroup_col, size_col], dropna=False)[ros_col]
            .sum()
            .reset_index()
        )
        cat3c["cat_l3_color_total_ros"] = cat3c.groupby(
            [cat_l3_col, colorgroup_col]
        )[ros_col].transform("sum")
        cat3c["cat_l3_color_ratio"] = np.where(
            cat3c["cat_l3_color_total_ros"] > 0,
            cat3c[ros_col] / cat3c["cat_l3_color_total_ros"],
            0.0,
        )
        cat3c = cat3c[
            [
                cat_l3_col,
                colorgroup_col,
                size_col,
                "cat_l3_color_ratio",
                "cat_l3_color_total_ros",
            ]
        ]

        cat2c = (
            child.groupby([cat_l2_col, colorgroup_col, size_col], dropna=False)[ros_col]
            .sum()
            .reset_index()
        )
        cat2c["cat_l2_color_total_ros"] = cat2c.groupby(
            [cat_l2_col, colorgroup_col]
        )[ros_col].transform("sum")
        cat2c["cat_l2_color_ratio"] = np.where(
            cat2c["cat_l2_color_total_ros"] > 0,
            cat2c[ros_col] / cat2c["cat_l2_color_total_ros"],
            0.0,
        )
        cat2c = cat2c[
            [
                cat_l2_col,
                colorgroup_col,
                size_col,
                "cat_l2_color_ratio",
                "cat_l2_color_total_ros",
            ]
        ]
    else:
        cat3c = None
        cat2c = None

    # 4) Strength maps (cat_l3 within cat_l2; colour shares)
    cat3_vol_map = cat3.groupby(cat_l3_col)["cat_l3_total_ros"].first().to_dict()
    cat2_vol_map = cat2.groupby(cat_l2_col)["cat_l2_total_ros"].first().to_dict()

    # mapping cat_l3 -> cat_l2 (assume 1-1)
    cat3_to_cat2 = (
        df[[cat_l3_col, cat_l2_col]]
        .dropna()
        .drop_duplicates()
        .set_index(cat_l3_col)[cat_l2_col]
        .to_dict()
    )

    cat3_strength_map: Dict[str, float] = {}
    for cat3_val, total3 in cat3_vol_map.items():
        cat2_val = cat3_to_cat2.get(cat3_val)
        if cat2_val is None:
            cat3_strength_map[cat3_val] = 0.0
            continue
        total2 = cat2_vol_map.get(cat2_val, 0.0)
        if total2 <= 0:
            cat3_strength_map[cat3_val] = 0.0
            continue
        share = float(total3) / float(total2)
        share = max(0.0, min(1.0, share))
        cat3_strength_map[cat3_val] = share ** cat3_strength_power

    if use_color:
        global_total_col_ros = sum(global_color_vol_map.values()) or 1.0
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
    else:
        global_total_col_ros = 1.0
        cat2_color_vol_map = {}
        cat3_color_vol_map = {}

    # 5) Auto core vs fringe definitions (cat_l3-driven, fallback cat_l2)
    def _is_one_size_value(s: str) -> bool:
        if s is None:
            return False
        su = str(s).strip().upper()
        return su in {"ONE-SIZE", "ONESIZE", "FREE-SIZE", "FREESIZE", "FREE SIZE"}

    def _infer_core_sizes(group: pd.DataFrame, ratio_col: str) -> pd.Series:
        g = group[[size_col, ratio_col]].copy()
        g[size_col] = g[size_col].astype(str).str.strip()
        g = g[~g[size_col].apply(_is_one_size_value)]
        g = g.sort_values(by=ratio_col, ascending=False)
        if g.empty:
            return pd.Series([], dtype=object)
        ratios = g[ratio_col].to_numpy()
        sizes = g[size_col].to_list()
        core = []
        cum = 0.0
        for s, r in zip(sizes, ratios):
            core.append(s)
            cum += float(r)
            if cum >= auto_core_target_share:
                break
        if len(core) < 2 and len(sizes) >= 2:
            core = sizes[:2]
        if len(core) > 3:
            core = core[:3]
        return pd.Series(core, dtype=object)

    cat3_core_map: Dict[str, set] = {}
    for cat3_val, grp in cat3.groupby(cat_l3_col):
        if grp["cat_l3_total_ros"].sum() <= 0:
            continue
        core_sizes = _infer_core_sizes(grp, "cat_l3_ratio")
        cat3_core_map[cat3_val] = set(core_sizes.tolist())

    cat2_core_map: Dict[str, set] = {}
    for cat2_val, grp in cat2.groupby(cat_l2_col):
        if grp["cat_l2_total_ros"].sum() <= 0:
            continue
        core_sizes = _infer_core_sizes(grp, "cat_l2_ratio")
        cat2_core_map[cat2_val] = set(core_sizes.tolist())

    def _size_type(row) -> str:
        size = str(row[size_col]).strip()
        cat3_val = row[cat_l3_col]
        cat2_val = row[cat_l2_col]
        if _is_one_size_value(size):
            return "one_size"
        # explicit overrides
        if size_config is not None:
            key = (str(cat2_val), str(cat3_val))
            cfg = size_config.get(key)
            if cfg is not None:
                if size in set(map(str, cfg.get("core", []))):
                    return "core"
                if size in set(map(str, cfg.get("fringe", []))):
                    return "fringe"
        # inferred
        if cat3_val in cat3_core_map:
            return "core" if size in cat3_core_map[cat3_val] else "fringe"
        if cat2_val in cat2_core_map:
            return "core" if size in cat2_core_map[cat2_val] else "fringe"
        if size.upper() in {"M", "L", "XL"}:
            return "core"
        return "fringe"

    child["size_type"] = child.apply(_size_type, axis=1)

    # ONE-SIZE only options
    option_one_size_flag = (
        child.groupby(option_col)[size_col]
        .agg(lambda x: all(_is_one_size_value(s) for s in x))
        .rename("is_one_size_only")
    )
    child = child.merge(option_one_size_flag, on=option_col, how="left")

    # 6) Merge all curves into child
    curves = (
        child.merge(cat3, on=[cat_l3_col, size_col], how="left")
        .merge(cat2, on=[cat_l2_col, size_col], how="left")
        .merge(global_size, on=size_col, how="left")
    )

    if use_color and cat3c is not None:
        curves = curves.merge(
            cat3c, on=[cat_l3_col, colorgroup_col, size_col], how="left"
        )
    else:
        curves["cat_l3_color_ratio"] = np.nan
        curves["cat_l3_color_total_ros"] = np.nan

    if use_color and cat2c is not None:
        curves = curves.merge(
            cat2c, on=[cat_l2_col, colorgroup_col, size_col], how="left"
        )
    else:
        curves["cat_l2_color_ratio"] = np.nan
        curves["cat_l2_color_total_ros"] = np.nan

    if use_color and global_color is not None:
        curves = curves.merge(global_color, on=[colorgroup_col, size_col], how="left")
    else:
        curves["global_color_ratio"] = np.nan
        curves["global_color_total_ros"] = np.nan

    # 7) Option-level strength vs cat_l3
    opt_level = (
        curves.groupby([cat_l3_col, option_col])["opt_total_ros"]
        .first()
        .reset_index()
    )
    cat3_median_opt_ros = (
        opt_level.groupby(cat_l3_col)["opt_total_ros"]
        .median()
        .rename("median_opt_ros_in_cat3")
    )
    opt_level = opt_level.merge(cat3_median_opt_ros, on=cat_l3_col, how="left")
    opt_level["median_opt_ros_in_cat3"].replace(0, np.nan, inplace=True)

    def _compute_option_strength(row) -> float:
        total = row["opt_total_ros"]
        median = row["median_opt_ros_in_cat3"]
        if not np.isfinite(median) or median <= 0:
            base = 0.5 if total > 0 else 0.0
        else:
            base = total / (total + median)
        base = base ** option_strength_power
        return float(max(0.0, min(1.0, base)))

    opt_level["option_strength"] = opt_level.apply(
        _compute_option_strength, axis=1
    )

    # attach option_strength & cat3_strength to curves
    cat3_strength_series = pd.Series(cat3_strength_map, name="cat3_strength")
    curves = curves.merge(
        opt_level[
            [cat_l3_col, option_col, "median_opt_ros_in_cat3", "option_strength"]
        ],
        on=[cat_l3_col, option_col],
        how="left",
    ).merge(
        cat3_strength_series.to_frame(),
        left_on=cat_l3_col,
        right_index=True,
        how="left",
    )

    curves["option_strength"].fillna(0.0, inplace=True)
    curves["cat3_strength"].fillna(0.0, inplace=True)

    # extended sizes (for tail floors)
    def _is_extended(size: str) -> bool:
        s = str(size).strip().upper()
        return any(x in s for x in ["2XL", "3XL", "4XL", "5XL"])

    # 8) Per-option blending with detailed diagnostics
    def _blend_for_option(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()

        # ONE-SIZE-only option: trivial, but still fill diagnostics
        if bool(g["is_one_size_only"].iloc[0]):
            g["final_ratio"] = 0.0
            mask = g[size_col].apply(lambda s: _is_one_size_value(str(s)))
            if mask.any():
                g.loc[mask, "final_ratio"] = 1.0

            g["ratio_source"] = "one_size"
            g["option_strength_used"] = 0.0
            g["core_coverage"] = 1.0

            # neutral diagnostics
            n = len(g)
            g["w_opt_size"] = 0.0
            g["w_cat3_size"] = 0.0
            g["w_cat2_size"] = 0.0
            g["shape_raw"] = g["final_ratio"]
            g["shape_norm_no_color"] = g["final_ratio"]

            g["w_color_global"] = 1.0
            g["w_color_cat2"] = 0.0
            g["w_color_cat3"] = 0.0

            g["bias_global"] = 1.0
            g["bias_cat2"] = 1.0
            g["bias_cat3"] = 1.0
            g["bias_total_clamped"] = 1.0

            g["shape_with_color_raw"] = g["final_ratio"]
            g["shape_with_color_norm"] = g["final_ratio"]

            core_mask = g["size_type"] == "core"
            core_sum = float(g.loc[core_mask, "final_ratio"].sum()) if core_mask.any() else 0.0

            g["final_before_core_guardrail"] = g["final_ratio"]
            g["final_after_core_guardrail"] = g["final_ratio"]
            g["core_share_before_guardrail"] = core_sum
            g["core_share_after_guardrail"] = core_sum

            final = g["final_ratio"].to_numpy()
            g["final_before_tail_floors"] = final
            g["tail_floor"] = np.zeros_like(final)
            g["final_after_tail_floors"] = final
            g["cat2_color_baseline_for_tail"] = np.nan

            return g

        opt_strength = float(g["option_strength"].iloc[0])
        cat3_val = g[cat_l3_col].iloc[0]
        cat2_val = g[cat_l2_col].iloc[0]
        cat3_strength = float(g["cat3_strength"].iloc[0])

        color_val = (
            g[colorgroup_col].iloc[0]
            if use_color and colorgroup_col in g.columns
            else None
        )

        base_cat3 = g["cat_l3_ratio"].fillna(0.0)
        base_cat2 = g["cat_l2_ratio"].fillna(0.0)
        global_ratio = g["global_ratio"].fillna(0.0)
        opt_ratio = g["opt_ratio"].fillna(0.0)

        # core coverage for this option
        sizes_present_core = (
            g.loc[g["size_type"] == "core", size_col].astype(str).unique()
        )
        if cat3_val in cat3_core_map:
            expected_core = cat3_core_map[cat3_val]
        elif cat2_val in cat2_core_map:
            expected_core = cat2_core_map[cat2_val]
        else:
            expected_core = {"M", "L", "XL"}
        expected_core_count = len(expected_core) or 1
        core_coverage = (
            len(set(sizes_present_core) & set(expected_core)) / expected_core_count
        )

        # --- option vs cat_l3 vs cat_l2 weights (core vs fringe) ---
        opt_strength_used = opt_strength * core_coverage
        opt_strength_used = max(0.0, min(1.0, opt_strength_used))

        def _weights_for_size(size_type: str):
            if size_type == "core":
                w_opt = opt_core_min + (opt_core_max - opt_core_min) * opt_strength_used
                w_cat_total = 1.0 - w_opt
                # cat_l3 gets small premium vs cat_l2, bounded 0.50..0.60
                f_cat3 = 0.50 + 0.10 * cat3_strength
            else:  # fringe / one_size
                w_opt = opt_fringe_min + (opt_fringe_max - opt_fringe_min) * opt_strength_used
                w_cat_total = 1.0 - w_opt
                # fringe: cat_l3 slightly below/around cat_l2, 0.45..0.50
                f_cat3 = 0.45 + 0.05 * cat3_strength
            f_cat3 = max(0.0, min(1.0, f_cat3))
            w_cat3 = w_cat_total * f_cat3
            w_cat2 = w_cat_total * (1.0 - f_cat3)
            return w_opt, w_cat3, w_cat2

        w_opt_arr = []
        w_cat3_arr = []
        w_cat2_arr = []
        for st in g["size_type"]:
            if st == "one_size":
                wopt, w3, w2 = _weights_for_size("core")  # treat as core-ish
            else:
                wopt, w3, w2 = _weights_for_size(st)
            w_opt_arr.append(wopt)
            w_cat3_arr.append(w3)
            w_cat2_arr.append(w2)
        w_opt_arr = np.array(w_opt_arr)
        w_cat3_arr = np.array(w_cat3_arr)
        w_cat2_arr = np.array(w_cat2_arr)

        # Fallback if cat3 & cat2 curves missing
        if base_cat3.sum() <= 0 and base_cat2.sum() <= 0 and global_ratio.sum() > 0:
            base_cat3 = global_ratio.copy()
            base_cat2 = global_ratio.copy()

        # blended shape before colour
        shape_raw = (
            w_opt_arr * opt_ratio.to_numpy()
            + w_cat3_arr * base_cat3.to_numpy()
            + w_cat2_arr * base_cat2.to_numpy()
        )
        if shape_raw.sum() <= 0:
            shape = np.full(len(g), 1.0 / len(g))
        else:
            shape = shape_raw / shape_raw.sum()

        # --- Colour bias (global / cat_l2 / cat_l3) ---
        if use_color and color_val is not None and not pd.isna(color_val):
            T_cat3 = float(cat3_vol_map.get(cat3_val, 0.0))
            T_cat2 = float(cat2_vol_map.get(cat2_val, 0.0))
            T_cat3_color = float(
                cat3_color_vol_map.get((cat3_val, color_val), 0.0)
            )
            T_cat2_color = float(
                cat2_color_vol_map.get((cat2_val, color_val), 0.0)
            )
            T_global_color = float(global_color_vol_map.get(color_val, 0.0))

            share_global_col = (
                T_global_color / global_total_col_ros if global_total_col_ros > 0 else 0.0
            )
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

            global_color_ratio = g["global_color_ratio"].fillna(0.0)
            base_color_cat2 = g["cat_l2_color_ratio"].fillna(0.0)
            base_color_cat3 = g["cat_l3_color_ratio"].fillna(0.0)

            with np.errstate(divide="ignore", invalid="ignore"):
                bias_global = np.where(
                    global_ratio > 0,
                    global_color_ratio / global_ratio,
                    1.0,
                )
                bias_cat2 = np.where(
                    base_cat2 > 0,
                    base_color_cat2 / base_cat2,
                    1.0,
                )
                bias_cat3 = np.where(
                    base_cat3 > 0,
                    base_color_cat3 / base_cat3,
                    1.0,
                )

            bias_global = np.nan_to_num(
                bias_global, nan=1.0, posinf=1.0, neginf=1.0
            )
            bias_cat2 = np.nan_to_num(
                bias_cat2, nan=1.0, posinf=1.0, neginf=1.0
            )
            bias_cat3 = np.nan_to_num(
                bias_cat3, nan=1.0, posinf=1.0, neginf=1.0
            )

            bias_total = []
            for i, st in enumerate(g["size_type"]):
                if st == "core" or st == "one_size":
                    lo, hi = core_bias_min, core_bias_max
                else:
                    lo, hi = fringe_bias_min, fringe_bias_max
                b = (
                    w_col_global * bias_global[i]
                    + w_col_cat2 * bias_cat2[i]
                    + w_col_cat3 * bias_cat3[i]
                )
                b = max(lo, min(hi, float(b)))
                bias_total.append(b)
            bias_total = np.array(bias_total)

            coloured = shape * bias_total
            if coloured.sum() > 0:
                coloured = coloured / coloured.sum()
            else:
                coloured = shape.copy()

            ratio_source = "shape+color_hier"
        else:
            # neutral colour behaviour
            w_col_global = 1.0
            w_col_cat2 = 0.0
            w_col_cat3 = 0.0
            bias_global = np.ones(len(g))
            bias_cat2 = np.ones(len(g))
            bias_cat3 = np.ones(len(g))
            bias_total = np.ones(len(g))
            coloured = shape.copy()
            ratio_source = "shape_only"

        base_final = coloured

        # store blending & colour diagnostics
        g["core_coverage"] = core_coverage
        g["w_opt_size"] = w_opt_arr
        g["w_cat3_size"] = w_cat3_arr
        g["w_cat2_size"] = w_cat2_arr
        g["shape_raw"] = shape_raw
        g["shape_norm_no_color"] = shape

        g["w_color_global"] = w_col_global
        g["w_color_cat2"] = w_col_cat2
        g["w_color_cat3"] = w_col_cat3

        g["bias_global"] = bias_global
        g["bias_cat2"] = bias_cat2
        g["bias_cat3"] = bias_cat3
        g["bias_total_clamped"] = bias_total

        g["shape_with_color_raw"] = shape * bias_total
        g["shape_with_color_norm"] = base_final

        g["final_ratio"] = base_final

        # --- Core share guardrails ---
        core_mask = g["size_type"] == "core"
        core_sum = float(g.loc[core_mask, "final_ratio"].sum())
        fringe_sum = float(g.loc[~core_mask, "final_ratio"].sum())

        g["final_before_core_guardrail"] = g["final_ratio"].to_numpy()
        g["core_share_before_guardrail"] = core_sum

        if core_mask.any() and fringe_sum > 0:
            target_core = core_sum
            if core_sum < core_share_min:
                target_core = core_share_min
            elif core_sum > core_share_max:
                target_core = core_share_max

            if target_core != core_sum:
                if core_sum > 0:
                    g.loc[core_mask, "final_ratio"] *= (target_core / core_sum)
                if fringe_sum > 0:
                    g.loc[~core_mask, "final_ratio"] *= (
                        (1.0 - target_core) / fringe_sum
                    )

        core_sum_after = float(g.loc[core_mask, "final_ratio"].sum()) if core_mask.any() else 0.0
        g["core_share_after_guardrail"] = core_sum_after
        g["final_after_core_guardrail"] = g["final_ratio"].to_numpy()

        # --- Tail floors vs cat_l2+colour (extended sizes) ---
        if use_color and color_val is not None and not pd.isna(color_val):
            with np.errstate(divide="ignore", invalid="ignore"):
                tail_bias_cat2 = np.where(
                    base_cat2 > 0,
                    g["cat_l2_color_ratio"].fillna(0.0) / base_cat2,
                    1.0,
                )
            tail_bias_cat2 = np.nan_to_num(
                tail_bias_cat2, nan=1.0, posinf=1.0, neginf=1.0
            )
            cat2_color_baseline = base_cat2.to_numpy() * tail_bias_cat2
            if cat2_color_baseline.sum() > 0:
                cat2_color_baseline = (
                    cat2_color_baseline / cat2_color_baseline.sum()
                )
        else:
            cat2_color_baseline = base_cat2.to_numpy().copy()
            if cat2_color_baseline.sum() > 0:
                cat2_color_baseline = (
                    cat2_color_baseline / cat2_color_baseline.sum()
                )

        final = g["final_ratio"].to_numpy()
        g["final_before_tail_floors"] = final.copy()

        if cat2_color_baseline.sum() > 0 and tail_floor_factor > 0:
            floors = []
            for i, sz in enumerate(g[size_col]):
                if _is_extended(sz):
                    floors.append(tail_floor_factor * cat2_color_baseline[i])
                else:
                    floors.append(0.0)
            floors = np.array(floors)
            floor_sum = floors.sum()
            if floor_sum > 1e-9:
                if floor_sum > 0.9:
                    floors *= 0.9 / floor_sum
                final = np.maximum(final, floors)
                if final.sum() > 0:
                    final /= final.sum()
        else:
            floors = np.zeros_like(final)

        g["tail_floor"] = floors
        g["final_ratio"] = final
        g["final_after_tail_floors"] = final
        g["cat2_color_baseline_for_tail"] = cat2_color_baseline

        g["ratio_source"] = ratio_source
        g["option_strength_used"] = opt_strength_used
        return g

    out = curves.groupby(option_col, group_keys=False).apply(_blend_for_option)

    # 9) Final tidy output (keep diagnostics)
    cols = [
        option_col,
        size_col,
        cat_l3_col,
        cat_l2_col,
        "size_type",
        "is_one_size_only",
        "opt_total_ros",
        "opt_ratio",
        "cat_l3_ratio",
        "cat_l3_total_ros",
        "cat_l2_ratio",
        "cat_l2_total_ros",
        "global_ratio",
        "global_color_ratio",
        "global_color_total_ros",
        "cat_l3_color_ratio",
        "cat_l3_color_total_ros",
        "cat_l2_color_ratio",
        "cat_l2_color_total_ros",
        "median_opt_ros_in_cat3",
        "option_strength",
        "option_strength_used",
        "cat3_strength",
        "core_coverage",
        "w_opt_size",
        "w_cat3_size",
        "w_cat2_size",
        "shape_raw",
        "shape_norm_no_color",
        "w_color_global",
        "w_color_cat2",
        "w_color_cat3",
        "bias_global",
        "bias_cat2",
        "bias_cat3",
        "bias_total_clamped",
        "shape_with_color_raw",
        "shape_with_color_norm",
        "final_before_core_guardrail",
        "final_after_core_guardrail",
        "core_share_before_guardrail",
        "core_share_after_guardrail",
        "cat2_color_baseline_for_tail",
        "final_before_tail_floors",
        "tail_floor",
        "final_after_tail_floors",
        "final_ratio",
        "ratio_source",
    ]

    if use_color:
        # ensure colour group column is near front
        cols.insert(2, colorgroup_col)

    out = out[cols].reset_index(drop=True)
    return out
