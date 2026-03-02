import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# =========================
# Inputs: RAW split images ONLY (no dots)
# =========================
RAW_TOKEN = "inputs/evo_token_scaling.png"
RAW_PARAM = "inputs/evo_param_scaling.png"

# =========================
# Outputs
# =========================
OUT_OVERLAY = "outputs/evo_scaling_overlay.png"
OUT_DATA_CSV = "outputs/evo_scaling_data.csv"
OUT_LAWS_CSV = "outputs/evo_scaling_laws.csv"
OUT_RECREATED = "outputs/evo_scaling_reproduction.png"


# =========================
# Assumed axis ranges
# =========================
FLOPS_MIN, FLOPS_MAX = 4e18, 1e20
TOKENS_MIN, TOKENS_MAX = 1e10, 1e11
PARAMS_MIN, PARAMS_MAX = 8.5e6, 1e9


# =========================
# Axis bounds in IMAGE SPACE (centerline of left/bottom spines)
# =========================
@dataclass(frozen=True)
class AxisBounds:
    xmin_x: float
    xmin_y: float
    xmax_x: float
    xmax_y: float
    ymax_x: float
    ymax_y: float

# Left panel (tokens)
TOKEN_BOUNDS = AxisBounds(
    xmin_x=222.0, xmin_y=797.0,   # (x_min, y_min)  bottom-left spine center
    xmax_x=858.0, xmax_y=797.0,   # (x_max, y_min)  bottom-right along x-axis
    ymax_x=222.0, ymax_y=98.0     # (x_min, y_max)  top-left along y-axis
)

# Right panel (params)
PARAM_BOUNDS = AxisBounds(
    xmin_x=174.0, xmin_y=780.0,
    xmax_x=825.0, xmax_y=780.0,
    ymax_x=174.0, ymax_y=33.0
)


# =========================
# Point centroids in IMAGE SPACE (black-dot centers)
# budget_idx is left->right FLOPS cluster (0..3)
# Missing markers are None.
# =========================
POINTS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "tokens": {
        "Transformer++": [(359.0, 588.0), (540.0, 360.0), (677.0, 311.0), (813.0, 187.0)],
        "Mamba":         [(359.0, 570.0), (540.0, 449.0), (677.0, 284.0), (813.0, 252.0)],
        "Hyena":         [(359.0, 602.0), (540.0, 492.0), (677.0, 401.0), (813.0, 246.0)],
        "StripedHyena":  [(359.0, 638.0), (540.0, 502.0), (677.0, 361.0), (813.0, 271.0)],
    },
    "params": {
        "Transformer++": [(314.0, 671.0), (499.0, 654.0), (639.0, 541.0), (780.0, 485.0)],
        "Mamba":         [(314.0, 535.0), (499.0, 483.0), (639.0, 462.0), (780.0, 387.0)],
        "Hyena":         [(314.0, 455.0), (499.0, 380.0), (639.0, 307.0), (780.0, 280.0)],
        "StripedHyena":  [(314.0, 466.0), (499.0, 389.0), (639.0, 351.0), (780.0, 287.0)],
    },
}


# =========================
# Drawing helpers
# =========================
MODEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Transformer++": (255, 100, 50),
    "Mamba":         (80, 80, 220),
    "Hyena":         (220, 50, 150),
    "StripedHyena":  (100, 180, 50),
}

BOUNDS_COLOR = (255, 0, 0)


def draw_bounds(d: ImageDraw.ImageDraw, b: AxisBounds, color: Tuple[int, int, int] = BOUNDS_COLOR) -> None:
    """Draw axis boundary markers: filled circles at the 3 corner points and lines along the spines."""
    r = 8
    for x, y in [(b.xmin_x, b.xmin_y), (b.xmax_x, b.xmax_y), (b.ymax_x, b.ymax_y)]:
        d.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=color)
    d.line([(b.xmin_x, b.xmin_y), (b.xmax_x, b.xmax_y)], fill=color, width=3)
    d.line([(b.xmin_x, b.xmin_y), (b.ymax_x, b.ymax_y)], fill=color, width=3)


def draw_point(d: ImageDraw.ImageDraw, x: float, y: float, color: Tuple[int, int, int], r: int = 8) -> None:
    """Draw a crosshair + circle marker with a red center dot at the given point."""
    d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
    d.line([(x - r, y), (x + r, y)], fill=color, width=2)
    d.line([(x, y - r), (x, y + r)], fill=color, width=2)
    cr = 2
    d.ellipse((x - cr, y - cr, x + cr, y + cr), fill=(255, 0, 0), outline=(255, 0, 0))


# =========================
# Log mapping helpers
# =========================
def map_log_axis(px: float, px_min: float, px_max: float, v_min: float, v_max: float) -> float:
    """px linear -> log10(value) linear"""
    t = (px - px_min) / (px_max - px_min)
    lv = math.log10(v_min) + t * (math.log10(v_max) - math.log10(v_min))
    return 10 ** lv

def map_point_to_values(
    x_px: float,
    y_px: float,
    xmin_px: float,
    xmax_px: float,
    ymin_px: float,
    ymax_px: float,
    x_min_val: float,
    x_max_val: float,
    y_min_val: float,
    y_max_val: float
) -> Tuple[float, float]:
    # x is straightforward (increases right)
    x_val = map_log_axis(x_px, xmin_px, xmax_px, x_min_val, x_max_val)

    # y increases upward in data coords, but downward in pixel coords:
    y_from_bottom = (ymin_px - y_px)
    y_span = (ymin_px - ymax_px)
    y_val = map_log_axis(y_from_bottom, 0.0, y_span, y_min_val, y_max_val)
    return x_val, y_val


# =========================
# Main: overlay bounds + points, write CSV
# =========================
def draw_overlay_on_image(im: Image.Image, bounds: AxisBounds, plot_name: str) -> None:
    """Draw axis bounds and all model points onto an image."""
    d = ImageDraw.Draw(im)
    draw_bounds(d, bounds)
    for model, pts in POINTS[plot_name].items():
        color = MODEL_COLORS[model]
        for p in pts:
            draw_point(d, p[0], p[1], color)


def draw_combined_overlay(out_path: str) -> None:
    """Draw both panels side-by-side, aligned at the top by ymax_y."""
    token_im = Image.open(RAW_TOKEN).convert("RGB")
    param_im = Image.open(RAW_PARAM).convert("RGB")

    draw_overlay_on_image(token_im, TOKEN_BOUNDS, "tokens")
    draw_overlay_on_image(param_im, PARAM_BOUNDS, "params")

    # Shift param panel down to align plot areas visually
    token_off = 0
    param_off = 3

    total_w = token_im.width + param_im.width
    total_h = max(token_im.height + token_off, param_im.height + param_off)

    combined = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    combined.paste(token_im, (0, token_off))
    combined.paste(param_im, (token_im.width, param_off))
    combined.save(out_path, dpi=(300, 300))


def build_csv(rows: list, plot_name: str, bounds: AxisBounds, y_min: float, y_max: float) -> None:
    for model, pts in POINTS[plot_name].items():
        for budget_idx, p in enumerate(pts):
            x_px, y_px = p

            flops, y_val = map_point_to_values(
                x_px=x_px, y_px=y_px,
                xmin_px=bounds.xmin_x, xmax_px=bounds.xmax_x,
                ymin_px=bounds.xmin_y, ymax_px=bounds.ymax_y,
                x_min_val=FLOPS_MIN, x_max_val=FLOPS_MAX,
                y_min_val=y_min, y_max_val=y_max
            )

            rows.append({
                "variable": plot_name,
                "model": model,
                "budget": budget_idx,
                "flops": flops,
                "value": y_val,
            })


def fit_scaling_laws(df: pd.DataFrame) -> pd.DataFrame:
    """Fit power laws: params = a_0 * flops^a, tokens = b_0 * flops^b in log-log space."""
    coef_names = {"params": ("a", "a_0"), "tokens": ("b", "b_0")}
    law_rows = []
    for model in df["model"].unique():
        row: dict = {"model": model}
        for var in ("tokens", "params"):
            subset = df[(df["model"] == model) & (df["variable"] == var)]
            log_f = np.log10(subset["flops"].values)
            log_v = np.log10(subset["value"].values)
            slope, log_intercept = np.polyfit(log_f, log_v, 1)
            slope_name, intercept_name = coef_names[var]
            row[slope_name] = slope
            row[intercept_name] = 10 ** log_intercept
        law_rows.append(row)
    return pd.DataFrame(law_rows)


MODEL_MARKERS = {
    "Transformer++": ("o", "#E07050"),
    "Mamba":         ("D", "#7B9DC7"),
    "Hyena":         ("s", "#D88BC4"),
    "StripedHyena":  ("D", "#A8BF72"),
}

PANEL_CFG = {
    "tokens": {
        "ylabel": "Compute-optimal #tokens",
        "coefs": ("b", "b_0"),
        "dep_var": "D^{*}",
        "ymin": TOKENS_MIN,
        "ymax": TOKENS_MAX,
    },
    "params": {
        "ylabel": "Compute-optimal model size",
        "coefs": ("a", "a_0"),
        "dep_var": "N^{*}",
        "ymin": PARAMS_MIN,
        "ymax": PARAMS_MAX,
    },
}


def _fmt_coef(v: float) -> str:
    """Format a coefficient in scientific notation for LaTeX."""
    exp = math.floor(math.log10(abs(v)))
    mantissa = v / 10 ** exp
    return f"{mantissa:.3f}" + r"\!\times\!" + f"10^{{{exp}}}"


_MAX_MODEL_LEN = max(len(m) for m in MODEL_MARKERS)


def _law_label(model: str, law_row: pd.Series, slope_key: str, intercept_key: str, dep_var: str) -> str:
    slope = law_row[slope_key]
    intercept = law_row[intercept_key]
    padded = model.ljust(_MAX_MODEL_LEN + 2)
    return f"{padded}(${dep_var} = {_fmt_coef(intercept)} \\cdot C^{{{slope:.3f}}}$)"


def plot_chinchilla_recreation(data_df: pd.DataFrame, laws_df: pd.DataFrame, out_path: str) -> None:
    """Recreate the two-panel Chinchilla scaling figure from extracted data and fitted laws."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    flops_line = np.logspace(np.log10(FLOPS_MIN), np.log10(FLOPS_MAX), 200)

    for ax, var in zip(axes, ("tokens", "params")):
        cfg = PANEL_CFG[var]
        slope_key, intercept_key = cfg["coefs"]
        dep_var = cfg["dep_var"]

        for model in data_df["model"].unique():
            marker, color = MODEL_MARKERS[model]
            law_row = laws_df[laws_df["model"] == model].iloc[0]
            label = _law_label(model, law_row, slope_key, intercept_key, dep_var)

            sub = data_df[(data_df["model"] == model) & (data_df["variable"] == var)]
            ax.scatter(
                sub["flops"], sub["value"],
                marker=marker, color=color, edgecolors="k", alpha=0.8,
                linewidths=0.5, s=60, zorder=3, label=label,
            )

            fit_vals = law_row[intercept_key] * flops_line ** law_row[slope_key]
            ax.plot(flops_line, fit_vals, color=color, linewidth=1.5, alpha=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(cfg["ymin"], cfg["ymax"] * 1.5)
        ax.set_xlabel("FLOPS")
        ax.set_ylabel(cfg["ylabel"])
        ax.legend(fontsize=6, framealpha=0.9, loc="upper left", labelspacing=0.8,
                  borderpad=0.8, prop={"family": "monospace", "size": 6})
        ax.grid(True, which="major", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    draw_combined_overlay(OUT_OVERLAY)

    rows: list = []
    build_csv(rows, "tokens", TOKEN_BOUNDS, TOKENS_MIN, TOKENS_MAX)
    build_csv(rows, "params", PARAM_BOUNDS, PARAMS_MIN, PARAMS_MAX)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA_CSV, index=False)

    laws_df = fit_scaling_laws(df)
    laws_df.to_csv(OUT_LAWS_CSV, index=False)

    plot_chinchilla_recreation(df, laws_df, OUT_RECREATED)

    print("Wrote:")
    print(" ", OUT_OVERLAY)
    print(" ", OUT_DATA_CSV)
    print(" ", OUT_LAWS_CSV)
    print(" ", OUT_RECREATED)


if __name__ == "__main__":
    main()