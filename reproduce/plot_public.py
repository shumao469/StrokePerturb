"""Render benchmark summaries from the public replay JSON files."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#758892", "#B4704B", "#5C8D9C", "#24766C", "#73568C")


def bars(ax, names, values, colors, xlabel, precision):
    pos = np.arange(len(names))
    ax.barh(pos, values, color=colors, height=0.60)
    ax.set(yticks=pos, yticklabels=names, xlabel=xlabel, xlim=(0, max(values) * 1.28))
    ax.invert_yaxis()
    offset = max(values) * 0.025
    for y, v in zip(pos, values):
        ax.text(v + offset, y, f"{v:.{precision}f}", va="center", fontsize=9)


def dual_gene(fig, data):
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.03, 1.2]})
    labels = {"no_change": "No change", "additive": "Additive", "ridge": "Linear ridge",
              "stroke_component": "Rank-16 projection"}
    summary = data["summary"]
    keys = [key for key in labels if key in summary]
    bars(left, [labels[k] for k in keys], [summary[k]["mse"] for k in keys],
         [COLORS[i] for i in (0, 1, 3, 4)][:len(keys)], "Mean held-out pair MSE", 5)
    left.set_title(f"A   Error across {len(data['pairs'])} combinations", loc="left")
    ex = data["examples"][0]
    row_y = np.arange(len(ex["genes"]))
    names = ("Measured", "Additive", "Linear ridge")
    for j, (key, name, color) in enumerate(zip(("measured", "additive", "ridge"), names, ("#253A4C", COLORS[1], COLORS[3]))):
        right.barh(row_y + (j - 1) * 0.24, ex[key], height=0.21, label=name, color=color)
    right.set(yticks=row_y, yticklabels=ex["genes"], xlabel="Expression difference from control")
    right.axvline(0, color="#9AA9B2", linewidth=0.8)
    right.invert_yaxis()
    right.set_title(f"B   Held-out example · {ex['pair']}", loc="left")
    right.legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("K562 dual-gene perturbation benchmark", fontsize=16, fontweight="bold")


def drug(fig, data):
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.3]})
    analysis = data.get("analysis", data)
    macro = analysis["macro_MSE"]
    labels = {"no_change": "No change", "additive": "Additive", "shrinkage": "Shrinkage",
              "fitted_linear": "Fitted linear", "cw_adaptation": "C/W adaptation"}
    keys = [key for key in labels if key in macro]
    bars(left, [labels[k] for k in keys], [macro[k] for k in keys], COLORS[:len(keys)], "Mean held-out pair MSE", 4)
    left.set_title("A   Mean error by predictor", loc="left")
    candidate, reference = ("cw_adaptation", "fitted_linear") if "cw_adaptation" in macro else ("shrinkage", "additive")
    pairs = [(" + ".join(row["compound_ids"]), row["mse"][candidate] - row["mse"][reference])
             for row in analysis["pairs"]]
    names, differences = zip(*sorted(pairs, key=lambda item: item[1], reverse=True))
    right.barh(range(len(pairs)), differences, height=0.58, color=[COLORS[4] if x >= 0 else COLORS[3] for x in differences])
    right.set(yticks=range(len(pairs)), yticklabels=names, xlabel=f"{labels[candidate]} MSE − {labels[reference]} MSE")
    right.invert_yaxis()
    right.axvline(0, color="#9AA9B2", linewidth=0.8)
    right.set_title("B   Difference within each held-out pair", loc="left")
    fig.suptitle("A549 ComboSciPlex drug-response benchmark", fontsize=16, fontweight="bold")


def development(fig, data):
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 0.85]})
    keys = ("interpolate", "carry_0h", "carry_96h")
    names = ("Linear interpolation", "0 h carry", "96 h carry")
    colors = (COLORS[3], COLORS[0], COLORS[1])
    times = [row for row in data["conditions"] if not row["endpoint"]]
    x = np.arange(len(times))
    for j, (key, name, color) in enumerate(zip(keys, names, colors)):
        left.bar(x + (j - 1) * 0.25, [row["mae"][key] for row in times], width=0.22, color=color, label=name)
    left.set(xticks=x, xticklabels=[f"{row['hours']} h" for row in times], ylabel="Mean absolute error")
    left.legend(frameon=False, fontsize=8)
    left.set_title("A   Error by held-out time", loc="left")
    bars(right, names, [data["equal_time_mae"][k] for k in keys], colors, "Equal-time mean absolute error", 2)
    right.set_title("B   Mean across held-out times", loc="left")
    fig.suptitle("H9 differentiation time-course interpolation", fontsize=16, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    layouts = {"fig3": dual_gene, "fig4": drug, "fig5": development}
    parser.add_argument("figure", choices=layouts)
    parser.add_argument("--input", type=Path, required=True, help="JSON produced by the corresponding replay")
    parser.add_argument("--output", type=Path, required=True, help="Output .png, .pdf or .svg")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#B6C2C9"}):
        fig = plt.figure(figsize=(11.5, 4.35), layout="constrained")
        layouts[args.figure](fig, payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=220, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
