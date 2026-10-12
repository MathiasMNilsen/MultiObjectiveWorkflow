"""Figures of the closed-loop results, in the style of Figs. 6-9 of the paper.

    python -m closed_loop.workflow.plot_results results_dir [interval ...]

Writes one figure per interval (all intervals, or only those given) to <results_dir>/figures/interval_XXX.png,
and the cumulative change over all finished intervals to <results_dir>/figures/cumulative.png (and .pdf).
"""
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
STYLE = {  # color, line style and label of every series
    "wind": ("#a39e93", "-", "Wind power"),
    "target": ("#b5473a", "-", "Target"),
    "proxy": ("#d99a2b", "--", "Proxy"),
    "optimized": ("#2f5f8f", "-", "Optimized"),
}
RATES = [("WOPR", "Oil production rate"), ("WGPR", "Gas production rate"),
         ("WWPR", "Water production rate"), ("WWIR", "Water injection rate")]
TOTALS = [("co2", "CO$_2$", 1e3, "kt"), ("npv", "NPV", 1e9, "bn USD"), ("oil", "Oil", 1e6, "MSm$^3$"),
          ("gas", "Gas", 1e6, "MSm$^3$"), ("wp", "Water\nprod.", 1e6, "MSm$^3$"), ("wi", "Water\ninj.", 1e6, "MSm$^3$")]
COMPACT = FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if abs(v) >= 1e6 else f"{v / 1e3:g}k" if abs(v) >= 1e4 else f"{v:g}")
CUMULATIVE = [  # key, label and format of the total in the cumulative figure
    ("co2", "CO₂", lambda v: f"{v / 1e3:,.3f} kt"),
    ("npv", "NPV", lambda v: f"${v / 1e9:,.3f}B"),
    ("oil", "Oil", lambda v: f"{v / 1e6:,.3f} MSm³"),
    ("gas", "Gas", lambda v: f"{v / 1e6:,.3f} MSm³"),
    ("wp", "Water\nproduction", lambda v: f"{v / 1e6:,.3f} MSm³"),
    ("wi", "Water\ninjection", lambda v: f"{v / 1e6:,.3f} MSm³"),
]
MINUS = str.maketrans("-", "−")  # typographic minus sign
STATUSES = ("accepted", "rejected", "skipped", "optimizer failed", "simulation failed")
BAR_SHADE = LinearSegmentedColormap.from_list("bar_shade", ["#f2f6fa", "#2f5f8f"])

plt.rcParams.update({
    "font.size": 9, "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 1.8, "legend.frameon": False, "savefig.dpi": 200,
})


def main(results_dir, *intervals):
    results_dir = Path(results_dir)
    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    files = sorted(results_dir.glob("interval_*.pkl"))
    if intervals:
        files = [results_dir / f"interval_{int(i):03d}.pkl" for i in intervals]
    for file in files:
        with open(file, "rb") as f:
            plot_interval(pickle.load(f), int(file.stem.split("_")[1]), figures / f"{file.stem}.png")
    plot_cumulative(pd.read_csv(results_dir / "summary.csv", parse_dates=["start", "end"]), figures / "cumulative.png")
    print(f"Figures written to {figures}")


def plot_interval(record, number, path):
    """Power demand, field rates and relative changes from the target of one interval (Figs. 7-9)."""
    dates, target = record["dates"], record["target"]
    shown = [s for s in ("target", "proxy", "optimized") if s in record]
    days = np.arange(1, len(dates) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), layout="constrained")
    fig.suptitle(f"Interval {number}: {dates[0]:%d %b %Y} - {dates[-1]:%d %b %Y} ({record['status']})",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")

    panels = [("Power demand", "MW", {"wind": record["wind"], **{s: record[s]["power"] for s in shown}})]
    panels += [(title, "Sm$^3$/day", {s: _field(record[s]["response"], vector) for s in shown})
               for vector, title in RATES]
    for ax, (title, unit, series) in zip(axes.flat, panels):
        for name, values in series.items():
            color, style, label = STYLE[name]
            ax.plot(days, values, style, color=color, label=label, drawstyle="steps-mid")
        ax.set(title=title, ylabel=unit, xlabel="Day of interval", xticks=days, xlim=(days[0], days[-1]))
        ax.yaxis.set_major_formatter(COMPACT)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper right", ncol=4)

    # Relative change of the interval totals, predicted by the proxy and simulated with the full model
    ax = axes[2, 1]
    ax.set_title("Change from target")
    keys = [(key, label) for key, label, *_ in TOTALS if target[key] != 0]
    candidates = [s for s in ("proxy", "optimized") if s in record]
    for n, s in enumerate(candidates):
        change = [100 * (record[s][key] / target[key] - 1) for key, _ in keys]
        x = np.arange(len(keys)) + 0.36 * (n - (len(candidates) - 1) / 2)
        bars = ax.bar(x, change, width=0.34, color=STYLE[s][0])
        if s == "optimized":
            ax.bar_label(bars, fmt="%+.1f%%", padding=2, fontsize=7, color=MUTED)
    if candidates:
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set(ylabel="%", xticks=range(len(keys)), xticklabels=[label for _, label in keys])
        ax.margins(y=0.15)
    else:
        ax.set_axis_off()
        ax.text(0, 0.9, "No optimized strategy", transform=ax.transAxes, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def plot_cumulative(summary, path):
    """Relative change of the cumulative totals of the applied strategy from the target (Fig. 6).

    Sized for the text width of a paper (6.5 in). Each bar is shaded from light at zero to dark at its end,
    the target (T) and optimized (O) totals are written above the bars, and a text box in the lower right
    holds the period and the number of intervals by status. A PDF is saved next to the PNG.
    """
    totals = summary.sum(numeric_only=True)
    with plt.rc_context({"font.size": 8, "axes.spines.left": False, "axes.spines.bottom": False,
                         "ytick.left": False, "xtick.bottom": False, "grid.linewidth": 0.5}):
        fig, ax = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
        ax.set_title("Cumulative change from the target strategy", fontsize=10, pad=12)
        changes = []
        for x, (key, label, total) in enumerate(CUMULATIVE):
            t, a = totals[f"target_{key}"], totals[f"applied_{key}"]
            c = 100 * (a / t - 1) if t else 0.0
            changes.append(c)
            if c != 0:
                shade = np.linspace(0, 1, 256)[:, None] if c > 0 else np.linspace(1, 0, 256)[:, None]
                ax.imshow(shade, extent=(x - 0.27, x + 0.27, min(0, c), max(0, c)), origin="lower", aspect="auto",
                          cmap=BAR_SHADE, vmin=0, vmax=1, zorder=2)
            ax.annotate(f"{c:.1f}%".translate(MINUS), (x, c), xytext=(0, 3 if c > 0 else -3),
                        textcoords="offset points", ha="center", va="bottom" if c > 0 else "top",
                        fontweight="semibold")
            ax.annotate(f"T {total(t)}\nO {total(a)}", (x, max(c, 0)), xytext=(0, 15 if c > 0 else 4),
                        textcoords="offset points", ha="center", va="bottom", family="monospace", fontsize=6.5,
                        color=MUTED, linespacing=1.3)

        counts = summary["status"].value_counts()
        info = [f"{summary['start'].iloc[0]:%d %b %Y} – {summary['end'].iloc[-1]:%d %b %Y}", "",
                "T  target", "O  optimized", "", f"{len(summary):>3} intervals"]
        info += [f"{counts[status]:>3} {status}" for status in STATUSES if status in counts]
        ax.text(0.995, 0.02, "\n".join(info), transform=ax.transAxes, ha="right", va="bottom",
                multialignment="left", family="monospace", fontsize=7, color=MUTED, linespacing=1.35,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=3)

        low, high = min(0, *changes), max(0, *changes)
        span = max(high - low, 0.5)
        ax.axhline(0, color=TEXT, linewidth=0.8, zorder=2)
        ax.set(xlim=(-0.5, len(CUMULATIVE) - 0.5), ylim=(low - 0.1 * span, high + 0.2 * span),
               ylabel="Change from target [%]")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}".translate(MINUS))
        ax.set_xticks(range(len(CUMULATIVE)), [label for _, label, _ in CUMULATIVE])
        fig.savefig(path, dpi=300)
        fig.savefig(Path(path).with_suffix(".pdf"))
        plt.close(fig)


def _field(response, vector):
    return np.sum([v for key, v in response.items() if key.startswith(vector + ":")], axis=0)


if __name__ == "__main__":
    main(*sys.argv[1:])
