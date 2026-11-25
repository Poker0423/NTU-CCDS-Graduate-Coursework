"""Analytical Beta PDF and curves drawn only from verified official records."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
try:
    from .artifacts import file_hash, write_json
    from .protocol import completed_bundle
except ImportError:
    from artifacts import file_hash, write_json
    from protocol import completed_bundle

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})


def plot_beta_pdf(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.001, 0.999, 2000)
    fig, ax = plt.subplots(figsize=(3.3, 2.4), layout="constrained")
    ax.plot(x, beta.pdf(x, 0.2, 0.2), color="#1d578c", linewidth=1.6)
    ax.set(xlabel=r"$\lambda$", ylabel="Probability density", xlim=(0, 1))
    ax.grid(alpha=0.2)
    fig.savefig(directory / "beta_pdf.png", dpi=300)
    plt.close(fig)
    write_json(directory / "beta_pdf.provenance.json", {"kind": "ANALYTICAL_FIGURE",
               "distribution": "Beta(0.2,0.2)", "interval": [0.001, 0.999],
               "endpoint_density": "Diverges to infinity at both endpoints; endpoints excluded",
               "sha256": file_hash(directory / "beta_pdf.png")})


def plot_curves(records, labels, title, path, mixup=False):
    """Two stacked panels readable at an AAAI single-column width."""
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.8), layout="constrained")
    colors = plt.cm.tab10.colors
    for i, (record, label) in enumerate(zip(records, labels)):
        h = record["history"]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        for ax, key in zip(axes, ("loss", "acc")):
            ax.plot(epochs, h[f"train_{key}"], color=colors[i], label=f"{label}: train")
            ax.plot(epochs, h[f"val_{key}"], "--", color=colors[i], label=f"{label}: val")
    for ax, ylabel in zip(axes, ("Cross-entropy", "Accuracy")):
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", framealpha=0.8)
    axes[1].set_ylim(0, 1)
    if mixup:
        axes[0].set_ylabel("CE (train objectives differ)")
        axes[1].set_ylabel("Accuracy (Mixup train weighted)")
    axes[0].set_title(title)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def generate_official_figures(context, output_root, figure_root):
    configs, records, baseline = completed_bundle(context, output_root)
    figure_root = Path(figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    baseline_index = [c.experiment_id for c in configs].index(baseline.experiment_id)
    groups = [(records[:3], [f"LR {c.lr:g}" for c in configs[:3]], "Learning rates", "sec3_lr.png", False),
              (records[3:5], ["Constant", "Cosine"], "Learning-rate schedule", "sec4_schedule.png", False),
              (records[5:7], ["WD 5e-4", "WD 1e-4"], "Weight decay", "sec5_wd.png", False),
              ([records[baseline_index], records[-1]], ["No Mixup", "Mixup"], "Data augmentation", "sec6_mixup.png", True)]
    for group, labels, title, name, mixed in groups:
        plot_curves(group, labels, title, figure_root / name, mixed)
    plot_beta_pdf(figure_root)
    write_json(figure_root / "experiment_figures.provenance.json", {
        "kind": "VERIFIED_EXPERIMENT_FIGURES",
        "records": {c.experiment_id: file_hash(Path(output_root) / c.experiment_id / "record.json") for c in configs},
        "figures": {group[3]: file_hash(figure_root / group[3]) for group in groups}})
