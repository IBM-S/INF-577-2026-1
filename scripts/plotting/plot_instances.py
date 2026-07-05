import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw" / "clustering"
OUT_DIR = REPO_ROOT / "data" / "figures"
INSTANCE_FILES = [
    "s1.txt", "s2.txt", "s3.txt", "s4.txt",
    "a1.txt", "a2.txt", "a3.txt",
    "unbalance.txt", "dim2.txt"
]

def load_instance(filepath):
    """Load an instance file with columns: x, y"""
    data = np.loadtxt(filepath)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def plot_all_instances():
    n = len(INSTANCE_FILES)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, filename in enumerate(INSTANCE_FILES):
        filepath = os.path.join(DATA_DIR, filename)
        x, y = load_instance(filepath)
        name = os.path.splitext(filename)[0]

        axes[i].scatter(x, y, s=5, alpha=0.6, color="steelblue", edgecolors="none")
        axes[i].set_title(name, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].tick_params(labelsize=8)
        axes[i].ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Instancias de clustering 2D", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "instances_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado como {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_all_instances()
