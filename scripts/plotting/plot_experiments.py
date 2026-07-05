"""
plot_experiments.py
-------------------
Grafica instancias desde data/instances/, leyendo R y preinstalados
directamente del archivo .dat (sin semilla aleatoria).

Uso rápido (valores por defecto: wsc_0 y wsc_3 para pHub, s1 y a1 para clustering):
    python plot_experiments.py

Elegir instancias:
    python plot_experiments.py --phub wsc_0 wsc_3
    python plot_experiments.py --clustering s1 a1
    python plot_experiments.py --cdmx tlalpan coyoacan
    python plot_experiments.py --phub wsc_5 --clustering dim2 unbalance

Solo un tipo:
    python plot_experiments.py --no-clustering
    python plot_experiments.py --no-phub
    python plot_experiments.py --cdmx tlalpan   (--no-phub --no-clustering implícito)

Ver todas las instancias disponibles:
    python plot_experiments.py --list
"""

import os
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from scipy.spatial import cKDTree

REPO_ROOT       = Path(__file__).resolve().parents[2]
INSTANCES_DIR   = REPO_ROOT / "data" / "instances"
PHUB_DIR        = INSTANCES_DIR / "phub"
CLUST_DIR       = INSTANCES_DIR / "clustering"
CDMX_DIR        = INSTANCES_DIR / "cdmx"
OUT_DIR         = REPO_ROOT / "data" / "figures"

_C_CAND = "#4472C4"
_C_PRE  = "#E63946"

# ── leer archivo .dat ─────────────────────────────────────────────────────────

def parse_dat(path):
    """
    Devuelve (points, pre_idx, R, nombre_instancia).
    - points  : (N, 2) array con coordenadas
    - pre_idx : índices donde flag == 1 (preinstalados)
    - R       : radio de cobertura leído del archivo
    - nombre  : valor de param nombre_instancia
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    R = float(re.search(r"param R:=\s*([\d.eE+\-]+)", content).group(1))

    nombre_m = re.search(r'param nombre_instancia\s*:=\s*"([^"]+)"', content)
    nombre = nombre_m.group(1) if nombre_m else os.path.splitext(os.path.basename(path))[0]

    block = content.split("param : coordx coordy flag prob_ohca:=")[1]
    block = block.strip().rstrip(";")

    rows = []
    for line in block.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        rows.append((float(parts[1]), float(parts[2]), int(parts[3])))

    arr     = np.array(rows)
    points  = arr[:, :2]
    pre_idx = np.where(arr[:, 2] == 1)[0]
    return points, pre_idx, R, nombre

# ── catálogo de instancias ────────────────────────────────────────────────────

def _build_catalog(folder):
    """Devuelve {nombre_corto_lowercase: path} para phub/clustering (.dat con _R en el nombre)."""
    catalog = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".dat"):
            continue
        short = fn.split("_R")[0].lower()   # e.g. "wsc_0", "s1", "a1"
        catalog[short] = os.path.join(folder, fn)
    return catalog

def _build_catalog_cdmx(folder):
    """Devuelve {alcaldia_lowercase: path} para CDMX (cam_N_ALCALDIA.dat)."""
    catalog = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".dat"):
            continue
        # cam_15743_ALVARO_OBREGON.dat -> alvaro_obregon
        parts = fn.split("_", 2)
        short = parts[2].replace(".dat", "").lower() if len(parts) >= 3 else fn
        catalog[short] = os.path.join(folder, fn)
    return catalog

def list_all():
    print(f"=== pHub  ({PHUB_DIR}) ===")
    for k in sorted(_build_catalog(PHUB_DIR)):
        print(f"  {k}")
    print(f"\n=== Clustering  ({CLUST_DIR}) ===")
    for k in sorted(_build_catalog(CLUST_DIR)):
        print(f"  {k}")
    print(f"\n=== CDMX  ({CDMX_DIR}) ===")
    for k in sorted(_build_catalog_cdmx(CDMX_DIR)):
        print(f"  {k}")

# ── graficar ──────────────────────────────────────────────────────────────────

def _proximity_edges(points, R):
    return list(cKDTree(points).query_pairs(R))

def _coverage(points, pre_idx, R):
    n = len(points)
    if len(pre_idx) == 0:
        return 0, n, 0.0
    dists, _ = cKDTree(points[pre_idx]).query(points, k=1)
    n_covered = int((dists <= R).sum())
    return n_covered, n - n_covered, n_covered / n * 100

def _draw_subplot(ax, points, R, pre_idx, label=""):
    """Dibuja el subplot y devuelve el texto de stats para el .txt."""
    pairs   = _proximity_edges(points, R)
    n_edges = len(pairs)
    n_covered, n_uncovered, pct = _coverage(points, pre_idx, R)
    n_total = len(points)

    if pairs:
        segs = [[points[i], points[j]] for i, j in pairs]
        ax.add_collection(mc.LineCollection(
            segs, colors="gray", linewidths=0.35, alpha=0.45, zorder=1))

    mask_pre = np.zeros(n_total, dtype=bool)
    mask_pre[pre_idx] = True

    ax.scatter(points[~mask_pre, 0], points[~mask_pre, 1],
               s=6, color=_C_CAND, edgecolors="none",
               alpha=0.75, zorder=2)

    if mask_pre.any():
        ax.scatter(points[mask_pre, 0], points[mask_pre, 1],
                   s=90, color=_C_PRE, marker="*",
                   edgecolors="darkred", linewidths=0.4,
                   zorder=3)

    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")

    prefix = f"{label}  ·  " if label else ""
    return (
        f"{prefix}R = {R:.4g}   |   {n_edges} aristas\n"
        f"  Candidatos: {n_total - len(pre_idx)}   "
        f"Preinstalados: {len(pre_idx)}\n"
        f"  Cubiertos: {n_covered}/{n_total} ({pct:.1f}%)   "
        f"Sin cubrir: {n_uncovered}"
    )

def _build_and_save(panels, save_path):
    n = len(panels)
    _, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    if n == 1:
        axes = [axes]

    txt_lines = []
    for ax, p in zip(axes, panels):
        stats = _draw_subplot(ax, p["points"], p["R"], p["pre_idx"], label=p["label"])
        txt_lines.append(stats)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {save_path}")
    plt.close()

    txt_path = os.path.splitext(save_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(txt_lines) + "\n")
    print(f"Guardado: {txt_path}")

# ── funciones públicas ────────────────────────────────────────────────────────

def plot_phub(instance_names, save_path=None):
    """
    Grafica instancias pHub desde data/instances/phub/.

    Parameters
    ----------
    instance_names : list[str]   e.g. ["wsc_0", "wsc_3"]
    save_path      : str | None  auto-generado si es None
    """
    catalog = _build_catalog(PHUB_DIR)
    panels  = []
    for name in instance_names:
        key = name.lower()
        if key not in catalog:
            available = ", ".join(sorted(catalog))
            raise ValueError(f"Instancia '{name}' no encontrada.\n"
                             f"Disponibles: {available}")
        pts, pre_idx, R, _ = parse_dat(catalog[key])
        panels.append({"points": pts, "R": R, "pre_idx": pre_idx, "label": name})

    if save_path is None:
        tag = "_".join(n.replace("_", "") for n in instance_names)
        save_path = str(OUT_DIR / f"phub_{tag}.png")

    _build_and_save(panels, save_path)


def plot_clustering(instance_names, save_path=None):
    """
    Grafica instancias de clustering desde data/instances/clustering/.

    Parameters
    ----------
    instance_names : list[str]   e.g. ["s1", "a1"]
    save_path      : str | None  auto-generado si es None
    """
    catalog = _build_catalog(CLUST_DIR)
    panels  = []
    for name in instance_names:
        key = name.lower()
        if key not in catalog:
            available = ", ".join(sorted(catalog))
            raise ValueError(f"Instancia '{name}' no encontrada.\n"
                             f"Disponibles: {available}")
        pts, pre_idx, R, _ = parse_dat(catalog[key])
        panels.append({"points": pts, "R": R, "pre_idx": pre_idx, "label": name.upper()})

    if save_path is None:
        tag = "_".join(instance_names)
        save_path = str(OUT_DIR / f"clustering_{tag}.png")

    _build_and_save(panels, save_path)

def plot_cdmx(instance_names, save_path=None):
    """
    Grafica instancias CDMX desde data/instances/cdmx/.

    Parameters
    ----------
    instance_names : list[str]   e.g. ["tlalpan", "coyoacan"]
    save_path      : str | None  auto-generado si es None
    """
    catalog = _build_catalog_cdmx(CDMX_DIR)
    panels  = []
    for name in instance_names:
        key = name.lower()
        if key not in catalog:
            available = ", ".join(sorted(catalog))
            raise ValueError(f"Instancia '{name}' no encontrada.\n"
                             f"Disponibles: {available}")
        pts, pre_idx, R, _ = parse_dat(catalog[key])
        label = name.replace("_", " ").title()
        panels.append({"points": pts, "R": R, "pre_idx": pre_idx, "label": label})

    if save_path is None:
        tag = "_".join(instance_names)
        save_path = str(OUT_DIR / f"cdmx_{tag}.png")

    _build_and_save(panels, save_path)

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grafica instancias reales desde data/instances/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument(
        "--phub", nargs="+", default=["wsc_0", "wsc_3"], metavar="INST",
        help="Instancias pHub (default: wsc_0 wsc_3)")
    parser.add_argument(
        "--clustering", nargs="+", default=["s1", "a1"], metavar="INST",
        help="Instancias clustering (default: s1 a1)")
    parser.add_argument(
        "--cdmx", nargs="+", default=None, metavar="ALCALDIA",
        help="Instancias CDMX a graficar, e.g. tlalpan coyoacan")
    parser.add_argument(
        "--no-phub", action="store_true",
        help="Omite el plot de pHub")
    parser.add_argument(
        "--no-clustering", action="store_true",
        help="Omite el plot de clustering")
    parser.add_argument(
        "--list", action="store_true",
        help="Lista todas las instancias disponibles y sale")

    args = parser.parse_args()

    if args.list:
        list_all()
    else:
        if not args.no_phub:
            plot_phub(args.phub)
        if not args.no_clustering:
            plot_clustering(args.clustering)
        if args.cdmx:
            plot_cdmx(args.cdmx)
