"""
build_instances.py
-------------------
Genera instancias AMPL (.dat) para el solver exacto a partir de data/raw/.

Salida:
  data/instances/cdmx/        — copias directas de cam_*.dat
  data/instances/clustering/  — a1..unbalance convertidas a formato AMPL
  data/instances/phub/        — 1000 instancias wsc_N convertidas a formato AMPL

Formato de salida (igual que cam_*.dat):
  param N_total := <N> ;
  param P       := <n_candidatos> ;
  param R       := <radio> ;
  param c1      := 1.0 ;
  param c2      := 0.2 ;
  param nombre_instancia := "..." ;
  param : coordx coordy flag prob_ohca:=
  1  x  y  1  0.0000   <- preinstalado
  2  x  y  0  0.7312   <- candidato (prob aleatoria)
  ...
  ;
"""

import os
import shutil
import numpy as np
from pathlib import Path

# ── rutas ──────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/instances")

# ── parámetros fijos del modelo ────────────────────────────────────────────────
C1 = 1.0
C2 = 0.2

# ── configuración p-Hub ────────────────────────────────────────────────────────
PHUB_N_INSTANCES  = 1000
PHUB_R_MIN        = 0.05
PHUB_R_MAX        = 0.20
PHUB_NPRE_MIN     = 5
PHUB_NPRE_MAX     = 20

# ── configuración clustering ───────────────────────────────────────────────────
CLUSTERING_NPRE_MIN = 5
CLUSTERING_NPRE_MAX = 50         # rango aleatorio de preinstalados por instancia
CLUSTERING_SEED     = 42         # seed base (desplazado por índice de archivo)

# R calculado para ~50% cobertura con 50 preinstalados (seed=42)
CLUSTERING_R = {
    "s1.txt":        28000,
    "s2.txt":        34000,
    "s3.txt":        35000,
    "s4.txt":        32000,
    "a1.txt":         1800,
    "a2.txt":         2400,
    "a3.txt":         3050,
    "dim2.txt":       5150,
    "unbalance.txt":  1450,
}

# ── helpers de carga p-Hub ─────────────────────────────────────────────────────

def _parse_coord_csv(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    x = np.array([float(v) for v in lines[0].strip().split(";")[1:]])
    y = np.array([float(v) for v in lines[1].strip().split(";")[1:]])
    return np.column_stack([x, y])


def _load_phub(instance_id):
    base = RAW_DIR / "phub" / f"wsc_{instance_id}" / "Input"
    branches = _parse_coord_csv(base / f"coordinates_branches_{instance_id}.csv")
    hubs     = _parse_coord_csv(base / f"coordinates_hubs_{instance_id}.csv")
    return np.vstack([branches, hubs])

# ── escritor de .dat ───────────────────────────────────────────────────────────

def _write_dat(out_path, points, pre_idx, R, nombre, rng_prob):
    N = len(points)
    mask = np.zeros(N, dtype=bool)
    mask[pre_idx] = True
    n_cand = int((~mask).sum())

    lines = [
        "/* CONJUNTOS */",
        f"param N_total:= {N} ;",
        "",
        "/* PARAMETROS */",
        f"param P:= {n_cand} ;",
        f"param R:= {R} ;",
        f"param c1:= {C1} ;",
        f"param c2:= {C2} ;",
        f'param nombre_instancia := "{nombre}" ;',
        "",
        "param : coordx coordy flag prob_ohca:=",
    ]

    # Beta(0.4, 2.0): mayoría de probs cerca de 0, con picos altos ocasionales.
    # Crea "hotspots" que diferencian el paisaje de optimización entre instancias.
    n_cand_pts = int((~mask).sum())
    cand_probs = rng_prob.beta(0.4, 2.0, size=n_cand_pts)
    cand_iter  = iter(cand_probs)

    for i, (pt, is_pre) in enumerate(zip(points, mask), 1):
        flag = 1 if is_pre else 0
        prob = 0.0 if is_pre else float(next(cand_iter))
        lines.append(f"{i} {pt[0]:.6f} {pt[1]:.6f} {flag} {prob:.4f}")

    lines.append(";")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ── generadores ────────────────────────────────────────────────────────────────

def copiar_cdmx():
    out = OUT_DIR / "cdmx"
    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in (RAW_DIR / "cdmx").glob("cam_*.dat"):
        shutil.copy2(src, out / src.name)
        copied += 1
    print(f"[OK] CDMX: {copied} archivos copiados  -->  {out}/")


def generar_clustering():
    out = OUT_DIR / "clustering"
    out.mkdir(parents=True, exist_ok=True)

    for k, (fn, R) in enumerate(CLUSTERING_R.items()):
        pts  = np.loadtxt(RAW_DIR / "clustering" / fn)
        N    = len(pts)

        rng   = np.random.default_rng(CLUSTERING_SEED + k)
        n_pre = int(rng.integers(CLUSTERING_NPRE_MIN, CLUSTERING_NPRE_MAX + 1))
        n_pre = min(n_pre, N)
        pre_idx = rng.choice(N, size=n_pre, replace=False)

        stem   = fn.replace(".txt", "")
        nombre = f"{stem}_R{R}_pre{n_pre}"
        _write_dat(out / f"{nombre}.dat", pts, pre_idx, R, nombre, rng)
        print(f"  [OK] {fn:15s}  N={N:6d}  R={R:7}  n_pre={n_pre}")

    print(f"[OK] Clustering: {len(CLUSTERING_R)} instancias  -->{out}/")


def generar_phub():
    out = OUT_DIR / "phub"
    out.mkdir(parents=True, exist_ok=True)

    for iid in range(PHUB_N_INSTANCES):
        # un solo rng por instancia garantiza reproducibilidad total
        rng    = np.random.default_rng(iid)
        R      = float(rng.uniform(PHUB_R_MIN, PHUB_R_MAX))
        n_pre  = int(rng.integers(PHUB_NPRE_MIN, PHUB_NPRE_MAX + 1))
        pts    = _load_phub(iid)
        pre_idx = rng.choice(len(pts), size=n_pre, replace=False)
        # probs de candidatos con el mismo rng (determinista por instance_id)

        R_str  = f"{R:.4f}".replace(".", "p")
        nombre = f"wsc_{iid}_R{R_str}_pre{n_pre}"
        _write_dat(out / f"{nombre}.dat", pts, pre_idx, R, nombre, rng)

        if iid % 100 == 0:
            print(f"  phub: {iid}/{PHUB_N_INSTANCES}  (R={R:.4f}, n_pre={n_pre})")

    print(f"[OK] p-Hub: {PHUB_N_INSTANCES} instancias  -->{out}/")


# ── ejecución ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generando instancias en data/instances/ ===\n")
    copiar_cdmx()
    print()
    generar_clustering()
    print()
    generar_phub()
    print("\n=== Listo ===")
