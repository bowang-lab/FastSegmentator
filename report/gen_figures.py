"""Render static PNG figures for the validation report into <outdir>/figures/.
Usage: .venv/bin/python report/gen_figures.py <outdir>
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import the data tables from the report generator (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_report import CLEAN, CAVEAT, STAGE1

rows = [dict(mode=r[0], dsc=r[2], spd=r[3]) for r in STAGE1 + CLEAN + CAVEAT]
rows.sort(key=lambda r: r["dsc"])  # ascending so best on top in barh
labels = [r["mode"] for r in rows]
dsc    = [r["dsc"] for r in rows]
spd    = [r["spd"] for r in rows]
col    = ["#2e7d32" if d >= 0.995 else ("#f9a825" if d >= 0.99 else "#e65100") for d in dsc]

outdir = Path(sys.argv[1]) / "figures"
outdir.mkdir(parents=True, exist_ok=True)

# 1. DSC by mode
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(labels, dsc, color=col)
ax.axvline(0.995, ls="--", c="#888", lw=1); ax.text(0.9955, 0.2, "0.995 gate", color="#666", fontsize=8)
ax.set_xlim(0.75, 1.0); ax.set_xlabel("mean DSC vs official (n=10)")
ax.set_title("Stage 2/3 parity — DSC by mode")
fig.tight_layout(); fig.savefig(outdir / "dsc_by_mode.png", dpi=130); plt.close(fig)

# 2. Speedup by mode (same order)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(labels, spd, color="#1565c0")
ax.set_xlabel("speedup vs official (×)"); ax.set_title("Fast-path speedup by mode")
for i, v in enumerate(spd): ax.text(v + 0.05, i, f"{v:.1f}×", va="center", fontsize=8, color="#333")
fig.tight_layout(); fig.savefig(outdir / "speedup_by_mode.png", dpi=130); plt.close(fig)

# 3. DSC vs speedup scatter
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(spd, dsc, c=col, s=70, edgecolor="#333", lw=0.5, zorder=3)
ax.axhline(0.995, ls="--", c="#888", lw=1)
for r in rows:
    ax.annotate(r["mode"], (r["spd"], r["dsc"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("speedup vs official (×)"); ax.set_ylabel("mean DSC"); ax.set_ylim(0.75, 1.005)
ax.set_title("DSC vs speedup")
fig.tight_layout(); fig.savefig(outdir / "dsc_vs_speedup.png", dpi=130); plt.close(fig)

print(f"wrote 3 figures to {outdir}")
