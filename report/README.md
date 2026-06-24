# FastSegmentator — TotalSegmentator parity validation

Validation of the optimized fast pipeline (`totalseg_infer.py`) vs. official
TotalSegmentator, run on the **same input** (DSC parity, not vs. ground truth),
n=10 cases/mode. Gate: mean DSC ≥ 0.995. Covers the full pipeline: Stage 1
(whole-body totals) → Stage 2 (crop pre-pass + postprocess) → Stage 3
(recursion / roi_subset / v1_order).

## Open these
- **`validation_report.html`** — main report: Stage 1/2/3 tables, DSC/speedup/scatter
  charts, 4 buckets (clean / caveats / license-blocked / no-data), and the confirmed
  root cause. Links to the diff viewer. (Charts use the Chart.js CDN → view online.)
- **`diff_examples.html`** — interactive visual gaps for every mode with DSC < 0.98:
  worst case, 5 largest-disagreement axial slices, raw | FastSeg (red) | official (green).
  Mode bar + slice prev/next. Fully self-contained (base64 images, works offline).
- **`figures/`** — static PNG charts (offline equivalents of the report charts).
- **`data/`** — raw per-mode parity CSVs (DSC, runtime, speedup, gate), incl. `stage1_parity.csv`.
- **`scripts/`** — generators to reproduce everything (`gen_report.py`, `gen_figures.py`, `gen_diff_viz.py`).

## Headline
- **Every validated mode now reaches ≥0.999 DSC parity** vs official, at 2–9× speedup.
- **Stage 1:** `total` (CT) **1.0000** ✅, `total_mr` (MR) **0.9999** ✅ (after the int32 fix).
- **Stage 2/3:** the previously sub-0.98 modes are all resolved —
  `pleural_pericard_effusion` 0.856→**0.999**, `lung_nodules` 0.967→**1.0**,
  `liver_lesions` 0.959→**1.0**, `liver_segments_mr` 0.972→**1.0**, `liver_lesions_mr`→**1.0**.
- **Lung vessels (VESSEL12, n=10):** `lung_vessels` (117) **0.99998**,
  `lung_vessels_LEGACY` (258) **0.99989**. (`stage2_lungvessels_vessel12.csv`)
- **Teeth (ToothFairy3, n=10):** `teeth` (113) **0.99992** — the last unvalidated mode, now
  confirmed. The recursive cascade `total → craniofacial_structures → teeth` runs end-to-end on
  dental CBCT, ≥0.9997 on every case at 3.94× speedup. (`stage2_teeth_toothfairy3.csv`)
- **Whole pipeline is GPU** (resample, crop, postprocess) and **deterministic** (cuDNN flags).

## Root causes & fixes (IMPLEMENTED)
Each sub-0.98 mode was isolated by bisecting against official's per-function intermediates
(`s01_0000` model input, `s01` prediction). Three independent causes, all fixed on GPU:
1. **Input resample order** — fast path used torch order-1 trilinear; official uses
   `skimage.resize(order=3)` cubic B-spline. Implemented a **GPU cubic B-spline**
   (Unser prefilter + 4-tap sampling + 12-voxel edge prepad + per-channel clip) matching
   `resample_data_or_seg` to ~1e-13. → `pleural`, `lung_nodules`.
2. **`dtype=np.int32`** missing on the cucim input resample (official truncates to int
   before the model). → `total_mr`, `liver_segments_mr`, liver crop bbox.
3. **Convert** threshold→softmax-argmax per low-confidence lesion mode. → `liver_lesions(_mr)`.

`liver_lesions_mr` is intrinsically nondeterministic (lesion on the crop boundary; official
flips fg 86/52 across runs) — our deterministic output matches official's same-draw at DSC 1.0.
Crop + connected-component postprocess were also GPU-ported (bit-identical to scipy).

## Datasets
AMOS22 (abdomen CT/MR), HECKTOR (head&neck CT), Luna25 (thoracic CT),
VESSEL12 (thoracic CT, lung vessels; n=10 of the 20 train scans, seed=42),
ToothFairy3 (dental CBCT, teeth; n=10 of the 532 cases, seed=42),
abdCT-187 PV (contrast abdomen CT, liver vessels/lesions).
