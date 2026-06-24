# nnU-Net Inference — Quick Guideline

Practical notes for running `FastSegmentator nnunet` on external abdominal-CT folders. Distilled from the ExternalTest-AbdomenCT/public runs (2026-06).

## Command

```bash
export nnUNet_extTrainer="$PWD/ext_trainers"          # needed for custom trainers (see below)
.venv/bin/python cli.py nnunet \
    -i <input_dir> -o <output_dir> \
    --model_path <RESULTS>/Dataset<NNN>_<Name>/nnUNetTrainerDA5__nnUNetResEncUNetMPlans__3d_fullres \
    --fold all --checkpoint checkpoint_final.pth --device cuda
```

- `--model_path` is the **trained-model dir under `nnUNet_results/`**, NOT `nnUNet_preprocessed/` (that only holds plans/fingerprints).
- Input filenames need no `_0000` suffix — the driver globs `*.nii.gz` and reads each directly. Output filename mirrors the input.

## Custom trainers (`nnUNetTrainerDA5`, etc.)

The vendored `src/` only ships the standard trainers. For a custom trainer, add a one-line stub in `ext_trainers/` and point `nnUNet_extTrainer` at it — do **not** edit `src/nnunetv2`:

```python
# ext_trainers/nnUNetTrainerDA5.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainerDA5(nnUNetTrainer): pass
```
DA5 differs only in (training-time) augmentation, so the network it builds is identical.

## Model ↔ contrast-phase mapping

Vessel models are phase-specific — match them to the input phase:

| Input phase | Folder marker / filename | Vessel model | Also run |
|---|---|---|---|
| Arterial (AV) | `*-AV-*`, `_AP` | `Dataset305_AVVessel` | — |
| Portal-venous (PV) | `*-PV-*`, `_PVP` | `Dataset304_PVVessel` | `Dataset302_LiverTumor` |

- Vessel labels: `1 = hepatic vessel`, `2 = portal vessel`. **PVVessel emits both (0,1,2); AVVessel emits only hepatic (0,1)** — portal vessels enhance only in portal-venous phase. An AV vessel output with no label-2 is expected, not a bug.
- Other organ models (`301_Liver`, `303_Spleen`) are phase-agnostic — run on any phase if needed.

## Batch hygiene

- **Resumable orchestrator:** `scripts/run_ext_public.sh` skips files that already have an output and stages only the missing ones via symlinks. Re-run anytime to fill gaps. Adapt the folder/model lists for new datasets.
- **Output convention:** `<input_folder>_pred/<ModelName>/` next to the inputs.
- **Per-file fault tolerance:** `nnunet_infer_nii.py` catches per-case exceptions, logs `FAILED <name>`, and continues — one bad file won't abort the batch.

## Empty / unprocessable triage

After a run, check label coverage and empty masks:

- **All sub-models empty on a case** ⇒ degenerate INPUT. Check per-Z nonzero counts — a healthy CT is nonzero on ~all slices (~0.1% zeros). (e.g. `SD177`: only 4 of 652 slices populated → `crop_to_nonzero` → empty.)
- **Single sub-model empty on a valid volume** ⇒ genuine model miss / absent structure (e.g. spleen out of FOV, no tumor). Acceptable.
- **Single-slice volume** (Z=1, e.g. `512×512×1`) ⇒ a 3D fullres model cannot resample it; it is logged and skipped. Out-of-distribution input — do not retry against a 3D model.

## Defaults that worked

`fold_all`, `checkpoint_final.pth`, `--device cuda`, no mirroring, argmax threshold (logit ≥ 0.5). Throughput ≈ 5 s/case for thick-slice CT, up to ~25 s/case for high-res (many-slice) volumes.
