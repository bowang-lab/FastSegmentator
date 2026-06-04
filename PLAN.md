# TotalSegmentator mode rollout — plan

Source: `totalseg_modes.html` (mode-by-mode pipeline diff vs. `total`).

## Goal

Replace the hard-coded `total`-only `totalseg_ct_infer.py` and `total_mr`-only
`totalseg_mr_infer.py` with a single config-driven entry point
`totalseg_infer.py` that supports every CT and MR mode catalogued in the HTML
doc, while preserving the cucim + `FastPreprocessor` + logit-threshold
fast-path that has been validated for `total`.

## Design choices (decided)

- **Single file**, CT + MR unified, dispatched by `--task`.
- **Fast-path applies to every mode.** Native-spacing modes (`resample=None`)
  skip the T2 forward/inverse cucim resample and pass the image's actual
  zooms as `props['spacing']`.
- **Few large stages (3).** Easier to ship than mode-by-mode.
- **DSC parity check per stage.** Each stage ends with a check that
  `mean DSC ≥ 0.995` vs. the official TotalSegmentator multilabel output on a
  small held-out set, for every newly-unlocked mode AND the regression check on
  `total`.

## Out of scope (current rollout)

- DICOM I/O paths.
- `--statistics`, `--radiomics`, `--preview`.
- `--save_probabilities`.
- Commercial license checks (`show_license_info()`). Modes run if the weights
  are present locally; no credential plumbing.
- `--roi_subset` is deferred to stage 3.

---

## Stage 1 — Refactor + all uncropped modes

**Scope.** Mode-aware dispatcher without new pipeline behaviour. Anything that
runs end-to-end without a crop pre-pass falls out for free.

### Work items

1. New file `totalseg_infer.py`.
2. `TaskConfig` dataclass (frozen) + `TASK_CONFIGS` dict keyed by mode name.
   Fields: `task_ids, resample, trainer, plans, model_config, folds,
   class_map_key, class_map_parts_key, partname_map_key, modality, licensed,
   crop, crop_addon, crop_model, mode_postprocess, remove_outside,
   remove_outside_dilation_mm, stage`.
3. CLI: `--task` with choices drawn from `TASK_CONFIGS`. Modes whose
   `cfg.stage > 1` exit with a clear "not yet implemented" message.
4. Generalize `build_predictors`: per-task `trainer`, `plans`, `model_config`,
   `folds`. Dataset folder resolved by glob `Dataset{tid:03d}_*` so new
   modes work without hard-coding dataset names.
5. Custom trainer registry: stub
   `nnUNetTrainer_DASegOrd0`, `nnUNetTrainer_DASegOrd0_NoMirroring`,
   `nnUNetTrainer_2000epochs_NoMirroring`,
   `nnUNetTrainer_4000epochs_NoMirroring`. Patch
   `nnunet_infer_nii.recursive_find_python_class` so `SimplePredictor`'s
   checkpoint load resolves them.
6. Generalize `infer_file`:
   - `cfg.resample is None`: skip T2 forward/inverse, build `props['spacing']`
     from `img_can.header.get_zooms()` (reversed XYZ → ZYX).
   - `cfg.resample` is set: T2 forward (cucim order=3) + inverse (cucim
     order=0), `props['spacing'] = list(reversed(cfg.resample))`.
   - Multi-model loop reads `class_map_parts_key` + `partname_map_key` from
     cfg (so `total_mr` and `headneck_muscles` work as easily as `total`).
   - Single-model loop just uses `class_map[cfg.class_map_key]` directly.
7. Tile-step rule: 0.8 only when `task=="total"` AND `resample[0] < 3.0`,
   else 0.5 (mirrors official literal).
8. Inline `remove_auxiliary_labels` for `appendicular_bones` so that single
   licensed CT mode runs to parity without a postprocessing module yet.

### Modes unlocked

| Modality | Mode | Multi-model? | Licensed? |
|---|---|---|---|
| CT  | `total`                 | yes (5) | — |
| CT  | `breasts`               | — | — |
| CT  | `trunk_cavities`        | — | — |
| CT  | `vertebrae_body`        | — | yes |
| CT  | `appendicular_bones`    | — | yes |
| CT  | `tissue_types`          | — | yes |
| CT  | `tissue_4_types`        | — | yes |
| CT  | `face`                  | — | yes |
| CT  | `thigh_shoulder_muscles`| — | yes |
| MR  | `total_mr`              | yes (2) | — |
| MR  | `body_mr`               | — | — |
| MR  | `vertebrae_mr`          | — | — |
| MR  | `appendicular_bones_mr` | — | yes |
| MR  | `tissue_types_mr`       | — | yes |
| MR  | `thigh_shoulder_muscles_mr` | — | yes |

Deferred to stage 2 even though they're technically uncropped:

- `body` — needs `keep_largest_blob_multilabel` + small-blob removal + derived
  `body.nii.gz` / `skin.nii.gz` save.
- `face_mr` — needs `face_mr_auxiliary` removal (brain, liver).
- `brain_aneurysm` — needs `folds=None` ensemble; we'll thread it once we
  prove `folds=[0]` works across stage 1.

### Validation gate

- Regression: `total` on 5 AMOS22 cases must match the pre-refactor output
  bit-exactly (logic is unchanged for `total`).
- New mode parity: DSC ≥ 0.995 vs. official on 3 cases for each of
  `trunk_cavities`, `body_mr`, `total_mr`, `breasts`, `vertebrae_mr`.
- All other stage-1 modes: smoke-run (loads weights, produces a non-empty
  output of the right shape/affine/label set).

### Risk

- Custom trainer name must resolve to the right class before checkpoint load.
  Asserting `predictor.configuration_manager.spacing` matches `cfg.resample`
  after init catches a mismatched trainer/plans pairing.

---

## Stage 2 — Crop pre-pass + mode-specific postprocess

**Scope.** Add the rough-seg → crop → predict → undo-crop machinery and the
four postprocess branches. The big unlock.

### Work items

1. `crop.py` (new module or inline):
   - `build_crop_mask(image, crop_rois, mode_context) -> (crop_mask_nib,
     rough_seg_nib)`.
   - CT default rough seg: `total` @ 6 mm (T298). If
     `crop_rois ∩ {body_trunc, body_extremities}` then `body` @ 6 mm (T300).
     MR default: `total_mr` @ 3 mm (T852).
2. Port `cropping.crop_to_mask` and `cropping.undo_crop` from
   `.venv_official/.../totalsegmentator/cropping.py`.
3. Wire `crop` and `bbox` through `infer_file`. Crop happens **before**
   `as_closest_canonical` (matches `nnunet.py:439-475`).
4. New `postprocess.py`:
   - `body` branch: `keep_largest_blob_multilabel(["body_trunc"])` +
     `remove_small_blobs_multilabel(["body_extremities"], 50000 mm³)` +
     save-time `body.nii.gz` + `skin.nii.gz` derivation.
   - `remove_auxiliary_labels(task_name)` — generic (drops labels from
     `class_map["{task}_auxiliary"]` when present). Replaces the inline
     version from stage 1.
   - `remove_outside_of_mask` — only fires for `heartchambers_highres` (with
     a 10 mm dilation, computed in voxels from image zooms).
5. Thread `folds=None` end-to-end (full-fold ensemble for
   `pleural_pericard_effusion` and `brain_aneurysm`).

### Modes unlocked

All remaining CT modes except `teeth`:
`body`, `cerebral_bleed`, `hip_implant`, `liver_vessels`, `lung_vessels`,
`lung_vessels_LEGACY`, `lung_nodules`, `pleural_pericard_effusion`,
`kidney_cysts`, `liver_segments`, `liver_lesions`, `head_glands_cavities`,
`head_muscles`, `headneck_bones_vessels`, `headneck_muscles`,
`craniofacial_structures`, `oculomotor_muscles`, `ventricle_parts`,
`abdominal_muscles`, `brain_aneurysm`, `brain_structures`,
`heartchambers_highres`, `coronary_arteries`, `coronary_arteries_LEGACY`,
`aortic_sinuses`.

MR side: `liver_segments_mr`, `liver_lesions_mr`, `face_mr` (now with
auxiliary-label removal).

### Validation gate

Five focus modes, DSC ≥ 0.995 vs. official on 3 cases each:

1. `body` — postprocess parity (multilabel diff + presence of `body.nii.gz` /
   `skin.nii.gz`).
2. `cerebral_bleed` — native-spacing crop sanity check.
3. `head_glands_cavities` — `3d_fullres_high` + skull crop + 0.75 mm
   in-plane resample.
4. `heartchambers_highres` — `remove_outside_of_mask` correctness.
5. `headneck_muscles` — multi-model crop combo.

Regression: every stage-1 mode still passes.

### Risk

Bounding-box arithmetic across the canonical reorient is the historical
foot-gun (cerebrum entry [2026-05-10]). Reference order in
`nnunet.py:439-478` is: crop in original orientation → reorient inside the
crop → undo in reverse.

---

## Stage 3 — Recursion + ROI subset + final sweep

**Scope.** Close the remaining gap and harden.

### Work items

1. Recursive `crop_model`: when `cfg.crop_model` is set, the rough seg comes
   from a recursive `totalseg_infer(task=cfg.crop_model)` call (currently
   only `teeth` → `craniofacial_structures`). Build crop mask from the
   recursive output's `class_map[crop_model]`.
2. `roi_subset` filtering — allowed only when `task.startswith("total")`. For
   multi-model totals, prune the model list to those whose part-map
   intersects the subset (mirrors `nnunet.py:539-550`).
3. `v1_order` reorder for `total` (uses `class_map["total_v1"]`).
4. Full mode-catalogue sweep: 1 case per non-licensed mode against official;
   record DSC and timing into `report/mode_parity.csv`.
5. Delete the now-unused `totalseg_ct_infer.py` and `totalseg_mr_infer.py`
   (no external callers, confirmed by `grep`).

### Modes unlocked

`teeth` — and any mode that still failed the sweep.

### Validation gate

Per-mode DSC ≥ 0.995 vs. official on the canonical AMOS22 / abdCT-AV-PV case
set. Runtime per mode within 1.5× of stage 1's `total` runtime (custom
fast-path benefit should generalise).
