import argparse
import gc
import glob
import os
import sys
from time import time
from typing import Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from fastsegmentator._vendor import nnunetv2
import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from fastsegmentator._vendor.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from fastsegmentator._vendor.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from fastsegmentator._vendor.nnunetv2.inference.sliding_window_prediction import compute_gaussian
from fastsegmentator._vendor.nnunetv2.preprocessing.resampling.default_resampling import fast_resample_logit_to_shape
from fastsegmentator._vendor.nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from fastsegmentator._vendor.nnunetv2.utilities.helpers import empty_cache, dummy_context
from fastsegmentator._vendor.nnunetv2.utilities.label_handling.label_handling import LabelManager, determine_num_input_channels
from fastsegmentator._vendor.nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from fastsegmentator._vendor.nnunetv2.utilities.utils import log_runtime


def _canonical_transform(direction_tuple):
    """Compute permutation and flips to reorient a (Z,Y,X) array to canonical (S,A,R) order.

    direction_tuple: SimpleITK GetDirection() — 9-element flat LPS direction matrix.
    Returns (fwd_perm, fwd_flips, inv_perm) as lists of int numpy-axis indices.
    """
    D_lps = np.array(direction_tuple).reshape(3, 3)
    D_ras = D_lps.copy()
    D_ras[0] *= -1  # L → R
    D_ras[1] *= -1  # P → A
    # numpy axis k (0=z,1=y,2=x) ↔ SimpleITK voxel axis (2-k)
    dominant_ras = np.array([np.argmax(np.abs(D_ras[:, 2 - k])) for k in range(3)])
    signs        = np.array([np.sign(D_ras[dominant_ras[k], 2 - k])  for k in range(3)])
    # canonical numpy layout: axis 0=S (RAS 2), axis 1=A (RAS 1), axis 2=R (RAS 0)
    fwd_perm  = [int(np.where(dominant_ras == ras_ax)[0][0]) for ras_ax in [2, 1, 0]]
    fwd_flips = [i for i in range(3) if signs[fwd_perm[i]] < 0]
    inv_perm  = [0, 0, 0]
    for i, p in enumerate(fwd_perm):
        inv_perm[p] = i
    return fwd_perm, fwd_flips, inv_perm


def _apply_canonical(x: torch.Tensor, fwd_perm, fwd_flips) -> torch.Tensor:
    """Permute + flip a (C,Z,Y,X) or (Z,Y,X) tensor to canonical space (GPU-friendly)."""
    has_channel = x.ndim == 4
    perm  = ([0] + [p + 1 for p in fwd_perm]) if has_channel else list(fwd_perm)
    flips = ([f + 1 for f in fwd_flips])       if has_channel else list(fwd_flips)
    x = x.permute(perm)
    if flips:
        x = torch.flip(x, flips)
    return x.contiguous()


def _undo_canonical(x: torch.Tensor, fwd_flips, inv_perm) -> torch.Tensor:
    """Inverse of _apply_canonical on a (Z,Y,X) tensor."""
    if fwd_flips:
        x = torch.flip(x, fwd_flips)
    return x.permute(inv_perm).contiguous()


@log_runtime
def logits_to_segmentation(predicted_logits):
    max_logit, max_class = torch.max(predicted_logits, dim=0)
    segmentation = torch.where(
        max_logit >= 0.5, max_class, torch.tensor(0, device=predicted_logits.device)
    )
    return segmentation


def convert_predicted_logits_to_segmentation_with_correct_shape(
    predicted_logits: Union[torch.Tensor, np.ndarray],
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    label_manager: LabelManager,
    properties_dict: dict,
    use_softmax: bool,
    return_probabilities: bool = False,
):
    # Compute spacing for resampling
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling'])
        else [spacing_transposed[0], *configuration_manager.spacing]
    )
    target_shape = properties_dict['shape_after_cropping_and_before_resampling']
    original_spacing = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]

    if target_shape[0] < 600:
        # torch separate-z output resample (order=1, matches nnU-Net resampling_fn_probabilities;
        # separate-z auto-detected by anisotropy). The production output resampler.
        from fastsegmentator._vendor.nnunetv2.preprocessing.resampling.default_resampling import torch_resample_data_or_seg_to_shape
        predicted_logits = torch_resample_data_or_seg_to_shape(
            predicted_logits, list(target_shape), current_spacing, original_spacing,
            is_seg=False, order=1, order_z=0, force_separate_z=None)
        gc.collect()
        empty_cache(predicted_logits.device)
        if use_softmax:
            segmentation = label_manager.convert_probabilities_to_segmentation(
                label_manager.apply_inference_nonlin(predicted_logits))
        else:
            segmentation = logits_to_segmentation(predicted_logits)
    else:
        # Large volume: memory-efficient resample with argmax during resampling
        # (avoids materializing the full resampled logit tensor).
        segmentation = fast_resample_logit_to_shape(
            predicted_logits, target_shape, current_spacing, original_spacing
        )

    # Revert cropping
    dtype = torch.uint8 if len(label_manager.foreground_labels) < 255 else torch.uint16
    segmentation_reverted_cropping = torch.zeros(properties_dict['shape_before_cropping'], dtype=dtype)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # Revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.permute(plans_manager.transpose_backward)

    return segmentation_reverted_cropping.cpu()


class SimplePredictor(nnUNetPredictor):
    """Simple predictor for nnUNet that supports direct NIfTI inference."""

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str]], None],
        checkpoint_name: str,
    ):
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(
                join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                map_location=torch.device('cpu'),
                weights_only=False,
            )
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes')
            ckpt = checkpoint['network_weights']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            parameters.append(ckpt)

        configuration_manager = plans_manager.get_configuration(configuration_name)

        # Restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            'fastsegmentator._vendor.nnunetv2.training.nnUNetTrainer',
        )
        if trainer_class is None:
            raise RuntimeError(
                f'Unable to locate trainer class {trainer_name} in '
                f'fastsegmentator._vendor.nnunetv2.training.nnUNetTrainer or any directory listed in '
                f'nnUNet_extTrainer={os.environ.get("nnUNet_extTrainer", "")!r}'
            )

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        # Load each fold's parameters (last one wins for single-fold; enables multi-fold ensemble averaging)
        for params in self.list_of_parameters:
            self.network.load_state_dict(params)

        if ('nnUNet_compile' in os.environ) and (
            os.environ['nnUNet_compile'].lower() in ('true', '1', 't')
        ) and not isinstance(self.network, torch._dynamo.OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def preprocess(self, image, props):
        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
        image = torch.from_numpy(image).to(dtype=torch.float32, memory_format=torch.contiguous_format).to(self.device)
        if props.get('_can_fwd_perm') is not None:
            image = _apply_canonical(image, props['_can_fwd_perm'], props['_can_fwd_flips'])
        data = preprocessor.run_case_npy(
            image, None, props,
            self.plans_manager, self.configuration_manager, self.dataset_json,
        )
        return data

    @log_runtime
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers,
        do_on_device: bool = True,
    ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            gaussian = (
                compute_gaussian(
                    tuple(self.configuration_manager.patch_size),
                    sigma_scale=1. / 8,
                    value_scaling_factor=10,
                    device=results_device,
                )
                if self.use_gaussian
                else 1
            )

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None].to(self.device)
                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions

            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError(
                    'Encountered inf in predicted array. Aborting... If this problem persists, '
                    'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                    'predicted_logits to fp32'
                )
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e

        return predicted_logits

    def inference(self, image, properties_dict, use_softmax):
        image = self.preprocess(image, properties_dict)

        with torch.no_grad():
            assert isinstance(image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()
            empty_cache(self.device)

            use_autocast = self.device.type == 'cuda'
            with torch.autocast(self.device.type, enabled=True) if use_autocast else dummy_context():
                data, slicer_revert_padding = pad_nd_image(
                    image, self.configuration_manager.patch_size,
                    'constant', {'value': 0}, True, None,
                )
                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                predicted_logits = self._internal_predict_sliding_window_return_logits(
                    data, slicers, self.perform_everything_on_device,
                )
                empty_cache(self.device)
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

                segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
                    predicted_logits,
                    self.plans_manager,
                    self.configuration_manager,
                    self.label_manager,
                    properties_dict,
                    use_softmax,
                    return_probabilities=False,
                )

                if properties_dict.get('_can_fwd_flips') is not None:
                    segmentation = _undo_canonical(
                        segmentation,
                        properties_dict['_can_fwd_flips'],
                        properties_dict['_can_inv_perm'],
                    )

        return segmentation


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference for nnUNet model")
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input image folder')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output segmentation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model directory')
    parser.add_argument('--fold', type=str, default='all', help='Fold to use for inference (default: all)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', help='Checkpoint filename')
    parser.add_argument('--use_softmax', action='store_true', default=False, help='Apply softmax to output')
    parser.add_argument('--device', type=str, default='cuda', help='Device (e.g., "cuda" or "cpu")')
    return parser.parse_args()


def main():
    args = parse_arguments()

    perform_everything_on_device = args.device != 'cpu'
    device = torch.device(args.device, 0)

    # Validate inputs before loading the model (fail fast, no wasted GPU load).
    if not os.path.isdir(args.input_path):
        sys.exit(f"ERROR: input path is not a directory: {args.input_path}")
    files = sorted(glob.glob(os.path.join(args.input_path, '*.nii.gz')))
    if not files:
        sys.exit(f"ERROR: no .nii.gz files found in {args.input_path}")
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)

    predictor = SimplePredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=perform_everything_on_device,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    try:
        predictor.initialize_from_trained_model_folder(
            args.model_path,
            use_folds=args.fold,
            checkpoint_name=args.checkpoint,
        )
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"ERROR: could not load model from {args.model_path} "
                 f"(fold={args.fold}, checkpoint={args.checkpoint}): {e}")
    predictor.network.to(device)

    failures = []
    for file in tqdm(files):
        try:
            image, props = SimpleITKIO().read_images([file])
            t0 = time()
            seg = predictor.inference(image, props, args.use_softmax)
            print(f'total: {time() - t0:.2f}s')

            sitk_img = sitk.GetImageFromArray(seg)
            sitk_img.SetSpacing(props['sitk_stuff']['spacing'])
            sitk_img.SetOrigin(props['sitk_stuff']['origin'])
            sitk_img.SetDirection(props['sitk_stuff']['direction'])

            case_name = os.path.basename(file).replace('_0000.nii.gz', '.nii.gz')
            sitk.WriteImage(sitk_img, os.path.join(output_folder, case_name))
        except Exception as e:
            # Don't let one bad case (e.g. a single-slice volume that a 3D model
            # cannot resample) abort the whole batch — log it and move on.
            failures.append(os.path.basename(file))
            print(f'FAILED {os.path.basename(file)}: {type(e).__name__}: {e}')
            empty_cache(device)

    if failures:
        print(f'\n{len(failures)} case(s) FAILED and were skipped:')
        for f in failures:
            print(f'  - {f}')
        sys.exit(1)


if __name__ == "__main__":
    main()
