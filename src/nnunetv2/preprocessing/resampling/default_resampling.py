from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import sklearn
import torch
import gc
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
from skimage.transform import resize
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.utils import log_runtime


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape



def determine_do_sep_z_and_axis(
        force_separate_z: bool,
        current_spacing,
        new_spacing,
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD) -> Tuple[bool, Union[int, None]]:
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
            axis = None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_data_or_seg_to_spacing(data: np.ndarray,
                                    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    is_seg: bool = False,
                                    order: int = 3, order_z: int = 0,
                                    force_separate_z: Union[bool, None] = False,
                                    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    shape = np.array(data.shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped

@log_runtime
def resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    """
    needed for segmentation export. Stupid, I know
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg(data: np.ndarray, new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False, axis: Union[None, int] = None, order: int = 3,
                         do_separate_z: bool = False, order_z: int = 0, dtype_out = None):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if np.any(shape != new_shape):
        data = data.astype(float, copy=False)
        if do_separate_z:
            # print("separate z, order in z is", order_z, "order inplane is", order)
            assert axis is not None, 'If do_separate_z, we need to know what axis is anisotropic'
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape

                    # align_corners=False
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')[None]
                    else:
                        unique_labels = np.sort(pd.unique(reshaped_here.ravel()))  # np.unique(reshaped_data)
                        for i, cl in enumerate(unique_labels):
                            reshaped_final[c][np.round(
                                map_coordinates((reshaped_here == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest')) > 0.5] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            # print("no separate z, order", order)
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        return reshaped_final
    else:
        # print("no resampling necessary")
        return data

def fast_resize_segmentation(segmentation, new_shape, mode="nearest"):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype

    if isinstance(segmentation, torch.Tensor):
        assert len(segmentation.shape[2:]) == len(new_shape), f"segmentation.shape = {segmentation.shape}, new_shape = {new_shape}"
    else:
        assert len(segmentation.shape[1:]) == len(new_shape), f"segmentation.shape = {segmentation.shape}, new_shape = {new_shape}"
        segmentation = torch.from_numpy(segmentation).unsqueeze(0).float()
    #if order == 0:
        #return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    if mode == "nearest":
        seg_torch = torch.nn.functional.interpolate(segmentation, new_shape, mode=mode)
        reshaped = seg_torch
    else:
        #reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        unique_labels = torch.unique(segmentation)
        seg_torch = segmentation
        reshaped = torch.zeros([*seg_torch.shape[:2], *new_shape], dtype=seg_torch.dtype, device=seg_torch.device)
        for i, c in enumerate(unique_labels):
            #mask = segmentation == c
            #reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            mask = seg_torch == c
            reshaped_multihot = torch.nn.functional.interpolate(mask.float(), new_shape, mode=mode, align_corners=False)
            reshaped[reshaped_multihot >= 0.5] = c

    return reshaped

@log_runtime
def fast_resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")
    order_to_mode_map = {
        0: "nearest",
        1: "trilinear" if new_shape[0] > 1 else "bilinear",
        2: "trilinear" if new_shape[0] > 1 else "bilinear",
        3: "trilinear" if new_shape[0] > 1 else "bicubic",
        4: "trilinear" if new_shape[0] > 1 else "bicubic",
        5: "trilinear" if new_shape[0] > 1 else "bicubic",
    }
    
    if is_seg:
        resize_fn = fast_resize_segmentation
        kwargs = {
            "mode": order_to_mode_map[order]
        }
    else:
        resize_fn = torch.nn.functional.interpolate
        kwargs = {
            'mode': order_to_mode_map[order],
            'align_corners': False
        }
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        if not isinstance(data, torch.Tensor):
            torch_data = torch.as_tensor(data.get())
        else:
            torch_data = data.float()
        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
            new_shape = new_shape[1:]
        else:
            torch_data = torch_data.unsqueeze(0)
        
        torch_data = resize_fn(torch_data.to(device), tuple(new_shape), **kwargs)

        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
        else:
            torch_data = torch_data.squeeze(0)


        reshaped_final_data = torch_data

        assert reshaped_final_data.ndim == 4, f"reshaped_final_data.shape = {reshaped_final_data.shape}"
        return reshaped_final_data
    else:
        print("no resampling necessary")
        return data
    
def logit_to_segment(predicted_logits):
    max_logit, max_class = torch.max(predicted_logits, dim=0)
                
                # Apply threshold: Only assign the class if its logit exceeds the threshold
    segmentation = torch.where(max_logit >= 0.5, max_class, torch.tensor(0, device=predicted_logits.device))

    return segmentation

def resize_by_chunk(torch_data, new_shape, chunk_size = 300):
    torch_data = torch_data.detach().cpu()
    torch.cuda.empty_cache()
    step = new_shape[0] // chunk_size + 1
    seg_old_spacing = np.zeros(new_shape)
    z = torch_data.shape[2]
    stride = int(z / step)
    step1 = [i * stride for i in range(step)] + [z]
    z = new_shape[0]
    stride = int(z / step)
    step2 = [i * stride for i in range(step)] + [z]
    for i in range(step):
        size = list(new_shape)
        size[0] = step2[i + 1] - step2[i]
        slicer = torch_data[:,:, step1[i]:step1[i + 1]]#.half()
        slicer = torch.nn.functional.interpolate(slicer.cuda(), mode='trilinear', size=size, align_corners=True)[0]
        seg_old_spacing[step2[i]:step2[i + 1]] = logit_to_segment(slicer).cpu()
        del slicer
        torch.cuda.empty_cache()

    return torch.from_numpy(seg_old_spacing)

@log_runtime
def fast_resample_logit_to_shape(torch_data: Union[torch.Tensor, np.ndarray],
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")
    order_to_mode_map = {
        0: "nearest",
        1: "trilinear" if new_shape[0] > 1 else "bilinear",
        2: "trilinear" if new_shape[0] > 1 else "bilinear",
        3: "trilinear" if new_shape[0] > 1 else "bicubic",
        4: "trilinear" if new_shape[0] > 1 else "bicubic",
        5: "trilinear" if new_shape[0] > 1 else "bicubic",
    }
    

    resize_fn = torch.nn.functional.interpolate
    kwargs = {
        'mode': order_to_mode_map[order],
        'align_corners': False,
    }
    shape = np.array(torch_data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        
        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
            new_shape = new_shape[1:]
        else:
            torch_data = torch_data.unsqueeze(0)
        gc.collect()
        empty_cache(device)
        if new_shape[0] < 600:
            torch_data = resize_fn(torch_data.to(device), tuple(new_shape), **kwargs)

            if new_shape[0] == 1:
                torch_data = torch_data.transpose(1, 0)
            else:
                torch_data = torch_data.squeeze(0)
        else:
            torch_data = resize_by_chunk(torch_data.to(device), tuple(new_shape))

        

        reshaped_final_data = torch_data


        return reshaped_final_data
    else:
        print("no resampling necessary")
        # Must match the resample-path boundary (>= 600 → segmentation) and the
        # caller convert_...(target_shape[0] < 600). Using > 600 here left exactly
        # Z=600 returning 4D logits where the caller expects a 3D segmentation.
        if new_shape[0] >= 600:
            torch_data = logit_to_segment(torch_data)
        return torch_data


import math as _math


def _cubic_bspline_basis(u: torch.Tensor) -> torch.Tensor:
    """Cubic B-spline kernel B3 evaluated at signed distance `u`."""
    a = u.abs()
    w = torch.zeros_like(a)
    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)
    w[m1] = 2.0 / 3.0 - a[m1] ** 2 + 0.5 * a[m1] ** 3
    w[m2] = ((2.0 - a[m2]) ** 3) / 6.0
    return w


def _bspline3_prefilter_lastdim(c: torch.Tensor) -> torch.Tensor:
    """Cubic B-spline prefilter (Unser/ITK recursive IIR) along the LAST dim.
    c: (..., n) float. Mirror boundary. Returns spline coefficients, same shape.
    The python loops run along the (short) filtered axis but are vectorized over all
    leading dims, so each iteration is a single elementwise kernel on the (...,) plane.
    """
    z = _math.sqrt(3.0) - 2.0          # cubic pole
    n = c.shape[-1]
    if n == 1:
        return c
    c = c * 6.0                         # gain = (1-z)(1-1/z) = 6 for cubic
    # --- causal: truncated mirror initialization, then forward recursion ---
    tol = 1e-9
    horizon = min(n, int(_math.ceil(_math.log(tol) / _math.log(abs(z)))))
    zk = z
    c0 = c[..., 0].clone()
    for k in range(1, horizon):
        c0 = c0 + zk * c[..., k]
        zk *= z
    cols = [c0]
    prev = c0
    for k in range(1, n):
        prev = c[..., k] + z * prev
        cols.append(prev)
    c = torch.stack(cols, dim=-1)
    # --- anti-causal: mirror initialization, then backward recursion ---
    cols[-1] = (z / (z * z - 1.0)) * (c[..., n - 1] + z * c[..., n - 2])
    prev = cols[-1]
    for k in range(n - 2, -1, -1):
        prev = z * (prev - c[..., k])
        cols[k] = prev
    return torch.stack(cols, dim=-1)


_CUBIC_NPAD = 12   # scipy.ndimage._interpolation._prepad_for_spline_filter for mode='nearest'


def _cubic_resize_lastdim(coeffs: torch.Tensor, n_orig: int, out_size: int,
                          npad: int = _CUBIC_NPAD) -> torch.Tensor:
    """Sample cubic B-spline coefficients of a `npad`-edge-padded signal (so coeffs has
    length n_orig+2*npad) onto `out_size` points. Replicates scipy.ndimage.zoom with
    grid_mode=True: input coord = (o+0.5)*(n_orig/out_size) - 0.5, shifted by +npad into
    the padded array. The pad makes the boundary edge/nearest (matching skimage mode='edge')."""
    npadded = coeffs.shape[-1]
    device = coeffs.device
    scale = float(n_orig) / out_size
    j = torch.arange(out_size, device=device, dtype=torch.float64)
    x = (j + 0.5) * scale - 0.5 + npad                  # coords into the padded array
    fl = torch.floor(x)
    out = torch.zeros(*coeffs.shape[:-1], out_size, device=device, dtype=coeffs.dtype)
    for o in (-1, 0, 1, 2):
        m = fl + o
        w = _cubic_bspline_basis(x - m)                 # (out_size,)
        idx = m.long().clamp(0, npadded - 1)            # taps stay interior thanks to the pad
        out = out + coeffs.index_select(-1, idx) * w
    return out


def _gpu_cubic_resample(t: torch.Tensor, new_shape, axes) -> torch.Tensor:
    """Separable cubic B-spline resample of t (c, *spatial) along `axes` (spatial dims,
    0-based within spatial) to new_shape sizes. Faithfully reproduces scipy.ndimage.zoom
    (order=3, mode='nearest', grid_mode=True) — the path nnUNet's resample_data_or_seg
    takes via skimage.resize(order=3, mode='edge'): 12-voxel edge prepad, B-spline
    prefilter, then grid-mode cubic sampling. Runs in float64 (scipy uses float64; the IIR
    recursion amplifies error at sharp edges, so float32 diverges there)."""
    in_dtype = t.dtype
    t = t.to(torch.float64)
    # skimage.resize clips cubic-ringing overshoot back to the input's value range
    # (_clip_warp_output), per channel — ndi.zoom alone does not. Capture before resampling.
    flat = t.reshape(t.shape[0], -1)
    vmin = flat.amin(dim=1).view(-1, *([1] * (t.ndim - 1)))
    vmax = flat.amax(dim=1).view(-1, *([1] * (t.ndim - 1)))
    npad = _CUBIC_NPAD
    for ax in axes:
        n = t.shape[1 + ax]
        out_size = int(new_shape[ax])
        if n == out_size:
            continue
        x = t.movedim(1 + ax, -1).contiguous()          # (..., n)
        left = x[..., :1].expand(*x.shape[:-1], npad)    # edge (replicate) prepad
        right = x[..., -1:].expand(*x.shape[:-1], npad)
        x = torch.cat([left, x, right], dim=-1)          # (..., n+2*npad)
        x = _bspline3_prefilter_lastdim(x)
        x = _cubic_resize_lastdim(x, n, out_size, npad)
        t = x.movedim(-1, 1 + ax).contiguous()
    t = torch.minimum(torch.maximum(t, vmin), vmax)      # _clip_warp_output (per channel)
    return t.to(in_dtype)


def torch_resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                        current_spacing, new_spacing,
                                        is_seg: bool = False, order: int = 3, order_z: int = 0,
                                        force_separate_z: Union[bool, None] = False,
                                        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
                                        device=None) -> torch.Tensor:
    """GPU/torch reimplementation of resample_data_or_seg_to_shape, following the SAME
    separate-z logic as the scipy version (resample_data_or_seg):

      - decide do_separate_z + anisotropic `axis` via determine_do_sep_z_and_axis;
      - if separate-z: interpolate each axis-slice IN-PLANE (the other two dims) with
        `order` (0->nearest, 1->bilinear, 3->bicubic), batched over slices; then resample
        ALONG `axis` with `order_z` (0 -> nearest, matching scipy map_coordinates'
        scale*(i+0.5)-0.5 / mode='nearest' index mapping);
      - else: full 3-D trilinear (order>=1) / nearest (order==0), like the current fast path.

    data is (c, x, y, z). Returns a torch.Tensor (c, *new_shape) on `device`.
    is_seg uses nearest in-plane (order overridden to 0) to avoid label blending.
    """
    import torch.nn.functional as F
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = (torch.as_tensor(data, dtype=torch.float32, device=device)
         if isinstance(data, np.ndarray) else data.to(device=device, dtype=torch.float32))
    c = t.shape[0]
    shape = list(t.shape[1:])
    new_shape = [int(s) for s in new_shape]
    if shape == new_shape:
        return t

    do_sep, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                               separate_z_anisotropy_threshold)
    eff_order = 0 if is_seg else order   # segmentation: nearest in-plane
    mode2d = {0: "nearest", 1: "bilinear", 2: "bilinear", 3: "bicubic"}.get(eff_order, "bilinear")

    def interp2d(x, hw):
        if mode2d == "nearest":
            return F.interpolate(x, size=tuple(hw), mode="nearest")
        if mode2d == "bicubic":                       # order>=2 data: cubic B-spline (matches scipy)
            return _gpu_cubic_resample(x[:, 0], list(hw), axes=(0, 1)).unsqueeze(1)
        return F.interpolate(x, size=tuple(hw), mode=mode2d, align_corners=False)

    if do_sep and axis is not None:
        other = [a for a in (0, 1, 2) if a != axis]
        tp = t.permute(0, 1 + axis, 1 + other[0], 1 + other[1]).contiguous()   # (c, A, B0, B1)
        A = tp.shape[1]
        out_b = (new_shape[other[0]], new_shape[other[1]])
        x = interp2d(tp.reshape(c * A, 1, tp.shape[2], tp.shape[3]), out_b).reshape(c, A, *out_b)
        newA = new_shape[axis]
        if A != newA:                                  # resample along the anisotropic axis
            if order_z == 0 or is_seg:
                sc = float(A) / newA
                idx = torch.round(sc * (torch.arange(newA, device=device) + 0.5) - 0.5).clamp(0, A - 1).long()
                x = x.index_select(1, idx)
            else:
                x = x.permute(0, 2, 3, 1).reshape(c * out_b[0] * out_b[1], 1, A)
                x = F.interpolate(x, size=newA, mode="linear", align_corners=False)
                x = x.reshape(c, out_b[0], out_b[1], newA).permute(0, 3, 1, 2)
        order_axes = [axis, other[0], other[1]]        # x's spatial dims are in this order
        back = [order_axes.index(a) for a in (0, 1, 2)]
        out = x.permute(0, 1 + back[0], 1 + back[1], 1 + back[2]).contiguous()
    elif eff_order >= 2:                               # order>=2 data: 3-D cubic B-spline (matches scipy order=3)
        out = _gpu_cubic_resample(t, new_shape, axes=(0, 1, 2))
    else:
        m = "nearest" if eff_order == 0 else "trilinear"
        x = t.unsqueeze(0)
        out = (F.interpolate(x, size=tuple(new_shape), mode="nearest") if m == "nearest"
               else F.interpolate(x, size=tuple(new_shape), mode="trilinear", align_corners=False)).squeeze(0)
    return out


if __name__ == '__main__':
    input_array = np.random.random((1, 42, 231, 142))
    output_shape = (52, 256, 256)
    out = resample_data_or_seg(input_array, output_shape, is_seg=False, axis=3, order=1, order_z=0, do_separate_z=True)
    print(out.shape, input_array.shape)
