import numpy as np
from typing import List
# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
import cupy as cp
import torch
import nibabel as nib
from cupyx.scipy import ndimage
from scipy.ndimage import binary_fill_holes
from fastsegmentator._vendor.nnunetv2.utilities.utils import log_runtime
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask as get_bbox_from_mask_cpu


def _crop_dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crop_to_mask_gpu(img_in: "nib.Nifti1Image", mask_img: "nib.Nifti1Image",
                     addon=(0, 0, 0), dtype=None):
    """GPU/torch port of totalsegmentator.cropping.crop_to_mask.

    The bbox is computed with torch.any reductions on GPU (vs np.where on CPU in
    get_bbox_from_mask) — identical min/max indices, identical mm→voxel addon and
    clamping. Returns (cropped nib.Nifti1Image, bbox) exactly like the original.
    """
    dev = _crop_dev()
    # ascontiguousarray: nibabel canonical/undo views can have negative strides, which
    # torch.as_tensor rejects.
    mask_t = torch.as_tensor(np.ascontiguousarray(mask_img.dataobj), device=dev)
    addon_vox = (np.array(addon) / img_in.header.get_zooms()).astype(int)  # mm → voxels
    s = mask_t.shape
    nz = mask_t > 0                                          # get_bbox_from_mask outside_value=0
    if not bool(nz.any()):
        bbox = [[0, int(s[0])], [0, int(s[1])], [0, int(s[2])]]
    else:
        bbox = []
        for ax in range(3):
            proj = torch.any(nz, dim=tuple(d for d in range(3) if d != ax))
            idx = torch.nonzero(proj, as_tuple=False).flatten()
            lo = int(idx[0]) - int(addon_vox[ax])
            hi = int(idx[-1]) + 1 + int(addon_vox[ax])
            bbox.append([max(0, lo), min(int(s[ax]), hi)])
    data = torch.as_tensor(np.ascontiguousarray(img_in.get_fdata()), device=dev)
    cropped = data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    affine = np.copy(img_in.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1.0]))[:3]
    dt = img_in.dataobj.dtype if dtype is None else dtype
    return nib.Nifti1Image(cropped.cpu().numpy().astype(dt), affine), bbox


def undo_crop_gpu(img: "nib.Nifti1Image", ref_img: "nib.Nifti1Image", bbox):
    """GPU/torch port of totalsegmentator.cropping.undo_crop (zero-fill ref-shaped array)."""
    dev = _crop_dev()
    out = torch.zeros(tuple(ref_img.shape), device=dev, dtype=torch.float64)
    cropped = torch.as_tensor(np.ascontiguousarray(img.get_fdata()), device=dev)
    out[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = cropped
    return nib.Nifti1Image(out.cpu().numpy(), ref_img.affine)

# Assume you have a binary array 'mask' on the GPU
def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    if isinstance(data, torch.Tensor) and data.is_cuda:
        # binary_fill_holes requires a CuPy array; convert via __cuda_array_interface__
        filled_cp = ndimage.binary_fill_holes(cp.asarray(nonzero_mask))
        filled_mask = torch.as_tensor(filled_cp, device=data.device)
    elif isinstance(data, torch.Tensor):
        filled_mask = torch.as_tensor(binary_fill_holes(nonzero_mask.numpy()))
    elif isinstance(data, cp.ndarray):
        filled_mask = ndimage.binary_fill_holes(nonzero_mask)
    else:
        filled_mask = binary_fill_holes(nonzero_mask)
    return filled_mask

def get_bbox_from_mask(mask: cp.ndarray) -> List[List[int]]:
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision 
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    
    This implementation uses CuPy for GPU acceleration. The mask should be a CuPy array.
    
    Args:
        mask (cp.ndarray): 3D mask array on GPU
        
    Returns:
        List[List[int]]: Bounding box coordinates as [[minz, maxz], [minx, maxx], [miny, maxy]]
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    
    # Create range arrays on GPU
    zidx = cp.arange(Z)
    xidx = cp.arange(X)
    yidx = cp.arange(Y)
    
    # Z dimension
    for z in zidx.get():  # .get() to iterate over CPU array
        if cp.any(mask[z]).get():  # .get() to get boolean result to CPU
            minzidx = z
            break
    for z in zidx[::-1].get():
        if cp.any(mask[z]).get():
            maxzidx = z + 1
            break
            
    # X dimension
    for x in xidx.get():
        if cp.any(mask[:, x]).get():
            minxidx = x
            break
    for x in xidx[::-1].get():
        if cp.any(mask[:, x]).get():
            maxxidx = x + 1
            break
            
    # Y dimension
    for y in yidx.get():
        if cp.any(mask[:, :, y]).get():
            minyidx = y
            break
    for y in yidx[::-1].get():
        if cp.any(mask[:, :, y]).get():
            maxyidx = y + 1
            break
            
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def bounding_box_to_slice(bounding_box: List[List[int]]):
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73
    """
    return tuple([slice(*i) for i in bounding_box])

@log_runtime
def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    is_cuda = isinstance(data, torch.Tensor) and data.is_cuda
    if is_cuda:
        bbox = get_bbox_from_mask(cp.asarray(nonzero_mask))
    elif isinstance(data, cp.ndarray):
        bbox = get_bbox_from_mask(nonzero_mask)
    else:
        bbox = get_bbox_from_mask_cpu(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]

    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        if isinstance(nonzero_mask, torch.Tensor):
            nm_np = nonzero_mask.cpu().numpy()
        elif isinstance(nonzero_mask, cp.ndarray):
            nm_np = cp.asnumpy(nonzero_mask)
        else:
            nm_np = nonzero_mask
        seg = np.where(nm_np, np.int8(0), np.int8(nonzero_label))
        seg = torch.as_tensor(seg).to(data.device)
    return data, seg, bbox


