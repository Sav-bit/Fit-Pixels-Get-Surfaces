import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

t1_path = "test_img.nii"
seg_path = "test_seg_original.nii"

KEEP_LABELS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56
]

t1_img = nib.load(t1_path)
seg_img = nib.load(seg_path)

t1 = t1_img.get_fdata()
seg = seg_img.get_fdata().astype(np.int16)

# This must be true. If not, resample the segmentation to T1 space first.
if t1.shape != seg.shape:
    raise ValueError(f"Shape mismatch: T1 {t1.shape} vs SEG {seg.shape}")

mask = np.isin(seg, KEEP_LABELS)
mask = binary_fill_holes(mask).astype(np.uint8)

t1_stripped = t1 * mask

# mask_img = nib.Nifti1Image(mask.astype(np.uint8), t1_img.affine, t1_img.header)
stripped_img = nib.Nifti1Image(t1_stripped.astype(t1.dtype), t1_img.affine, t1_img.header)

# nib.save(mask_img, "brain_mask.nii.gz")
nib.save(stripped_img, "t1_skullstripped.nii.gz")