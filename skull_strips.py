import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes
from pathlib import Path
import argparse
import sys

# SynthSeg label IDs that should be retained in the T1 image.
SYNTHSEG_KEEP_LABELS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56
]

def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        "SAVE_DEBUG must be a boolean value like true/false or 1/0"
    )


def stripped_output_path(t1_path: str) -> Path:
    input_path = Path(t1_path)
    base_name = input_path.name
    if base_name.endswith(".nii.gz"):
        stem = base_name[:-7]
        suffix = ".nii.gz"
    else:
        stem = input_path.stem
        suffix = input_path.suffix

    return input_path.with_name(f"{stem}_skullstripped{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Skull strip a T1 image using a SynthSeg segmentation mask."
    )
    parser.add_argument("t1_path", help="Path to the input nii/nii.gz image")
    parser.add_argument("seg_path", help="Path to the SynthSeg segmentation image")
    parser.add_argument(
        "SAVE_DEBUG",
        type=parse_bool,
        help="Whether to save the debug labels image (true/false)",
    )
    args = parser.parse_args()

    t1_img = nib.load(args.t1_path)
    seg_img = nib.load(args.seg_path)

    t1 = t1_img.get_fdata()
    seg = seg_img.get_fdata().astype(np.int16)

    # This must be true. If not, resample the segmentation to T1 space first.
    if t1.shape != seg.shape:
        raise ValueError(f"Shape mismatch: T1 {t1.shape} vs SEG {seg.shape}")

    mask_orig = np.isin(seg, SYNTHSEG_KEEP_LABELS)
    mask = binary_fill_holes(mask_orig).astype(np.uint8)

    t1_stripped = t1 * mask
    stripped_img = nib.Nifti1Image(t1_stripped.astype(t1.dtype), t1_img.affine, t1_img.header)

    nib.save(stripped_img, str(stripped_output_path(args.t1_path)))

    if args.SAVE_DEBUG:
        debug_seg = np.where(mask_orig, seg, 0).astype(np.int16)
        nib.save(
            nib.Nifti1Image(debug_seg, seg_img.affine, seg_img.header),
            "debug_labels_inside_mask.nii.gz"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())