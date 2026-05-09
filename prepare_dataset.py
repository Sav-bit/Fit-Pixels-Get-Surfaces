import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

"""
Script to prepare Dataset, converting Freesurfer labels to a simplified set of 4 classes (WM, GM, CSF, Extra) + background. Also checks for presence of lesion label and unexpected labels.
"""


# Collapsed classes
BACKGROUND = 0
WM = 1
GM = 2
CSF = 3
EXTRA = 4

LESION_LABEL = 33

BACKGROUND_LABELS = {
    0,   # Unknown
    45,  # background
}

WM_LABELS = {
    1,   # Left-Cerebral-White-Matter
    5,   # Left-Cerebellum-White-Matter
    13,  # Brain-Stem -> forced to WM
    18,  # Right-Cerebral-White-Matter
    22,  # Right-Cerebellum-White-Matter
    31,  # WM-hypointensities
    32,  # Optic-Chiasm
    49,  # Left-Fornix
    50,  # Right-Fornix
}

GM_LABELS = {
    2,   # Left-Cerebral-Cortex
    6,   # Left-Cerebellum-Cortex
    7,   # Left-Thalamus
    8,   # Left-Caudate
    9,   # Left-Putamen
    10,  # Left-Pallidum
    14,  # Left-Hippocampus
    15,  # Left-Amygdala
    17,  # Left-Accumbens-area
    19,  # Right-Cerebral-Cortex
    23,  # Right-Cerebellum-Cortex
    24,  # Right-Thalamus
    25,  # Right-Caudate
    26,  # Right-Putamen
    27,  # Right-Pallidum
    28,  # Right-Hippocampus
    29,  # Right-Amygdala
    30,  # Right-Accumbens-area
    47,  # Left-HypoThal-noMB
    48,  # Right-HypoThal-noMB
    51,  # Left-MammillaryBody
    52,  # Right-MammillaryBody
    53,  # Left-Basal-Forebrain
    54,  # Right-Basal-Forebrain
    55,  # Left-SeptalNuc
    56,  # Right-SeptalNuc
}

CSF_LABELS = {
    3,   # Left-Lateral-Ventricle
    4,   # Left-Inf-Lat-Vent
    11,  # 3rd-Ventricle
    12,  # 4th-Ventricle
    16,  # CSF
    20,  # Right-Lateral-Ventricle
    21,  # Right-Inf-Lat-Vent
}

ALL_DEFINED_LABELS = (
    BACKGROUND_LABELS
    | WM_LABELS
    | GM_LABELS
    | CSF_LABELS
    | {LESION_LABEL}
    | {
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46
    }  # known extracerebral labels from your LUT
)


def load_nifti_array(path: Path, dtype=None):
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    if dtype is not None:
        arr = arr.astype(dtype)
    return img, arr


def save_nifti(array: np.ndarray, reference_img: nib.Nifti1Image, out_path: Path, dtype=None):
    header = reference_img.header.copy()
    out_img = nib.Nifti1Image(array, reference_img.affine, header)
    if dtype is not None:
        out_img.set_data_dtype(dtype)
    nib.save(out_img, str(out_path))


def collapse_segmentation(seg: np.ndarray) -> np.ndarray:
    """
    Collapse original labels into:
      0 background
      1 white matter
      2 gray matter
      3 csf
      4 extra-cerebral
    """
    collapsed = np.full(seg.shape, EXTRA, dtype=np.uint8)

    collapsed[np.isin(seg, list(BACKGROUND_LABELS))] = BACKGROUND
    collapsed[np.isin(seg, list(WM_LABELS))] = WM
    collapsed[np.isin(seg, list(GM_LABELS))] = GM
    collapsed[np.isin(seg, list(CSF_LABELS))] = CSF

    # lesion remains EXTRA by design
    return collapsed


def warn_on_labels(seg: np.ndarray, subject_id: str):
    present = set(np.unique(seg).astype(int).tolist())

    if LESION_LABEL in present:
        lesion_voxels = int(np.count_nonzero(seg == LESION_LABEL))
        print(f"[{subject_id}] WARNING: lesion found ({lesion_voxels} voxels with label 33)")

    unexpected = sorted(present - ALL_DEFINED_LABELS)
    if unexpected:
        print(f"[{subject_id}] WARNING: unexpected labels found: {unexpected}")


def copy_t1_as_gz(t1_src: Path, t1_dst: Path):
    img = nib.load(str(t1_src))
    nib.save(img, str(t1_dst))


def process_subject(sub_dir: Path, out_root: Path):
    subject_id = sub_dir.name

    t1_src = sub_dir / "T1w.nii"
    seg_src = sub_dir / "brainseg_with_extracerebral.nii"

    if not t1_src.exists():
        print(f"[{subject_id}] Skipping: missing {t1_src.name}")
        return

    if not seg_src.exists():
        print(f"[{subject_id}] Skipping: missing {seg_src.name}")
        return

    out_dir = out_root / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_dst = out_dir / "t1w.nii.gz"
    seg_dst = out_dir / "seg.nii.gz"

    # Save T1 compressed with desired name
    copy_t1_as_gz(t1_src, t1_dst)

    # Collapse segmentation and save
    seg_img, seg = load_nifti_array(seg_src, dtype=np.int16)
    warn_on_labels(seg, subject_id)

    collapsed = collapse_segmentation(seg)
    save_nifti(collapsed, seg_img, seg_dst, dtype=np.uint8)

    vals, cnts = np.unique(collapsed, return_counts=True)
    summary = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts))
    print(f"[{subject_id}] Done -> {seg_dst} | counts {{{summary}}}")


def find_subject_dirs(src_root: Path):
    return sorted([p for p in src_root.glob("sub-*") if p.is_dir()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/projects/brainnet-data/mni152/HCP"),
        help="Root containing sub-xxx folders"
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("/scratch/thesis-saverio/data/HCP"),
        help="Output root"
    )
    args = parser.parse_args()

    subject_dirs = find_subject_dirs(args.src_root)
    if not subject_dirs:
        raise RuntimeError(f"No subject folders found under {args.src_root}")

    args.dst_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(subject_dirs)} subjects")
    print(f"Source: {args.src_root}")
    print(f"Destination: {args.dst_root}")

    for sub_dir in subject_dirs:
        process_subject(sub_dir, args.dst_root)

    print("All done.")


if __name__ == "__main__":
    main()