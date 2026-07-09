#!/usr/bin/env python
"""Build one fixed brain sampling mask on the shared MNI152 grid.

The INR-MAML surface training (``train_maml_surface.py``) only ever queries the
INR at/near the cortical surface — the graph never samples background. So there
is no reason to reconstruct (and draw recon coords from) the whole volume, most
of which is background. This builds a single generous brain mask, shared by
every subject, that ``CoordSurfaceDataset`` samples recon coords from.

Definition (chosen after measuring alternatives, see
project-maml-surface-training memory):
  - union over ALL training subjects of the 5-class seg labels 1,2,3
    (WM + GM + CSF; label 0 background and 4 extra-cerebral excluded).
  - This deliberately keeps the *whole brain* (cerebellum, brainstem,
    subcortical GM, ventricles are all folded into 1-3). Carving those out was
    measured to save only ~1% of volume once dilated, so we don't bother.
  - optionally dilated by ``DILATE_RADIUS`` voxels (euclidean ball -> uniform mm
    margin). 0 -> no dilation.

All subjects share the identical MNI152 grid (182,218,182; 1 mm; origin
[90,-126,-72]), so a single mask fits all. The mask is stored WITH its affine
(as a NIfTI) so lookup is done in world space and works for any subject
resolution (0.5 mm etc.): world coord -> mask-voxel via inverse affine ->
nearest-neighbour. Saved as binary uint8.

Run:
    python build_brain_mask.py               # default DILATE_RADIUS
    python build_brain_mask.py --dilate 0    # raw union, no margin
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

# --- config ---------------------------------------------------------------
SCRATCH_DIR = Path("/scratch/thesis-saverio/data/HCP")  # holds <sub>/seg.nii.gz
SPLIT_JSON = Path("config/HCP_split.json")
SPLIT = "train"  # build the mask from the training split only
BRAIN_LABELS = (1, 2, 3)  # WM, GM, CSF (exclude 0 bg and 4 extra-cerebral)
DILATE_RADIUS = 8  # voxels (== mm at 1 mm iso); euclidean ball; 0 disables


def ball(radius: int) -> np.ndarray:
    """Boolean euclidean ball structuring element of given radius (voxels)."""
    r = int(radius)
    zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy + zz * zz) <= r * r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dilate", type=int, default=DILATE_RADIUS,
                    help="dilation radius in voxels (0 = raw union)")
    ap.add_argument("--split", default=SPLIT)
    args = ap.parse_args()

    out_path = SCRATCH_DIR / f"brain_sampling_mask_union_dilate{args.dilate}.nii.gz"
    subjects = [e["img"].split("/")[0] for e in json.loads(SPLIT_JSON.read_text())[args.split]]

    union = None
    ref_nib = None
    n_used = 0
    for i, subj in enumerate(subjects):
        seg_path = SCRATCH_DIR / subj / "seg.nii.gz"
        if not seg_path.exists():
            print(f"  [{subj}] no seg.nii.gz, skipping")
            continue
        n = nib.load(seg_path)
        brain = np.isin(np.asanyarray(n.dataobj), BRAIN_LABELS)  # int labels
        if union is None:
            union, ref_nib = brain.copy(), n
        else:
            assert brain.shape == union.shape, f"{subj} shape {brain.shape} != {union.shape}"
            union |= brain
        n_used += 1
        if (i + 1) % 50 == 0 or i == len(subjects) - 1:
            print(f"  {i + 1}/{len(subjects)} subjects, union voxels: {int(union.sum()):,}")

    assert union is not None, "no segmentations found"
    tot = union.size
    print(f"\nUsed {n_used} subjects.")
    print(f"Union (labels {BRAIN_LABELS}): {int(union.sum()):,} vox ({union.sum() / tot:.1%})")

    mask = union if args.dilate <= 0 else binary_dilation(union, structure=ball(args.dilate))
    if args.dilate > 0:
        print(f"Dilated (r={args.dilate}): {int(mask.sum()):,} vox ({mask.sum() / tot:.1%})")

    out = nib.Nifti1Image(mask.astype(np.uint8), ref_nib.affine, ref_nib.header)
    out.header.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
