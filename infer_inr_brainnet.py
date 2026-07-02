#!/usr/bin/env python
"""
Run one subject through a checkpoint and save predicted surfaces for inspection.

Architecture (out_order, GT level, white/pial) is read from the checkpoint's
'config'; legacy checkpoints fall back to the original level-6 white+pial model.

Output GIFTI files in --out-dir (default: ./inspection/<subject>_epoch<N>/),
e.g. for a white-only model:
  lh.white.pred.surf.gii  rh.white.pred.surf.gii   ← predicted
  lh.white.gt.surf.gii    rh.white.gt.surf.gii     ← ground truth

Inspect in NiiVue (VS Code): the script prints absolute paths to click/drag in.

Usage:
  python infer_inr_brainnet.py sub-105
  python infer_inr_brainnet.py sub-105 --epoch 200
  python infer_inr_brainnet.py sub-105 --epoch 200 --dump-dir /path/to/experiment
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable

import brainnet.modules.graph as graph_module
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from brainnet.mesh.surface import Surface, load_deepsurfer_template

from modules import models as metaseg_models

sys.path.insert(0, "/home/savpr/thesis/Fit-Pixels-Get-Surfaces")

# ==============================================================================
# Configuration — keep in sync with train_inr_brainnet.py / validate_inr_brainnet.py
# ==============================================================================

SCRATCH_DIR = Path("/scratch/thesis-saverio/data/HCP")
GT_DIR = Path("/projects/brainnet-data/mni152/HCP")
META_WEIGHTS = (
    "/scratch/thesis-saverio/dumps/"
    "metaseg_3d_step1-normalized-robust/weights3d_num_classes_4_IS_2.pth"
)
DUMP_DIR = Path("/scratch/thesis-saverio/dumps/inr_brainnet_white_level4")

INR_FIT_STEPS = 300
SKIP_PIXELS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INR_CONFIG = {
    "in_features": 3,
    "out_features": 1,
    "hidden_features": 256,
    "hidden_layers": 4,
}
INR_FEAT_DIM = 256

# ==============================================================================
# Reused helpers (same as validate_inr_brainnet.py)
# ==============================================================================


def _normalize_image(img_data: np.ndarray) -> np.ndarray:
    img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)
    mask = img_data != 0
    if mask.sum() == 0:
        return np.zeros_like(img_data)
    vals = img_data[mask]
    low, high = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
    if high - low < 1e-8:
        return np.zeros_like(img_data)
    img_data = np.clip(img_data, low, high)
    img_data = (img_data - low) / (high - low)
    img_data[~mask] = 0.0
    return img_data


def _get_coords_from_affine(h, w, d, affine):
    ii = torch.arange(h, dtype=torch.float32)
    jj = torch.arange(w, dtype=torch.float32)
    kk = torch.arange(d, dtype=torch.float32)
    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing="ij")
    voxel_coords = torch.stack([gi, gj, gk, torch.ones_like(gi)], dim=-1)
    return (voxel_coords @ torch.from_numpy(affine).float().T)[..., :3]


def fit_and_save_inr(subject: str, meta_weights: dict) -> None:
    out_path = SCRATCH_DIR / subject / "inr_weights.pth"
    if out_path.exists():
        print(f"  [{subject}] INR cached, skipping fit")
        return

    t1w_path = SCRATCH_DIR / subject / "t1w.nii.gz"
    if not t1w_path.exists():
        print(f"  [{subject}] t1w.nii.gz not found", file=sys.stderr)
        sys.exit(1)

    img_nib = nib.load(t1w_path)
    img_data = _normalize_image(img_nib.get_fdata().astype(np.float32))
    h, w, d = img_data.shape

    coords_mtx = _get_coords_from_affine(h, w, d, img_nib.affine)
    flat = coords_mtx.reshape(-1, 3)
    c_min = flat.min(dim=0).values
    c_max = flat.max(dim=0).values

    coords_norm = 2.0 * (coords_mtx - c_min) / (c_max - c_min) - 1.0
    coords_sub = coords_norm[::SKIP_PIXELS, ::SKIP_PIXELS, ::SKIP_PIXELS]
    img_sub = img_data[::SKIP_PIXELS, ::SKIP_PIXELS, ::SKIP_PIXELS]
    coords_t = coords_sub.reshape(-1, 3).float().to(DEVICE)[None]
    img_t = torch.from_numpy(img_sub.reshape(-1, 1)).float().to(DEVICE)[None]

    inr = metaseg_models.INR(**INR_CONFIG).float().to(DEVICE)
    inr.load_state_dict(
        {
            k.replace("inr.", ""): v.clone().detach()
            for k, v in deepcopy(meta_weights).items()
        }
    )
    inr.compile()

    print(f"  [{subject}] fitting INR ({INR_FIT_STEPS} steps)...")
    inr.fit(coords_t, img_t, epochs=INR_FIT_STEPS, disable_tqdm=False)

    torch.save(
        {"inr_weights": inr.state_dict(), "c_min": c_min, "c_max": c_max}, out_path
    )
    del inr
    torch.cuda.empty_cache()


def load_subject_inr(subject: str) -> tuple[nn.Module, Callable]:
    ckpt = torch.load(SCRATCH_DIR / subject / "inr_weights.pth", map_location=DEVICE)
    c_min = ckpt["c_min"].to(DEVICE)
    c_max = ckpt["c_max"].to(DEVICE)

    inr = metaseg_models.INR(**INR_CONFIG).float().to(DEVICE)
    inr.load_state_dict(ckpt["inr_weights"])
    inr.eval()
    inr.requires_grad_(False)

    def inr_transform(v: torch.Tensor) -> torch.Tensor:
        return 2.0 * (v.mT - c_min) / (c_max - c_min) - 1.0

    return inr, inr_transform


def read_config(ckpt: dict) -> dict:
    """Architecture config for the checkpoint.

    New checkpoints store it directly; legacy ones (no 'config' key) are assumed
    to be the original level-6 white+pial model, with return_registration probed
    from the deformation layer weight shape.
    """
    cfg = ckpt.get("config")
    if cfg is not None:
        return cfg
    key = next(
        k
        for k in ckpt["model_state"]
        if ".1.conv_self.weight" in k and k.startswith("white_deform.")
    )
    out_ch = ckpt["model_state"][key].shape[0]
    return {
        "out_order": 6,
        "gt_level": 6,
        "return_pial": True,
        "return_registration": out_ch >= 9,
        "inr_feat_dim": INR_FEAT_DIM,
    }


def load_gt_surfaces(
    subject: str, topology: dict, gt_level: int, surf_types: tuple[str, ...]
) -> dict[str, dict[str, Surface]]:
    gt_dir = GT_DIR / subject
    gt: dict[str, dict[str, Surface]] = {}
    for surf_type in surf_types:
        gt[surf_type] = {}
        for hemi in ("lh", "rh"):
            verts = torch.load(
                gt_dir / f"{hemi}.{surf_type}.resample.{gt_level}.pt",
                map_location=DEVICE,
            )
            gt[surf_type][hemi] = Surface(verts.unsqueeze(0), topology[hemi])
    return gt


def build_graph_model(config: dict) -> graph_module.TopoFit:
    return graph_module.TopoFit(
        in_channels={"inr": config.get("inr_feat_dim", INR_FEAT_DIM)},
        white_feature_maps=[["inr"]] * 7,
        pial_feature_maps=["inr"],
        in_order=0,
        out_order=config["out_order"],
        max_order=6,
        white_channels={"encoder": [96, 96, 96, 96], "decoder": [96, 96, 96]},
        pial_channels=[32],
        pial_deform_module="LinearDeformationBlock",
        return_registration=config["return_registration"],
    ).to(DEVICE)


# ==============================================================================
# Save helpers
# ==============================================================================


def _write_surf(path: Path, surf: Surface) -> None:
    v = surf.vertices[0].cpu().numpy().astype(np.float32)
    f = surf.get_faces().cpu().numpy().astype(np.int32)
    coords = nib.gifti.GiftiDataArray(
        v,
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"],
        datatype="NIFTI_TYPE_FLOAT32",
    )
    tris = nib.gifti.GiftiDataArray(
        f,
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"],
        datatype="NIFTI_TYPE_INT32",
    )
    nib.save(nib.gifti.GiftiImage(darrays=[coords, tris]), str(path))


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="INR-BrainNet single-subject inference.")
    p.add_argument("subject", help="Subject ID, e.g. sub-105")
    p.add_argument(
        "--epoch", type=int, default=None, help="Checkpoint epoch (default: latest)"
    )
    p.add_argument(
        "--dump-dir", type=Path, default=DUMP_DIR,
        help=f"Experiment dir with checkpoints (default: {DUMP_DIR})",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    subject = args.subject
    dump_dir = args.dump_dir

    if args.epoch is not None:
        ckpt_path = dump_dir / f"checkpoint_epoch_{args.epoch:03d}.pth"
    else:
        candidates = sorted(dump_dir.glob("checkpoint_epoch_*.pth"))
        if not candidates:
            print(f"No checkpoints found in {dump_dir}", file=sys.stderr)
            sys.exit(1)
        ckpt_path = candidates[-1]

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    epoch_num = int(ckpt_path.stem.split("_")[-1])

    out_dir = args.out_dir or Path(f"inspection/{subject}_epoch{epoch_num:03d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subject  : {subject}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Output   : {out_dir}/")

    # Fit INR if not cached
    if not (SCRATCH_DIR / subject / "inr_weights.pth").exists():
        print("\nFitting INR (not cached)...")
        meta_ckpt = torch.load(META_WEIGHTS, map_location="cpu")
        fit_and_save_inr(subject, meta_ckpt["best_inr_weights"])
        del meta_ckpt
        torch.cuda.empty_cache()

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = read_config(ckpt)
    model = build_graph_model(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    surf_types: tuple[str, ...] = ("white", "pial") if config["return_pial"] else ("white",)
    print(f"Config   : out_order={config['out_order']} "
          f"gt_level={config['gt_level']} surfaces={surf_types}")

    template_surfaces = load_deepsurfer_template(model.in_order, "white")
    template = {h: s.vertices.to(DEVICE) for h, s in template_surfaces.items()}

    inr, inr_transform = load_subject_inr(subject)
    gt = load_gt_surfaces(subject, model.out_topology, config["gt_level"], surf_types)

    print("\nRunning model...")
    with torch.no_grad():
        out = model(
            features=None,
            template=template,
            return_pial=config["return_pial"],
            inr_model=inr,
            inr_transform=inr_transform,
        )

    print("Saving surfaces...")
    saved: list[Path] = []
    for hemi in ("lh", "rh"):
        for surf_type in surf_types:
            p_pred = out_dir / f"{hemi}.{surf_type}.pred.surf.gii"
            p_gt = out_dir / f"{hemi}.{surf_type}.gt.surf.gii"
            _write_surf(p_pred, out[hemi][surf_type])
            _write_surf(p_gt, gt[surf_type][hemi])
            saved.extend([p_pred, p_gt])

    t1w_path = SCRATCH_DIR / subject / "t1w.nii.gz"
    print("\nDone. Open in NiiVue (VS Code): click a file below, then 'Open With… → NiiVue',")
    print("or drag several onto an open NiiVue tab. Predicted = .pred, ground truth = .gt.\n")
    print(f"  {t1w_path.resolve()}   (background volume)")
    for p in saved:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
