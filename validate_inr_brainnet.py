#!/usr/bin/env python
"""
Validation for INR-BrainNet using BrainNet's evaluation metrics.

Metrics (matching BrainNet's surface_evaluation_analysis.py):
  - ASSD  (mm) — Average Symmetric Surface Distance:
        0.5 * (mean_d(pred→GT) + mean_d(GT→pred))
        where each direction uses nearest-neighbour on 100k sampled surface points.
  - HD90  (mm) — 90th percentile Hausdorff distance:
        max(p90(pred→GT), p90(GT→pred))
  - SIF   (%)  — self-intersection fraction of faces

Val subjects are HCP subjects that come AFTER the first NUM_TRAIN_SUBJECTS in
sorted order — never seen during training.

Usage:
    python validate_inr_brainnet.py                  # latest checkpoint
    python validate_inr_brainnet.py --epoch 350      # specific epoch
    python validate_inr_brainnet.py --all            # all checkpoints in DUMP_DIR
"""

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable

import brainnet.modules.graph as graph_module
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from brainnet.mesh.surface import Surface, load_deepsurfer_template
from cortech.surface import Surface as CortechSurface

import wandb
from modules import models as metaseg_models

sys.path.insert(0, "/home/savpr/thesis/Fit-Pixels-Get-Surfaces")

# ==============================================================================
# Configuration — keep SPLIT_JSON and INR settings in sync with
# train_inr_brainnet.py
# ==============================================================================

SCRATCH_DIR = Path("/scratch/thesis-saverio/data/HCP")
GT_DIR = Path("/projects/brainnet-data/mni152/HCP")
META_WEIGHTS = (
    "/scratch/thesis-saverio/dumps/"
    "metaseg_3d_step1-normalized-robust/weights3d_num_classes_4_IS_2.pth"
)
DUMP_DIR = Path("/scratch/thesis-saverio/dumps/inr_brainnet_white_level4")
SPLIT_JSON = Path("config/HCP_split.json")

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
# Helpers
# ==============================================================================


# ==============================================================================
# Subject discovery
# ==============================================================================


def load_split_subjects(split: str, gt_level: int) -> list[str]:
    """Subject IDs for a split from SPLIT_JSON, filtered to those with a cached
    T1w and GT surfaces at gt_level. Matches train_inr_brainnet.py's selection."""
    entries = json.loads(SPLIT_JSON.read_text())[split]
    all_ids = [e["img"].split("/")[0] for e in entries]
    return [
        s
        for s in all_ids
        if (SCRATCH_DIR / s / "t1w.nii.gz").exists()
        and (GT_DIR / s / f"lh.white.resample.{gt_level}.pt").exists()
    ]


# ==============================================================================
# Phase 1: INR fitting
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
        print(f"  [{subject}] INR cached, skipping")
        return

    t1w_path = SCRATCH_DIR / subject / "t1w.nii.gz"
    if not t1w_path.exists():
        print(f"  [{subject}] t1w.nii.gz not found, skipping")
        return

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
    print(f"  [{subject}] saved → {out_path}")

    del inr
    torch.cuda.empty_cache()


# ==============================================================================
# Shared helpers
# ==============================================================================


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


def load_gt_surfaces(
    subject: str, topology: dict, gt_level: int, surf_types: tuple[str, ...]
) -> dict[str, dict[str, Surface]]:
    """GT surfaces as {surf_type: {hemi: Surface}} at subdivision level gt_level."""
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


def read_config(ckpt: dict) -> dict:
    """Architecture config for the checkpoint.

    New checkpoints store it directly; legacy ones (no 'config' key) are assumed
    to be the original level-6 white+pial model, with return_registration probed
    from the deformation layer weight shape.
    """
    cfg = ckpt.get("config")
    if cfg is not None:
        return cfg
    # white_deform.<topo>.1.conv_self.weight — index 1 is the deformation output
    # layer (index 0 is the UNet, whose conv_self layers have shape [96, 96, 1]
    # and would give a false positive for return_registration).
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


def build_graph_model(config: dict) -> graph_module.TopoFit:
    """Build TopoFit matching the checkpoint's stored architecture config."""
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
# Metrics
# ==============================================================================


def _to_cortech(surf: Surface) -> CortechSurface:
    return CortechSurface(
        surf.vertices[0].cpu().numpy(),
        surf.get_faces().cpu().numpy(),
    )


def _surface_distances(pred_surf: Surface, gt_surf: Surface) -> dict[str, float]:
    """ASSD, HD90, SIF — 1:1 match with BrainNet's surface_evaluation.py.

    Note: medial wall masking is not applied (medial_wall.npz not shipped in repo).
    """
    s_pred = _to_cortech(pred_surf)
    s_true = _to_cortech(gt_surf)

    true_to_pred = s_pred.distance_query(s_true.vertices)
    pred_to_true = s_true.distance_query(s_pred.vertices)

    assd = float(np.concatenate([true_to_pred, pred_to_true]).mean())
    hd90 = float(max(np.percentile(true_to_pred, 90), np.percentile(pred_to_true, 90)))
    sif = np.unique(s_pred.self_intersections().ravel()).size / s_pred.n_faces * 100

    return {"assd_mm": assd, "hd90_mm": hd90, "sif_pct": sif}


def _is_invalid(surf: Surface) -> bool:
    return not torch.isfinite(surf.vertices).all().item()


def compute_metrics(
    out: dict,  # {hemi: {surf_type: Surface}}
    gt: dict,  # {surf_type: {hemi: Surface}}
    surf_types: tuple[str, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for surf_type in surf_types:
        assd_acc = hd90_acc = sif_acc = 0.0
        for hemi in ("lh", "rh"):
            pred_surf: Surface = out[hemi][surf_type]
            gt_surf: Surface = gt[surf_type][hemi]
            if _is_invalid(pred_surf):
                print(
                    f"    WARNING: NaN/Inf in {surf_type} {hemi} pred vertices — skipping distances"
                )
                assd_acc += float("nan")
                hd90_acc += float("nan")
                sif_acc += float("nan")
                continue
            d = _surface_distances(pred_surf, gt_surf)
            assd_acc += d["assd_mm"]
            hd90_acc += d["hd90_mm"]
            sif_acc += d["sif_pct"]
        metrics[f"{surf_type}/assd_mm"] = assd_acc / 2
        metrics[f"{surf_type}/hd90_mm"] = hd90_acc / 2
        metrics[f"{surf_type}/sif_pct"] = sif_acc / 2
    return metrics


# ==============================================================================
# Validation loop for one checkpoint
# ==============================================================================


def validate_checkpoint(
    ckpt_path: Path,
    val_subjects: list[str],
    template: dict,
    run: wandb.sdk.wandb_run.Run,
    step: int,
) -> tuple[dict[str, float], int]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = read_config(ckpt)
    model = build_graph_model(config)
    epoch = ckpt["epoch"]
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    surf_types: tuple[str, ...] = (
        ("white", "pial") if config["return_pial"] else ("white",)
    )

    agg: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for subj in val_subjects:
            inr, inr_transform = load_subject_inr(subj)
            gt = load_gt_surfaces(subj, model.out_topology, config["gt_level"], surf_types)

            out = model(
                features=None,
                template=template,
                return_pial=config["return_pial"],
                inr_model=inr,
                inr_transform=inr_transform,
            )

            m = compute_metrics(out, gt, surf_types)
            for k, v in m.items():
                agg[k].append(v)

            summary = " | ".join(
                f"{st} {m[f'{st}/assd_mm']:.3f} mm "
                f"(HD90 {m[f'{st}/hd90_mm']:.3f}, SIF {m[f'{st}/sif_pct']:.1f}%)"
                for st in surf_types
            )
            print(f"  {subj} | {summary}")
            run.log(
                {f"val/subj/{k}": v for k, v in m.items()}
                | {"subject": subj, "epoch": epoch},
                step=step,
            )
            step += 1

            del inr, inr_transform, gt, out
            torch.cuda.empty_cache()

    agg_mean = {k: float(np.mean(v)) for k, v in agg.items()}

    print(f"\n  === Epoch {epoch:03d} aggregate (n={len(val_subjects)}) ===")
    for st in surf_types:
        print(
            f"  {st:6s} ASSD {agg_mean[f'{st}/assd_mm']:.3f} mm  "
            f"HD90 {agg_mean[f'{st}/hd90_mm']:.3f} mm  "
            f"SIF {agg_mean[f'{st}/sif_pct']:.2f}%"
        )
    print()

    run.log({f"val/{k}": v for k, v in agg_mean.items()} | {"epoch": epoch}, step=step)

    del model
    torch.cuda.empty_cache()
    return agg_mean, step


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Validate INR-BrainNet checkpoints.")
    p.add_argument(
        "--dump-dir", type=Path, default=DUMP_DIR,
        help=f"Experiment dir with checkpoints (default: {DUMP_DIR})",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--epoch", type=int, default=None, help="Evaluate this specific epoch."
    )
    g.add_argument(
        "--all", action="store_true", help="Evaluate all checkpoints in DUMP_DIR."
    )
    return p.parse_args()


def main():
    args = parse_args()
    dump_dir = args.dump_dir

    # Checkpoints to evaluate
    if args.all:
        ckpt_paths = sorted(dump_dir.glob("checkpoint_epoch_*.pth"))
    elif args.epoch is not None:
        ckpt_paths = [dump_dir / f"checkpoint_epoch_{args.epoch:03d}.pth"]
    else:
        ckpt_paths = sorted(dump_dir.glob("checkpoint_epoch_*.pth"))[-1:]

    if not ckpt_paths:
        print(f"No checkpoints found in {dump_dir}.", file=sys.stderr)
        sys.exit(1)

    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        print(f"Checkpoint(s) not found: {[p.name for p in missing]}", file=sys.stderr)
        sys.exit(1)

    print(f"Checkpoints to evaluate: {[p.name for p in ckpt_paths]}")

    # Architecture config is shared across a dump's checkpoints — read it once.
    config = read_config(torch.load(ckpt_paths[0], map_location="cpu"))
    print(
        f"Config: out_order={config['out_order']} gt_level={config['gt_level']} "
        f"return_pial={config['return_pial']}"
    )

    val_subjects = load_split_subjects("val", config["gt_level"])
    if not val_subjects:
        print(f"No val subjects found in {SPLIT_JSON}.", file=sys.stderr)
        sys.exit(1)
    print(f"Val subjects ({len(val_subjects):2d}): {val_subjects}")

    # Phase 1: fit INR for val subjects missing cached weights
    needs_fitting = [
        s for s in val_subjects if not (SCRATCH_DIR / s / "inr_weights.pth").exists()
    ]
    if needs_fitting:
        print(f"\nFitting INR for {len(needs_fitting)} val subject(s)…")
        meta_ckpt = torch.load(META_WEIGHTS, map_location="cpu")
        meta_weights = meta_ckpt["best_inr_weights"]
        for subj in needs_fitting:
            fit_and_save_inr(subj, meta_weights)
        del meta_weights, meta_ckpt
        torch.cuda.empty_cache()

    template_surfaces = load_deepsurfer_template(0, "white")  # in_order=0
    template = {h: s.vertices.to(DEVICE) for h, s in template_surfaces.items()}

    run = wandb.init(
        entity="s240099-danmarks-tekniske-universitet-dtu",
        project="Master Thesis [INR-BrainNet]",
        name=(
            "Validation — all checkpoints"
            if args.all
            else f"Validation — epoch {args.epoch}"
            if args.epoch
            else "Validation — latest checkpoint"
        ),
        notes=(
            "ASSD, HD90, SIF via cortech.Surface — 1:1 match with BrainNet's "
            "surface_evaluation.py. Val subjects from config/HCP_split.json."
        ),
        config={
            "val_subjects": val_subjects,
            "num_val_subjects": len(val_subjects),
            "split_json": str(SPLIT_JSON),
            "inr_fit_steps": INR_FIT_STEPS,
            "checkpoints": [p.name for p in ckpt_paths],
            "dump_dir": str(dump_dir),
            "arch_config": config,
            "metric": "ASSD + HD90 (cortech distance_query, symmetric) + SIF (n_faces)",
        },
    )

    step = 0
    for ckpt_path in ckpt_paths:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {ckpt_path.name}")
        print("=" * 60)
        _, step = validate_checkpoint(ckpt_path, val_subjects, template, run, step)

    run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
