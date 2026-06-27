#!/usr/bin/env python
"""
INR-BrainNet training.

Phase 1 (preprocess): Fit MetaSeg SIREN INR to each subject's T1w. Weights
    are cached at <scratch_dir>/<subject>/inr_weights.pth and reused on
    re-runs so Phase 1 is skippable once done.

Phase 2 (train): Build a graph-only TopoFit (no UNet) where in_channels=256
    matches the INR penultimate-layer feature dim. At each step:
    - load the subject's fitted INR weights
    - build inr_transform: MNI152 world mm -> INR [-1,1]
    - call graph.forward(features=None, template, inr_model, inr_transform)
    - compute loss vs Freesurfer GT surfaces (level 6, world mm)
    - backprop through the graph deformation network only (INR is frozen)

Losses (BrainNet original weights and schedule):
    Epochs   0-200: sampled chamfer (1.0) + Taubin (40/20) + edge_var + tri_quality + thickness
    Epoch  201+:    negloglik (0.5, uses sigma from graph) replaces chamfer; LR x0.5
    Epoch  401+:    edge_var/tri_quality reduced further
    Epoch  601+:    LR x0.5

Sigma (per-vertex uncertainty) is produced by the graph deformation network
as an additional output alongside the vertex displacement — enabled by default
(return_uncertainty=True). NegLogLik weights each surface point's error by
1/sigma^2, so the model learns to be uncertain where prediction is hard.
"""

import copy
import math
import sys

sys.path.insert(0, "/home/savpr/thesis/Fit-Pixels-Get-Surfaces")

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import brainnet.modules.graph as graph_module
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from brainnet.config.base import LossParameters
from brainnet.config.topofit.losses import train as _brainnet_loss_config
from brainnet.mesh.surface import Surface, load_deepsurfer_template
from brainnet.modules.criterion import Criterion

import wandb
from modules import models as metaseg_models

# ==============================================================================
# Configuration
# ==============================================================================

SCRATCH_DIR = Path("/scratch/thesis-saverio/data/HCP")
GT_DIR = Path("/projects/brainnet-data/mni152/HCP")
META_WEIGHTS = "/scratch/thesis-saverio/dumps/metaseg_3d_step1-normalized-robust/weights3d_num_classes_4_IS_2.pth"
DUMP_DIR = Path("/scratch/thesis-saverio/dumps/inr_brainnet_sampled_chamfer")

INR_FIT_STEPS = 300
SKIP_PIXELS = 2
TRAIN_EPOCHS = 500
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SUBJECTS = 100  # set to None to use all available

# Loss schedule — identical to brainnet/config/topofit/events_trainer.py.
# At epoch 201 chamfer→0 and negloglik→0.5 (we now have sigma from the graph,
# so we can follow BrainNet's schedule exactly). Taubin stays at 40/20.
_LOSS_SCHEDULE: dict[int, dict] = {
    201: {
        "weights": {
            ("white", "chamfer"): 0.0,
            ("pial", "chamfer"): 0.0,
            ("white", "negloglik"): 0.5,
            ("pial", "negloglik"): 0.5,
            ("white", "edge_var"): 2.0,
            ("pial", "edge_var"): 1.0,
            ("white", "tri_quality"): 1.0,
            ("pial", "tri_quality"): 1.0,
        },
        "lr_factor": 0.5,
    },
    401: {
        "weights": {
            ("white", "edge_var"): 1.0,
            ("pial", "edge_var"): 0.5,
            ("white", "tri_quality"): 0.5,
            ("pial", "tri_quality"): 0.5,
        },
    },
    601: {"lr_factor": 0.5},
}


def _swap(d: dict) -> dict:
    """Swap the two outer levels of a nested dict: {A: {B: v}} → {B: {A: v}}."""
    outer = tuple(d.keys())
    inner = tuple(d[outer[0]].keys())
    return {b: {a: d[a][b] for a in outer} for b in inner}


def _dict_sum(d: dict) -> torch.Tensor:
    total = 0.0
    for v in d.values():
        total = total + (_dict_sum(v) if isinstance(v, dict) else v)
    return total


def _discover_subjects(n: int | None) -> list[str]:
    """Return sorted subject IDs that have both T1w and GT surfaces."""
    subjects = sorted(
        d.name
        for d in SCRATCH_DIR.iterdir()
        if d.is_dir()
        and (d / "t1w.nii.gz").exists()
        and (GT_DIR / d.name / "lh.white.resample.6.pt").exists()
    )
    return subjects[:n] if n is not None else subjects


SUBJECTS = _discover_subjects(NUM_SUBJECTS)

INR_CONFIG = {
    "in_features": 3,
    "out_features": 1,
    "hidden_features": 256,
    "hidden_layers": 4,
}
INR_FEAT_DIM = 256


# ==============================================================================
# Phase 1 helpers: image loading + INR fitting
# ==============================================================================


def normalize_image(img_data: np.ndarray) -> np.ndarray:
    """Percentile clip + [0,1] normalization on non-zero voxels (matches dataloader)."""
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


def get_coords_from_affine(h: int, w: int, d: int, affine: np.ndarray) -> torch.Tensor:
    """Build a (H, W, D, 3) tensor of world coords (mm) via the NIfTI affine."""
    ii = torch.arange(h, dtype=torch.float32)
    jj = torch.arange(w, dtype=torch.float32)
    kk = torch.arange(d, dtype=torch.float32)
    gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing="ij")
    voxel_coords = torch.stack([gi, gj, gk, torch.ones_like(gi)], dim=-1)
    affine_t = torch.from_numpy(affine).float()
    return (voxel_coords @ affine_t.T)[..., :3]


def fit_and_save_inr(subject: str, meta_weights: dict) -> None:
    """Fit INR to subject's T1w; skip if weights already cached."""
    out_path = SCRATCH_DIR / subject / "inr_weights.pth"
    if out_path.exists():
        print(f"  [{subject}] cached, skipping")
        return

    t1w_path = SCRATCH_DIR / subject / "t1w.nii.gz"
    if not t1w_path.exists():
        print(f"  [{subject}] t1w.nii.gz not found, skipping")
        return

    img_nib = nib.load(t1w_path)
    img_data = normalize_image(img_nib.get_fdata().astype(np.float32))
    h, w, d = img_data.shape

    coords_mtx = get_coords_from_affine(h, w, d, img_nib.affine)
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
    print(f"  [{subject}] saved -> {out_path}")

    del inr
    torch.cuda.empty_cache()


# ==============================================================================
# Phase 2 helpers: INR loading, GT surfaces, criterion, model
# ==============================================================================


def load_subject_inr(
    subject: str,
) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """Load cached INR + build the per-subject inr_transform closure."""
    ckpt = torch.load(SCRATCH_DIR / subject / "inr_weights.pth", map_location=DEVICE)
    c_min = ckpt["c_min"].to(DEVICE)
    c_max = ckpt["c_max"].to(DEVICE)

    inr = metaseg_models.INR(**INR_CONFIG).float().to(DEVICE)
    inr.load_state_dict(ckpt["inr_weights"])
    inr.eval()
    inr.requires_grad_(False)

    def inr_transform(v: torch.Tensor) -> torch.Tensor:
        # v: (N, 3, M) — template vertices in MNI152 world mm.
        # Normalize to INR coordinate space [-1, 1].
        return 2.0 * (v.mT - c_min) / (c_max - c_min) - 1.0

    return inr, inr_transform


def load_gt_surfaces(subject: str, topology: dict) -> dict[str, dict[str, Surface]]:
    """GT surfaces as {surf_type: {hemi: Surface}} at subdivision level 6."""
    gt_dir = GT_DIR / subject
    gt: dict[str, dict[str, Surface]] = {}
    for surf_type in ("white", "pial"):
        gt[surf_type] = {}
        for hemi in ("lh", "rh"):
            verts = torch.load(
                gt_dir / f"{hemi}.{surf_type}.resample.6.pt", map_location=DEVICE
            )
            gt[surf_type][hemi] = Surface(verts.unsqueeze(0), topology[hemi])
    return gt


def build_criterion() -> Criterion:
    """BrainNet Criterion with original loss weights; sphere.reg disabled (no GT)."""
    head_weights = copy.deepcopy(_brainnet_loss_config.head_weights)
    head_weights["sphere.reg"] = 0.0
    config = LossParameters(
        _brainnet_loss_config.functions,
        head_weights,
        copy.deepcopy(_brainnet_loss_config.loss_weights),
    )
    return Criterion(config)


def apply_schedule(
    epoch: int, criterion: Criterion, optimizer: torch.optim.Optimizer
) -> bool:
    """Apply BrainNet loss schedule at checkpoint epochs."""
    event = _LOSS_SCHEDULE.get(epoch)
    if event is None:
        return False
    if "weights" in event:
        criterion.update_loss_weights(event["weights"])
    if "lr_factor" in event:
        for g in optimizer.param_groups:
            g["lr"] *= event["lr_factor"]
    return True


def compute_surface_loss(
    out: dict,  # {hemi: {surf_type: Surface}} — graph model output
    gt: dict,  # {surf_type: {hemi: Surface}} — GT surfaces
    criterion: Criterion,
) -> tuple[torch.Tensor, dict]:
    """Compute losses using BrainNet's Criterion.

    Format contract:
      prepare_for_surface_loss expects {surf_type: {hemi: Surface}}
      criterion.forward        expects {hemi: {surf_type: Surface}}

    `out` is already in {hemi: {surf_type: Surface}} format.
    `gt`  is already in {surf_type: {hemi: Surface}} format.
    """
    # Swap out to {surf_type: {hemi: Surface}} for sampling preprocessing.
    # Surface objects are mutated in-place (interpolated data populated), so
    # the same objects are accessible via `out` during criterion.forward.
    y_pred_surf = _swap(out)
    criterion.prepare_for_surface_loss(y_pred_surf, gt)

    # criterion.forward needs {hemi: {surf_type: Surface}} for both pred and gt.
    y_true_hemi = _swap(gt)
    loss_dict = criterion(out, y_true_hemi)

    weighted = criterion.apply_weights(loss_dict)
    total = _dict_sum(weighted)

    # Flatten loss_dict for logging
    log_vals: dict[str, float] = {}
    for head, losses in loss_dict.items():
        for name, val in losses.items():
            log_vals[f"{head}/{name}"] = val.item()

    # Chamfer in mm (RMSE) — only present when weight > 0 (epochs 0-200)
    if "chamfer" in loss_dict.get("white", {}):
        log_vals["white/chamfer_mm"] = math.sqrt(loss_dict["white"]["chamfer"].item())
    if "chamfer" in loss_dict.get("pial", {}):
        log_vals["pial/chamfer_mm"] = math.sqrt(loss_dict["pial"]["chamfer"].item())

    return total, log_vals


def build_graph_model() -> graph_module.TopoFit:
    """Graph-only TopoFit with a single 256-dim INR feature map at every level.

    return_uncertainty=True (default): graph outputs sigma alongside vertex
    displacements, enabling NegLogLik loss at epoch 201+.
    return_registration=False: no sphere.reg prediction (no GT available).
    """
    return graph_module.TopoFit(
        in_channels={"inr": INR_FEAT_DIM},
        white_feature_maps=[["inr"]] * 7,
        pial_feature_maps=["inr"],
        in_order=0,
        out_order=6,
        max_order=6,
        white_channels={"encoder": [96, 96, 96, 96], "decoder": [96, 96, 96]},
        pial_channels=[32],
        pial_deform_module="LinearDeformationBlock",
        return_registration=False,
    ).to(DEVICE)


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: fit INR per subject (skipped for subjects already cached)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: INR fitting")
    print("=" * 60)

    meta_ckpt = torch.load(META_WEIGHTS, map_location="cpu")
    meta_weights = meta_ckpt["best_inr_weights"]

    for subj in SUBJECTS:
        fit_and_save_inr(subj, meta_weights)

    del meta_weights, meta_ckpt
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: train graph module
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: training graph module")
    print("=" * 60)

    model = build_graph_model()
    criterion = build_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    template_surfaces = load_deepsurfer_template(model.in_order, "white")
    template = {h: s.vertices.to(DEVICE) for h, s in template_surfaces.items()}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Subjects: {SUBJECTS}")
    print(f"Epochs: {TRAIN_EPOCHS}, LR: {LR}")
    print(f"Graph params: {n_params:,}\n")

    run = wandb.init(
        entity="s240099-danmarks-tekniske-universitet-dtu",
        project="Master Thesis [INR-BrainNet]",
        name="Graph training — Sampled Chamfer + NegLogLik + Taubin 40/20",
        notes=(
            "BrainNet original losses: sampled chamfer (100k pts + CUDA NN) for "
            "epochs 0-200, then negloglik (using sigma from graph deform output) "
            "from epoch 201. Taubin restored to BrainNet original weights (40/20). "
            "Sphere.reg disabled (no GT). return_uncertainty=True (default)."
        ),
        config={
            "num_subjects": len(SUBJECTS),
            "num_epochs": TRAIN_EPOCHS,
            "subjects": SUBJECTS,
            "inr_fit_steps": INR_FIT_STEPS,
            "skip_pixels": SKIP_PIXELS,
            "lr": LR,
            "inr_feat_dim": INR_FEAT_DIM,
            "graph_params": n_params,
            "device": DEVICE,
            "loss_weights_white": _brainnet_loss_config.loss_weights["white"],
            "loss_weights_pial": _brainnet_loss_config.loss_weights["pial"],
            "head_weights": {
                k: v
                for k, v in _brainnet_loss_config.head_weights.items()
                if k != "sphere.reg"
            },
        },
    )

    step = 0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        if apply_schedule(epoch, criterion, optimizer):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  [schedule] epoch {epoch:03d}: lr={current_lr:.2e}")

        epoch_loss = 0.0
        model.train()

        for subj in SUBJECTS:
            inr, inr_transform = load_subject_inr(subj)
            gt = load_gt_surfaces(subj, model.out_topology)

            optimizer.zero_grad()

            out = model(
                features=None,
                template=template,
                return_pial=True,
                inr_model=inr,
                inr_transform=inr_transform,
            )

            loss, log_vals = compute_surface_loss(out, gt, criterion)

            loss.backward()
            optimizer.step()

            subj_loss = loss.item()
            epoch_loss += subj_loss
            print(f"  epoch {epoch:03d} | {subj} | loss {subj_loss:.4f}")

            wandb.log(
                {
                    "train/loss": subj_loss,
                    **{f"train/{k}": v for k, v in log_vals.items()},
                    "subject": subj,
                    "epoch": epoch,
                },
                step=step,
            )
            step += 1

            del inr, inr_transform, gt, out
            torch.cuda.empty_cache()

        avg = epoch_loss / len(SUBJECTS)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  -- epoch {epoch:03d} avg loss: {avg:.4f}")
        wandb.log(
            {"train/epoch_avg_loss": avg, "train/lr": current_lr, "epoch": epoch},
            step=step,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "avg_loss": avg,
            },
            DUMP_DIR / f"checkpoint_epoch_{epoch:03d}.pth",
        )

    run.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
