"""
MetaSeg 3D test-set evaluation.

Iterates over a dataset split defined in a JSON config, runs per-image INR
fitting + segmentation inference (same pipeline as metaseg_3d_inference.py),
and reports average Dice score.

Usage:
    python metaseg_3d_test.py \
        --config config/HCP_split.json \
        --dataset-dir /scratch/thesis-saverio/data/HCP \
        [--mode test] \
        [--steps 300] \
        [--weights-file /path/to/step1.pth] \
        [--classifier-weights-file /path/to/clf.pth] \
        [--num-subjects 10]
"""

import argparse
import json
import os
import os.path as osp
import sys
from copy import deepcopy

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, osp.dirname(__file__))
from modules import metrics, models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(422)

# ===== Defaults (overridden by CLI args) =====
DEFAULT_WEIGHTS_FILE = "/scratch/thesis-saverio/dumps/metaseg_3d_step1-normalized-robust/weights3d_num_classes_4_IS_2.pth"
DEFAULT_CLASSIFIER_WEIGHTS_FILE = "/scratch/thesis-saverio/dumps/weights_3d/classifierfinal_weights_LR_5e-05_exp_gamma_3.0_INR_300it_skip_pixels_2_subset200.pth"

NUM_CLASSES = 4
NUM_CLASSES_AND_ONE = NUM_CLASSES + 1
NORMALIZE_FEATURES = False

inr_config = {
    "in_features": 3,
    "out_features": 1,
    "hidden_features": 256,
    "hidden_layers": 4,
}
segmentation_config = {
    "hidden_features": [256],
    "output_features": NUM_CLASSES_AND_ONE,
}


def get_coords_from_affine(h: int, w: int, d: int, affine: np.ndarray) -> torch.Tensor:
    ii = torch.arange(h, dtype=torch.float32)
    jj = torch.arange(w, dtype=torch.float32)
    kk = torch.arange(d, dtype=torch.float32)
    grid_i, grid_j, grid_k = torch.meshgrid(ii, jj, kk, indexing="ij")
    ones = torch.ones_like(grid_i)
    voxel_coords = torch.stack([grid_i, grid_j, grid_k, ones], dim=-1)
    affine_t = torch.from_numpy(affine).float()
    world_coords = voxel_coords @ affine_t.T
    return world_coords[..., :3]


def load_and_prepare(img_path: str, seg_path: str, skip_pixels: int):
    """Load NIfTI image + segmentation, normalize, compute coords."""
    img_nib = nib.load(img_path)
    img_data = img_nib.get_fdata().astype(np.float32)
    affine = img_nib.affine
    original_shape = img_data.shape

    seg_data = nib.load(seg_path).get_fdata().astype(np.int64)

    # Normalize — identical to metaseg_3d_inference.py and the dataloader
    img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)
    support_mask = img_data != 0
    if support_mask.sum() == 0:
        img_data = np.zeros_like(img_data)
    else:
        vals = img_data[support_mask]
        low, high = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
        if high - low < 1e-8:
            img_data = np.zeros_like(img_data)
        else:
            img_data = np.clip(img_data, low, high)
            img_data = (img_data - low) / (high - low)
            img_data[~support_mask] = 0.0

    # Affine-based world coords normalized to [-1, 1]
    h, w, d = img_data.shape
    coords_mtx = get_coords_from_affine(h, w, d, affine)
    flat = coords_mtx.reshape(-1, 3)
    c_min = flat.min(dim=0).values
    c_max = flat.max(dim=0).values
    coords_mtx = 2.0 * (coords_mtx - c_min) / (c_max - c_min) - 1.0

    # Subsampled tensors for INR fitting (matches training)
    img_sub = img_data[::skip_pixels, ::skip_pixels, ::skip_pixels]
    coords_sub = coords_mtx[::skip_pixels, ::skip_pixels, ::skip_pixels]
    img_tensor = torch.from_numpy(img_sub.reshape(-1, 1)).float().cuda()[None, ...]
    coords_sub_tensor = coords_sub.reshape(-1, 3).float().cuda()[None, ...]

    # Full-resolution coords for the classifier forward pass
    coords_full_tensor = coords_mtx.reshape(-1, 3).float().cuda()[None, ...]

    return img_tensor, coords_sub_tensor, coords_full_tensor, original_shape, seg_data


def inference_on_image(
    image_tensor,
    coords_sub_tensor,
    coords_full_tensor,
    original_shape,
    best_inr_weights,
    classifier_model,
    test_run_steps: int,
) -> np.ndarray:
    inr_model = models.INR(**inr_config).float().cuda()
    inr_model.load_state_dict(
        {
            k.replace("inr.", ""): v.clone().detach()
            for k, v in deepcopy(best_inr_weights).items()
        }
    )
    inr_model.compile()
    inr_model.fit(
        coords_sub_tensor, image_tensor, epochs=test_run_steps, disable_tqdm=True
    )

    with torch.no_grad():
        _, img_features = inr_model.forward_w_features(coords_full_tensor)
        classifier_input = img_features[-2]
        if NORMALIZE_FEATURES:
            classifier_input = nn.functional.normalize(classifier_input, dim=-1)
        output = classifier_model(classifier_input)
        pred_probs = nn.functional.softmax(output.unsqueeze(0).unsqueeze(0), dim=-1)
        predictions = pred_probs.argmax(dim=-1).reshape(original_shape)

    del inr_model
    torch.cuda.empty_cache()

    return predictions.detach().cpu().numpy().astype(np.uint8)


def compute_dice(predictions_np: np.ndarray, seg_data: np.ndarray) -> float:
    pred_onehot = torch.nn.functional.one_hot(
        torch.from_numpy(predictions_np.astype(np.int64)),
        num_classes=NUM_CLASSES_AND_ONE,
    ).float()
    gt_onehot = torch.nn.functional.one_hot(
        torch.from_numpy(seg_data).clamp(0, NUM_CLASSES),
        num_classes=NUM_CLASSES_AND_ONE,
    ).float()
    dice = metrics.multiclass_dice_score_3d(
        pred_onehot.cuda(), gt_onehot.cuda(), num_classes=NUM_CLASSES_AND_ONE
    )
    return float(dice.item())


def main():
    parser = argparse.ArgumentParser(description="MetaSeg 3D dataset evaluation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON split config"
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True, help="Root directory for NIfTI files"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate (default: test)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="INR fitting steps per subject (default: 300)",
    )
    parser.add_argument(
        "--skip-pixels",
        type=int,
        default=2,
        help="Spatial subsampling factor for INR fitting (default: 2)",
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default=DEFAULT_WEIGHTS_FILE,
        help="Path to Step 1 meta-learned weights .pth",
    )
    parser.add_argument(
        "--classifier-weights-file",
        type=str,
        default=DEFAULT_CLASSIFIER_WEIGHTS_FILE,
        help="Path to fine-tuned classifier weights .pth",
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="Limit evaluation to first N subjects (default: all)",
    )
    args = parser.parse_args()

    # Load weights
    print(f"Loading meta-learned weights from: {args.weights_file}")
    weights_from_metalearning = torch.load(args.weights_file)
    best_inr_weights = weights_from_metalearning["best_inr_weights"]

    inr_seg_model = (
        models.SirenSegINR(
            inr_type="siren",
            inr_config=inr_config,
            segmentation_config=segmentation_config,
            normalize_features=NORMALIZE_FEATURES,
        )
        .float()
        .cuda()
    )

    classifier_model = deepcopy(inr_seg_model.segmentation_head)
    if osp.isfile(args.classifier_weights_file):
        print(f"Loading classifier weights from: {args.classifier_weights_file}")
        clf_weights = torch.load(args.classifier_weights_file)
        try:
            classifier_model.load_state_dict(clf_weights["final_clf_weights"])
        except KeyError:
            clf_weights_fixed = {
                k.replace("segmentation_head.segmentation_head", "segmentation_head"): v
                for k, v in clf_weights.items()
            }
            classifier_model.load_state_dict(clf_weights_fixed)
    else:
        print("Classifier weights file not found, using best_classifier_weights from Step 1")
        best_clf_weights = weights_from_metalearning["best_classifier_weights"]
        best_clf_weights_fixed = {
            k.replace("segmentation_head.segmentation_head", "segmentation_head"): v
            for k, v in best_clf_weights.items()
        }
        classifier_model.load_state_dict(best_clf_weights_fixed)
    classifier_model.eval()

    # Load split
    with open(args.config, "r") as f:
        split_data = json.load(f)
    subjects = split_data[args.mode]
    if args.num_subjects is not None:
        subjects = subjects[: args.num_subjects]

    print(f"\nEvaluating {len(subjects)} subjects from '{args.mode}' split")
    print(f"INR steps: {args.steps}  |  skip_pixels: {args.skip_pixels}")
    print("=" * 80)

    dice_scores = []
    for ix, subject in enumerate(tqdm(subjects, desc="Subjects")):
        img_path = osp.join(args.dataset_dir, subject["img"])
        seg_path = osp.join(args.dataset_dir, subject[f"seg{NUM_CLASSES}"])

        img_tensor, coords_sub, coords_full, original_shape, seg_data = load_and_prepare(
            img_path, seg_path, args.skip_pixels
        )

        predictions = inference_on_image(
            img_tensor,
            coords_sub,
            coords_full,
            original_shape,
            best_inr_weights,
            classifier_model,
            args.steps,
        )

        dice = compute_dice(predictions, seg_data)
        dice_scores.append(dice)
        tqdm.write(
            f"  [{ix + 1:3d}/{len(subjects)}] {osp.basename(img_path)}  Dice={dice:.5f}"
        )

    print("=" * 80)
    print(
        f"Average Segmentation Dice = {np.mean(dice_scores):.5f} +/- {np.std(dice_scores):.5f}"
    )
    print(f"Min: {np.min(dice_scores):.5f}  Max: {np.max(dice_scores):.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
