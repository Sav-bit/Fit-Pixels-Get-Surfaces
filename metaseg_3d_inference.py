import argparse
import os
import os.path as osp
import sys
from copy import deepcopy

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from modules import models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


torch.manual_seed(422)

# ===== Configuration =====
weights_file = "/scratch/thesis-saverio/dumps/metaseg_3d_step1-normalized-robust/weights3d_num_classes_4_IS_2.pth"
# classifier_weights_file = "/scratch/thesis-saverio/dumps/weights_3d/classifierfinal_weights_LR_5e-05_exp_gamma_3.0_INR_300it_skip_pixels_2_continue.pth"
classifier_weights_file = "/scratch/thesis-saverio/dumps/weights_3d/classifierfinal_weights_LR_5e-05_exp_gamma_3.0_INR_300it_skip_pixels_2_subset200.pth"

TEST_RUN_STEPS = 300
SKIP_PIXELS = 2
NUM_CLASSES = 4
NUM_CLASSES_AND_ONE = NUM_CLASSES + 1
NORMALIZE_FEATURES = False

# ===== Model Setup =====
inr_config = {
    "in_features": 3,
    "out_features": 1,
    "hidden_features": 256,
    "hidden_layers": 4,
}
segmentation_config = {
    "hidden_features": [
        256,
    ],
    "output_features": NUM_CLASSES_AND_ONE,
}

# Load pre-trained INR-Seg model
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

# Load meta-learned weights
print(f"Loading meta-learned weights from: {weights_file}")
weights_from_metalearning = torch.load(weights_file)
best_inr_weights = weights_from_metalearning["best_inr_weights"]

# Load classifier weights
classifier_model = deepcopy(inr_seg_model.segmentation_head)
if osp.isfile(classifier_weights_file):
    print(f"Loading classifier weights from: {classifier_weights_file}")
    classifier_weights = torch.load(classifier_weights_file)
    try:
        classifier_model.load_state_dict(classifier_weights["final_clf_weights"])
    except KeyError:
        classifier_weights_fixed = {
            k.replace("segmentation_head.segmentation_head", "segmentation_head"): v
            for k, v in classifier_weights.items()
        }
        classifier_model.load_state_dict(classifier_weights_fixed)
else:
    print(
        "Classifier weights file not found, using best_classifier_weights from Step 1 checkpoint"
    )
    best_clf_weights = weights_from_metalearning["best_classifier_weights"]
    best_clf_weights_fixed = {
        k.replace("segmentation_head.segmentation_head", "segmentation_head"): v
        for k, v in best_clf_weights.items()
    }
    classifier_model.load_state_dict(best_clf_weights_fixed)
classifier_model.eval()


def inference_on_image(
    image_tensor, coords_sub_tensor, coords_full_tensor, original_shape
):
    """Run inference on a single image."""
    # Create and load INR model with meta-learned initialization
    inr_model = models.INR(**inr_config).float().cuda()
    inr_model.load_state_dict(
        {
            k.replace("inr.", ""): v.clone().detach()
            for k, v in deepcopy(best_inr_weights).items()
        }
    )
    inr_model.compile()

    # Fit INR on subsampled coords (matches training, low VRAM)
    print(f"  Fitting INR model ({TEST_RUN_STEPS} steps)...")
    inr_model.fit(
        coords_sub_tensor, image_tensor, epochs=TEST_RUN_STEPS, disable_tqdm=True
    )

    print("  Done fitting INR model.")
    print("  Extracting features and running classifier...")

    # Extract features at full resolution (no gradients needed)
    with torch.no_grad():
        _, img_features = inr_model.forward_w_features(coords_full_tensor)
        classifier_input = img_features[-2]

        if NORMALIZE_FEATURES:
            classifier_input = nn.functional.normalize(classifier_input, dim=-1)

        output = classifier_model(classifier_input)
        pred_probs = nn.functional.softmax(output.unsqueeze(0).unsqueeze(0), dim=-1)
        predictions = pred_probs.argmax(dim=-1).reshape(original_shape)

    # Clean up
    del inr_model
    torch.cuda.empty_cache()

    return predictions.detach().cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="MetaSeg 3D inference on a single NIfTI image"
    )
    parser.add_argument(
        "input_image", type=str, help="Path to input NIfTI image (.nii or .nii.gz)"
    )
    parser.add_argument(
        "output_seg", type=str, help="Path to save output segmentation NIfTI file"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TEST_RUN_STEPS,
        help=f"INR fitting steps (default: {TEST_RUN_STEPS})",
    )

    args = parser.parse_args()

    if not osp.isfile(args.input_image):
        print(f"ERROR: Input file not found: {args.input_image}")
        sys.exit(1)

    print("=" * 80)
    print("MetaSeg 3D Inference")
    print("=" * 80)
    print(f"Input:  {args.input_image}")
    print(f"Output: {args.output_seg}")

    # Load input NIfTI
    print("\nLoading image...")
    img_nib = nib.load(args.input_image)
    img_data = img_nib.get_fdata().astype(np.float32)
    affine = img_nib.affine
    original_shape = img_data.shape

    print(f"  Shape: {original_shape}")

    # Normalize (matches dataloader: percentile clip on non-zero voxels, zero mask preserved)
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

    # Generate affine-based world coordinates and normalize to [-1, 1] (matches dataloader)
    h, w, d = img_data.shape
    coords_mtx = get_coords_from_affine(h, w, d, affine)
    flat = coords_mtx.reshape(-1, 3)
    c_min = flat.min(dim=0).values
    c_max = flat.max(dim=0).values
    coords_mtx = 2.0 * (coords_mtx - c_min) / (c_max - c_min) - 1.0

    # Subsampled tensors for INR fitting (matches training, low VRAM)
    img_sub = img_data[::SKIP_PIXELS, ::SKIP_PIXELS, ::SKIP_PIXELS]
    coords_sub = coords_mtx[::SKIP_PIXELS, ::SKIP_PIXELS, ::SKIP_PIXELS]
    img_tensor = torch.from_numpy(img_sub.reshape(-1, 1)).float().cuda()[None, ...]
    coords_sub_tensor = coords_sub.reshape(-1, 3).float().cuda()[None, ...]

    # Full-resolution coords for the classifier forward pass
    coords_full_tensor = coords_mtx.reshape(-1, 3).float().cuda()[None, ...]

    # Run inference
    print("\nRunning inference...")
    predictions = inference_on_image(
        img_tensor, coords_sub_tensor, coords_full_tensor, original_shape
    )

    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Unique classes: {np.unique(predictions)}")

    # Save output
    print(f"\nSaving to: {args.output_seg}")
    out_nib = nib.Nifti1Image(predictions.astype(np.uint8), affine)
    nib.save(out_nib, args.output_seg)

    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
