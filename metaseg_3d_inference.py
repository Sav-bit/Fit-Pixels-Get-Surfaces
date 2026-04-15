import os
import os.path as osp
import sys
import argparse
from copy import deepcopy

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import alpine

from modules import models, metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(422)

# ===== Configuration =====
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
weights_file = osp.join(SCRIPT_DIR, "notebooks", "3d", "dumps", "weights3d_num_classes_4_IS_2.pth")
classifier_weights_file = osp.join(
    SCRIPT_DIR, "notebooks", "3d", "dumps", "weights_3d",
    "classifierfinal_weights_LR_5e-05_exp_gamma_3.0_INR_300it_skip_pixels_2_continue.pth"
)

TEST_RUN_STEPS = 300
SKIP_PIXELS = 2
NUM_CLASSES = 4
NUM_CLASSES_AND_ONE = NUM_CLASSES + 1
RES = (160, 160, 200)
VAL_RES = [160//SKIP_PIXELS, 160//SKIP_PIXELS, 200//SKIP_PIXELS]
NORMALIZE_FEATURES = False

# ===== Model Setup =====
inr_config = {"in_features": 3, "out_features": 1, "hidden_features": 256, "hidden_layers": 4}
segmentation_config = {'hidden_features': [256,], 'output_features': NUM_CLASSES_AND_ONE}

# Load pre-trained INR-Seg model
inr_seg_model = models.SirenSegINR(
    inr_type='siren',
    inr_config=inr_config,
    segmentation_config=segmentation_config,
    normalize_features=NORMALIZE_FEATURES,
).float().cuda()

# Load meta-learned weights
print(f"Loading meta-learned weights from: {weights_file}")
weights_from_metalearning = torch.load(weights_file)
best_inr_weights = weights_from_metalearning['best_inr_weights']

# Load classifier weights
print(f"Loading classifier weights from: {classifier_weights_file}")
classifier_weights = torch.load(classifier_weights_file)
classifier_model = deepcopy(inr_seg_model.segmentation_head)
try:
    classifier_model.load_state_dict(classifier_weights['final_clf_weights'])
except KeyError:
    classifier_weights_fixed = {
        k.replace("segmentation_head.segmentation_head", "segmentation_head"): v 
        for k, v in classifier_weights.items()
    }
    classifier_model.load_state_dict(classifier_weights_fixed)
classifier_model.eval()


def inference_on_image(image_tensor, coords_tensor):
    """Run inference on a single image."""
    # Create and load INR model with meta-learned initialization
    inr_model = models.INR(**inr_config).float().cuda()
    inr_model.load_state_dict({
        k.replace("inr.", ""): v.clone().detach() 
        for k, v in deepcopy(best_inr_weights).items()
    })
    inr_model.compile()
    
    # Fit INR on the image
    print(f"  Fitting INR model ({TEST_RUN_STEPS} steps)...")
    inr_model.fit(coords_tensor, image_tensor, epochs=TEST_RUN_STEPS, disable_tqdm=True)
    
    # Extract features
    _, img_features = inr_model.forward_w_features(coords_tensor)
    classifier_input = img_features[-2]
    
    # Normalize if needed
    if NORMALIZE_FEATURES:
        classifier_input = nn.functional.normalize(classifier_input, dim=-1)
    
    # Run classifier
    with torch.no_grad():
        output = classifier_model(classifier_input)
        pred_probs = nn.functional.softmax(output.unsqueeze(0).unsqueeze(0), dim=-1)
        predictions = pred_probs.argmax(dim=-1).reshape(VAL_RES)
    
    # Clean up
    del inr_model
    torch.cuda.empty_cache()
    
    return predictions.detach().cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="MetaSeg 3D inference on a single NIfTI image")
    parser.add_argument("input_image", type=str, help="Path to input NIfTI image (.nii or .nii.gz)")
    parser.add_argument("output_seg", type=str, help="Path to save output segmentation NIfTI file")
    parser.add_argument("--steps", type=int, default=TEST_RUN_STEPS, help=f"INR fitting steps (default: {TEST_RUN_STEPS})")
    
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
    
    print(f"  Original shape: {original_shape}")
    
    # Apply same cropping as dataloader: [:, 16:-16, 12:-12]
    img_data = img_data[:, 16:-16, 12:-12]
    cropped_shape = img_data.shape
    print(f"  After cropping: {cropped_shape}")
    
    # Resize to model resolution if needed
    if img_data.shape != tuple(RES):
        print(f"  Resizing from {img_data.shape} to {tuple(RES)}")
        from scipy import ndimage
        zoom_factors = [r / s for r, s in zip(RES, img_data.shape)]
        img_data = ndimage.zoom(img_data, zoom_factors, order=1)
    
    # Normalize
    img_min, img_max = img_data.min(), img_data.max()
    if img_max > img_min:
        img_data = (img_data - img_min) / (img_max - img_min)
    
    # Prepare tensors - subsample by skip_pixels (same as dataloader does)
    img_data_subsampled = img_data[::SKIP_PIXELS, ::SKIP_PIXELS, ::SKIP_PIXELS]
    
    # Reshape to (N, 1) for image
    img_tensor = torch.from_numpy(img_data_subsampled.reshape(-1, 1)).float().cuda()[None, ...]
    
    # Generate coordinates using same method as dataloader
    h, w, d = RES
    xx = torch.linspace(-1, 1, h)
    yy = torch.linspace(-1, 1, w)
    zz = torch.linspace(-1, 1, d)
    coords_full = torch.meshgrid(xx, yy, zz, indexing='ij')
    coords_full = torch.stack(coords_full, dim=-1)  # shape: (h, w, d, 3)
    
    # Subsample coordinates
    coords_subsampled = coords_full[::SKIP_PIXELS, ::SKIP_PIXELS, ::SKIP_PIXELS, ...]
    coords_tensor = coords_subsampled.reshape(-1, 3).float().cuda()[None, ...]  # reshape to (1, N, 3)
    
    # Run inference
    print("\nRunning inference...")
    predictions = inference_on_image(img_tensor, coords_tensor)

    print(f"  Prediction shape (subsampled): {predictions.shape}")
    print(f"  Unique classes: {np.unique(predictions)}")

    # Upsample predictions back to model resolution (160, 160, 200)
    from scipy import ndimage
    zoom_factors = [r / s for r, s in zip(RES, predictions.shape)]
    predictions_upsampled = ndimage.zoom(predictions, zoom_factors, order=0)  # order=0 for nearest neighbor
    print(f"  Prediction shape (model res): {predictions_upsampled.shape}")

    # Resize predictions back to cropped image resolution
    zoom_factors_to_cropped = [c / r for c, r in zip(cropped_shape, RES)]
    predictions_cropped = ndimage.zoom(predictions_upsampled, zoom_factors_to_cropped, order=0)
    print(f"  Prediction shape (cropped space): {predictions_cropped.shape}")

    # Undo crop to match original input shape exactly
    predictions_original = np.zeros(original_shape, dtype=np.uint8)
    predictions_original[:, 16:-16, 12:-12] = predictions_cropped.astype(np.uint8)
    print(f"  Prediction shape (original): {predictions_original.shape}")

    # Save output
    print(f"\nSaving to: {args.output_seg}")
    out_nib = nib.Nifti1Image(predictions_original.astype(np.uint8), affine)
    nib.save(out_nib, args.output_seg)
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()