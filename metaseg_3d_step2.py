import json
import os
import os.path as osp
from copy import deepcopy

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

import dataloaders
from modules import loss_functions, models

torch.manual_seed(422)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset_dir = "/scratch/thesis-saverio/data/HCP"
config_file = "config/HCP_split.json"

weights_file = "/scratch/thesis-saverio/dumps/metaseg_3d_step1-normalized-robust/weights3d_num_classes_4_IS_2.pth"
CLASSIFIER_WEIGHTS_DIR = "/scratch/thesis-saverio/dumps/weights_3d"

# --- Subject subset & disk-caching ----------------------------------------
# Phase 1 (SAVE_FEATURE_VECS=True):  fit INR per subject, save features to disk.
# Phase 2 (SAVE_FEATURE_VECS=False): load cached features and train classifier.
# Re-extraction is skipped automatically if the .pth file already exists,
# so it's safe to re-run with SAVE_FEATURE_VECS=True after a partial run.
N_TRAIN_SUBJECTS = 200  # subset of the 657 HCP train subjects
N_VAL_SUBJECTS = None  # None = use the full val split (82 subjects)
SAVE_FEATURE_VECS = True  # set False after features are saved to jump to training
SAVE_PATHS = "/scratch/thesis-saverio/dumps/intermediate_vectors"
# --------------------------------------------------------------------------

files_data = json.load(open(config_file, "r"))
train_files = files_data["train"]
val_files = files_data["val"]
test_files = files_data["test"]

# check overlap for any samples.
print(len(train_files), len(val_files), len(test_files))
set_train = set([x["img"] for x in train_files])
set_val = set([x["img"] for x in val_files])
set_test = set([x["img"] for x in test_files])

if (
    len(set_train.intersection(set_val)) > 0
    or len(set_train.intersection(set_test)) > 0
    or len(set_val.intersection(set_test)) > 0
):
    print("WARNING: OVERLAPPING DATA SPLITS")
else:
    print("No overlap in data splits. GOOD TO GO!!!!!!!!!!!")

INNER_STEPS = 2
RANDOM_AUGMENT = False
TEST_RUN_STEPS = VAL_STEPS = (
    300  # for full res. Hyperparameters. you can use values like T=100, 300. For 3D you need to let it render for more number of iterations unlike simple 2D images.
)
SKIP_PIXELS = 2
VAL_META_STEPS = 100
OUTER_LOOP_ITERATIONS = 5000  # 5000
NUM_CLASSES = 4
NUM_CLASSES_AND_ONE = NUM_CLASSES + 1

NORMALIZE_FEATURES = False

nonlin = "siren"
inr_config = {
    "in_features": 3,
    "out_features": 1,
    "hidden_features": 256,
    "hidden_layers": 4,
}  # "first_omega_0":200.0, 'hidden_omega_0':200.0}
segmentation_config = {
    "hidden_features": [
        256,
    ],  # [128, 64],
    "output_features": NUM_CLASSES_AND_ONE,
}

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

weights_from_metalearning = torch.load(weights_file)
inr_seg_model_wts = weights_from_metalearning["inr_seg_model"]
best_inr_weights = weights_from_metalearning["best_inr_weights"]
best_classifier_weights = weights_from_metalearning["best_classifier_weights"]


def extract_and_save_features(
    dl, best_inr_weights, inr_config, steps, save_dir, prefix, desc
):
    """Fit INR per subject, save penultimate features to disk. Skips existing files."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, data in enumerate(tqdm(dl, desc=desc)):
        save_path = osp.join(save_dir, f"{prefix}_{idx}.pth")
        if osp.isfile(save_path):
            tqdm.write(f"  Skipping {prefix}_{idx} (already saved)")
            continue

        img = data["img"].float().cuda()
        seg = data["seg"].float().cuda()
        coords = data["coords"].float().cuda()

        inr_model = models.INR(**inr_config).float().cuda()
        inr_model.load_state_dict(
            {
                k.replace("inr.", ""): v.clone().detach()
                for k, v in deepcopy(best_inr_weights).items()
            }
        )
        inr_model.compile()
        inr_model.fit(coords, img, epochs=steps, disable_tqdm=True)
        _, img_features = inr_model.forward_w_features(coords)

        torch.save(
            {
                "features": img_features[-2].squeeze(0).detach().cpu(),  # (N, F)
                "seg": seg.squeeze(0).detach().cpu(),  # (N, C)
            },
            save_path,
        )

        del inr_model
        torch.cuda.empty_cache()


# ---- Phase 1: extract features and save to disk --------------------------
if SAVE_FEATURE_VECS:
    train_cfg = {"augment": RANDOM_AUGMENT, "normalize": True}
    if N_TRAIN_SUBJECTS is not None:
        train_cfg["N_samples"] = N_TRAIN_SUBJECTS

    val_cfg = {"augment": RANDOM_AUGMENT, "normalize": True}
    if N_VAL_SUBJECTS is not None:
        val_cfg["N_samples"] = N_VAL_SUBJECTS

    train_ds = dataloaders.TorchMRI3D_Dataloader(
        json_file=config_file,
        mode="train",
        config=train_cfg,
        num_classes=NUM_CLASSES,
        skip_pixels=SKIP_PIXELS,
        dataset_dir=dataset_dir,
    )
    val_ds = dataloaders.TorchMRI3D_Dataloader(
        json_file=config_file,
        mode="val",
        config=val_cfg,
        num_classes=NUM_CLASSES,
        skip_pixels=SKIP_PIXELS,
        dataset_dir=dataset_dir,
    )
    print(f"Subjects for extraction — train: {len(train_ds)}, val: {len(val_ds)}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    print("Extracting and saving train features...")
    extract_and_save_features(
        train_dl,
        best_inr_weights,
        inr_config,
        TEST_RUN_STEPS,
        osp.join(SAVE_PATHS, "train"),
        "train",
        desc="Train",
    )
    print("Extracting and saving val features...")
    extract_and_save_features(
        val_dl,
        best_inr_weights,
        inr_config,
        TEST_RUN_STEPS,
        osp.join(SAVE_PATHS, "val"),
        "val",
        desc="Val",
    )


# ---- Phase 2: train classifier from cached features ----------------------
CLASSIFIER_FINETUNE_EPOCHS = 100_000
EARLY_STOPPING_PATIENCE = 20  # validation checks (= PATIENCE * VAL_META_STEPS epochs)
# batch_size=1 works for any N; increase only if all subjects share the same N
SUBJECT_BATCH_SIZE = 1

train_feat_dl = torch.utils.data.DataLoader(
    dataloaders.CLFFeature(SAVE_PATHS, mode="train"),
    batch_size=SUBJECT_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
val_feat_dl = torch.utils.data.DataLoader(
    dataloaders.CLFFeature(SAVE_PATHS, mode="val"),
    batch_size=SUBJECT_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

#### IMPORTANT: this step may have key mismatch based on how the model was saved. simply use the str.replace() function to match your saved keys to the model's named parameters

classifier_model = deepcopy(inr_seg_model.segmentation_head)

classifier_weights = deepcopy(best_classifier_weights)
try:
    classifier_model.load_state_dict(classifier_weights["final_clf_weights"])
except KeyError:
    classifier_weights = deepcopy(
        {
            k.replace(
                "segmentation_head.segmentation_head", "segmentation_head"
            ): v.clone().detach()
            for k, v in best_classifier_weights.items()
        }
    )
    classifier_model.load_state_dict(classifier_weights)

LEARNING_RATE = 5e-5
FOCAL_LOSS_GAMMA = 3.0
ZERO_WT = 0.1

n_train_tag = N_TRAIN_SUBJECTS if N_TRAIN_SUBJECTS is not None else "all"
EXPERIMENT_NAME = (
    f"gamma_{FOCAL_LOSS_GAMMA}_INR_300it_skip_pixels_{SKIP_PIXELS}_subset{n_train_tag}"
)

classifier_opt = torch.optim.Adam(classifier_model.parameters(), lr=LEARNING_RATE)
print(
    classifier_model,
    list(classifier_weights.keys()),
    list(classifier_model.state_dict().keys()),
)

finetune_classifier_loss_fn = loss_functions.LossFunction(
    {"focal_loss": loss_functions.FocalSemanticLoss(gamma=FOCAL_LOSS_GAMMA)}
)

# torch.compile fuses elementwise ops in the small MLP — free ~10-20% speedup.
classifier_model = torch.compile(classifier_model)

scaler = torch.amp.GradScaler("cuda")

final_classifier_weights = None
best_val_score = 1e7
epochs_without_improvement = 0

pbar_epochs = tqdm(range(CLASSIFIER_FINETUNE_EPOCHS), position=0)
for epoch in pbar_epochs:
    avg_loss_per_set = 0.0
    classifier_model.train()

    for data_batch in train_feat_dl:
        feat_batch = data_batch["features"].float().cuda(non_blocking=True)  # (B, N, F)
        seg_batch = data_batch["seg"].float().cuda(non_blocking=True)  # (B, N, C)
        if NORMALIZE_FEATURES:
            feat_batch = nn.functional.normalize(feat_batch, dim=-1)

        B, N, F_ = feat_batch.shape
        classifier_opt.zero_grad()
        with torch.amp.autocast("cuda"):
            # flatten subjects into the batch dim for the MLP, then restore
            clf_out = classifier_model(feat_batch.view(B * N, F_)).view(B, N, -1)
            loss, _ = finetune_classifier_loss_fn(
                {"output": {"segmentation_output": clf_out}, "seg": seg_batch}
            )
        scaler.scale(loss).backward()
        scaler.step(classifier_opt)
        scaler.update()
        avg_loss_per_set += float(loss.item())

    avg_loss_per_set /= len(train_feat_dl)
    pbar_epochs.set_description(
        f"Loss (clf): {avg_loss_per_set:.5f}. Best Val Loss(clf): {best_val_score:.5f}"
    )
    pbar_epochs.refresh()

    if epoch % VAL_META_STEPS == 0 and epoch > 0:
        classifier_model.eval()
        with torch.no_grad():
            avg_val_loss = 0.0
            for data_batch in val_feat_dl:
                feat_batch = data_batch["features"].float().cuda(non_blocking=True)
                seg_batch = data_batch["seg"].float().cuda(non_blocking=True)
                if NORMALIZE_FEATURES:
                    feat_batch = nn.functional.normalize(feat_batch, dim=-1)
                B, N, F_ = feat_batch.shape
                with torch.amp.autocast("cuda"):
                    clf_out = classifier_model(feat_batch.view(B * N, F_)).view(
                        B, N, -1
                    )
                    val_loss, _ = finetune_classifier_loss_fn(
                        {"output": {"segmentation_output": clf_out}, "seg": seg_batch}
                    )
                avg_val_loss += float(val_loss.item())

            avg_val_loss /= len(val_feat_dl)
            pbar_epochs.set_description(
                f"Loss (clf): {avg_loss_per_set:.5f} Val Loss (clf): {avg_val_loss:.5f}"
            )
            pbar_epochs.refresh()

            if avg_val_loss < best_val_score:
                best_val_score = avg_val_loss
                epochs_without_improvement = 0
                # torch.compile wraps the model; unwrap for state_dict
                raw_state = (
                    classifier_model._orig_mod.state_dict()
                    if hasattr(classifier_model, "_orig_mod")
                    else classifier_model.state_dict()
                )
                final_classifier_weights = deepcopy(raw_state)
                tqdm.write(f"updated best val score to {best_val_score}")
                os.makedirs(CLASSIFIER_WEIGHTS_DIR, exist_ok=True)
                torch.save(
                    {
                        "final_clf_weights": final_classifier_weights,
                        "focal_loss_gamma": FOCAL_LOSS_GAMMA,
                        "zero_wt": ZERO_WT,
                    },
                    osp.join(
                        CLASSIFIER_WEIGHTS_DIR,
                        f"classifierfinal_weights_LR_{LEARNING_RATE}_exp_{EXPERIMENT_NAME}.pth",
                    ),
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    tqdm.write(
                        f"Early stopping at epoch {epoch}: no improvement for "
                        f"{EARLY_STOPPING_PATIENCE * VAL_META_STEPS} epochs."
                    )
                    break
