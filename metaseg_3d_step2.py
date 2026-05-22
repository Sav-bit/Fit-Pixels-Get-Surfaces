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

weights_file = "/scratch/thesis-saverio/dumps/metaseg_3d_step1-normalized/weights3d_num_classes_4_IS_2.pth"
CLASSIFIER_WEIGHTS_DIR = "/scratch/thesis-saverio/dumps/weights_3d"

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

train_ds = dataloaders.TorchMRI3D_Dataloader(
    json_file=config_file,
    mode="train",
    config={"augment": RANDOM_AUGMENT},
    num_classes=NUM_CLASSES,
    skip_pixels=SKIP_PIXELS,
    dataset_dir=dataset_dir,
)
val_ds = dataloaders.TorchMRI3D_Dataloader(
    json_file=config_file,
    mode="val",
    config={"augment": RANDOM_AUGMENT, "N_samples": 10},
    num_classes=NUM_CLASSES,
    skip_pixels=SKIP_PIXELS,
    dataset_dir=dataset_dir,
)
print(len(train_ds), len(val_ds))

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)


def extract_features(dl, best_inr_weights, inr_config, steps, desc):
    features = []
    for data in tqdm(dl, desc=desc):
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

        # store on CPU to free GPU memory between subjects
        features.append(
            {
                "seg": seg.detach().cpu(),        # (1, N, C)
                "features": img_features[-2].detach().cpu(),  # (1, N, F)
            }
        )
    return features


print("Extracting train features...")
train_features = extract_features(
    train_dl, best_inr_weights, inr_config, TEST_RUN_STEPS, desc="Train"
)
print("Extracting val features...")
val_features = extract_features(
    val_dl, best_inr_weights, inr_config, TEST_RUN_STEPS, desc="Val"
)


CLASSIFIER_FINETUNE_EPOCHS = 100_000
EARLY_STOPPING_PATIENCE = 20  # validation checks (= PATIENCE * VAL_META_STEPS epochs)
SUBJECT_BATCH_SIZE = 8  # subjects per gradient step; tune up if VRAM allows

# Pre-stack all features into contiguous tensors for batched GPU loading.
# All HCP subjects share the same N (fixed crop + fixed skip_pixels), so stacking is safe.
# del the original list afterward to free the duplicated CPU memory.
train_feat_tensor = torch.stack([d["features"].squeeze(0) for d in train_features])  # (S, N, F)
train_seg_tensor  = torch.stack([d["seg"].squeeze(0)      for d in train_features])  # (S, N, C)
del train_features
val_feat_tensor = torch.stack([d["features"].squeeze(0) for d in val_features])
val_seg_tensor  = torch.stack([d["seg"].squeeze(0)      for d in val_features])
del val_features


class _FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feats, segs):
        self.feats, self.segs = feats, segs

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i], self.segs[i]


train_feat_dl = torch.utils.data.DataLoader(
    _FeatureDataset(train_feat_tensor, train_seg_tensor),
    batch_size=SUBJECT_BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0,
)
val_feat_dl = torch.utils.data.DataLoader(
    _FeatureDataset(val_feat_tensor, val_seg_tensor),
    batch_size=SUBJECT_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0,
)

#### IMPORTANT: this step may have key mismatch based on how the model was saved. simply use the str.replace() function to match your saved keys to the model's named parameters

classifier_model = deepcopy(inr_seg_model.segmentation_head)

classifier_weights = deepcopy(best_classifier_weights)
try:
    classifier_model.load_state_dict(
        classifier_weights["final_clf_weights"]
    )  # check key, if final_clf_weights key does not exist, then just load classifier_weights as shown above.
except:
    classifier_weights = deepcopy(
        {
            k.replace(
                "segmentation_head.segmentation_head", "segmentation_head"
            ): v.clone().detach()
            for k, v in best_classifier_weights.items()
        }
    )

LEARNING_RATE = 5e-5
FOCAL_LOSS_GAMMA = 3.0
ZERO_WT = 0.1

EXPERIMENT_NAME = (
    f"gamma_{FOCAL_LOSS_GAMMA}_INR_300it_skip_pixels_{SKIP_PIXELS}_batched"
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

    for feat_batch, seg_batch in train_feat_dl:
        feat_batch = feat_batch.float().cuda(non_blocking=True)  # (B, N, F)
        seg_batch  = seg_batch.float().cuda(non_blocking=True)   # (B, N, C)
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
            for feat_batch, seg_batch in val_feat_dl:
                feat_batch = feat_batch.float().cuda(non_blocking=True)
                seg_batch  = seg_batch.float().cuda(non_blocking=True)
                if NORMALIZE_FEATURES:
                    feat_batch = nn.functional.normalize(feat_batch, dim=-1)
                B, N, F_ = feat_batch.shape
                with torch.amp.autocast("cuda"):
                    clf_out = classifier_model(feat_batch.view(B * N, F_)).view(B, N, -1)
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
                raw_state = classifier_model._orig_mod.state_dict() if hasattr(classifier_model, "_orig_mod") else classifier_model.state_dict()
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
