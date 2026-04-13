import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import glob
import json
import os.path as osp
import sys
from copy import deepcopy

import alpine
import dataloaders
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
torch.manual_seed(422)
import numpy as np
from tqdm.autonotebook import tqdm

from modules import loss_functions, metrics, models, utils, vis
from modules.learner import INRMetaLearner

dataset_dir = "/projects/thesis-saverio/data"
config_file = "./config/oasis_splits_3d.json"


SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
weights_file = osp.join(SCRIPT_DIR, "dumps", "weights3d_num_classes_4_IS_2.pth")
CLASSIFIER_WEIGHTS_DIR = osp.join(SCRIPT_DIR, "dumps", "weights_3d")

SAVE_FEATURE_VECS = False
USE_SAVED_FEATURE_VECS_TO_TRAIN_CLF = not SAVE_FEATURE_VECS

SAVE_PATHS = "/projects/thesis-saverio/dumps/intermediate_vectors"

files_data = json.load(open(config_file, 'r'))
train_files = files_data['train']
val_files = files_data['val']
test_files = files_data['test']



# check overlap for any samples.
print(len(train_files), len(val_files), len(test_files))
set_train = set([x['img'] for x in train_files])
set_val = set([x['img'] for x in val_files])
set_test = set([x['img'] for x in test_files])

if len(set_train.intersection(set_val)) > 0 or  len(set_train.intersection(set_test)) > 0 or  len(set_val.intersection(set_test)) > 0:
    print("WARNING: OVERLAPPING DATA SPLITS")
else:
    print("No overlap in data splits. GOOD TO GO!!!!!!!!!!!")

INNER_STEPS = 2
RANDOM_AUGMENT = False
TEST_RUN_STEPS = VAL_STEPS  = 300 # for full res. Hyperparameters. you can use values like T=100, 300. For 3D you need to let it render for more number of iterations unlike simple 2D images.
SKIP_PIXELS = 2
VAL_META_STEPS = 100
OUTER_LOOP_ITERATIONS =  5000  # 5000
NUM_CLASSES = 4
NUM_CLASSES_AND_ONE = NUM_CLASSES + 1
# RES = (160, 192, 224)

RES = (160,160,200)
VAL_RES = RES = [160//SKIP_PIXELS, 160//SKIP_PIXELS, 200//SKIP_PIXELS]

NORMALIZE_FEATURES = False


nonlin = 'siren'
inr_config = {"in_features":3, "out_features": 1, "hidden_features": 256, "hidden_layers": 4, }#"first_omega_0":200.0, 'hidden_omega_0':200.0} 
segmentation_config = {'hidden_features':[256,],#[128, 64],
                         'output_features' : NUM_CLASSES_AND_ONE}

inr_seg_model = models.SirenSegINR(
    inr_type='siren',
    inr_config=inr_config,
    segmentation_config=segmentation_config,
    normalize_features=NORMALIZE_FEATURES,
    ).float().cuda()


weights_from_metalearning = torch.load(weights_file)
inr_seg_model_wts = weights_from_metalearning['inr_seg_model']
best_inr_weights = weights_from_metalearning['best_inr_weights']
best_classifier_weights = weights_from_metalearning['best_classifier_weights']


coords_tmp = alpine.utils.coords.get_coords2d(RES[0], RES[1]).float().cuda()[None,...]
print(coords_tmp.shape)

train_ds = dataloaders.TorchMRI3D_Dataloader(json_file=config_file, mode='train', resolution=RES, coords=coords_tmp, config={'augment':RANDOM_AUGMENT}, num_classes=NUM_CLASSES, skip_pixels=SKIP_PIXELS,  dataset_dir=dataset_dir)
val_ds = dataloaders.TorchMRI3D_Dataloader(json_file=config_file, mode='val', resolution=RES, coords=coords_tmp, config={'augment':RANDOM_AUGMENT, 'N_samples':10}, num_classes=NUM_CLASSES, skip_pixels=SKIP_PIXELS,  dataset_dir=dataset_dir)
test_ds = dataloaders.TorchMRI3D_Dataloader(json_file=config_file, mode='test', resolution=RES, coords=coords_tmp, config={'augment':RANDOM_AUGMENT}, num_classes=NUM_CLASSES, skip_pixels=SKIP_PIXELS, dataset_dir=dataset_dir)
print(len(train_ds), len(val_ds), len(test_ds))

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)



train_clf_features = []
val_clf_features = []

if SAVE_FEATURE_VECS:

    pbar_train = tqdm(enumerate(train_dl), total=len(train_dl), position=0)
    for train_ix, train_data in pbar_train:
        _tmp_save_check =  osp.join(SAVE_PATHS,"train" , f"train_{train_ix}.pth")
        if osp.isfile(_tmp_save_check):
            continue
        
        inr_model_train = models.INR(**inr_config).float().cuda()
        inr_model_train.load_state_dict({k.replace("inr.",""): v.clone().detach() for k,v in deepcopy(best_inr_weights).items()})
        inr_model_train.compile()

        train_img = train_data['img'].float().cuda()
        train_seg = train_data['seg'].float().cuda()
        train_coords = train_data['coords'].float().cuda()

        inr_model_train.fit(train_coords, train_img, epochs=TEST_RUN_STEPS, disable_tqdm=True) 
        train_img_output, train_img_features = inr_model_train.forward_w_features(train_coords)

        data_dt = {
            'seg': train_seg.detach().clone().cpu().numpy(),
            'img': train_img_output.detach().clone().cpu().numpy(),
            'features': train_img_features[-2].detach().clone().cpu().numpy() # input to classifier
        }

        if SAVE_FEATURE_VECS:
            os.makedirs(osp.join(SAVE_PATHS,"train"), exist_ok=True)
            torch.save(data_dt, osp.join(SAVE_PATHS,"train" , f"train_{train_ix}.pth"),pickle_protocol=0)




if SAVE_FEATURE_VECS:

    pbar_val = tqdm(enumerate(val_dl), total=len(val_dl), position=0)
    for val_ix, val_data in pbar_val:
        _tmp_save_check =  osp.join(SAVE_PATHS,"val" , f"val_{val_ix}.pth")
        if osp.isfile(_tmp_save_check):
            continue
        
        inr_model_val = models.INR(**inr_config).float().cuda()
        inr_model_val.load_state_dict({k.replace("inr.",""): v.clone().detach() for k,v in deepcopy(best_inr_weights).items()})
        inr_model_val.compile()

        val_img = val_data['img'].float().cuda()
        val_seg = val_data['seg'].float().cuda()
        val_coords = val_data['coords'].float().cuda()

        inr_model_val.fit(val_coords, val_img, epochs=TEST_RUN_STEPS, disable_tqdm=True)
        val_img_output, val_img_features = inr_model_val.forward_w_features(val_coords)

        val_dt = {
            'seg':val_seg.detach().clone().cpu().numpy(),
            'img': val_img.detach().clone().cpu().numpy(),
            'features': val_img_features[-2].detach().clone().cpu().numpy(),
        }

        if SAVE_FEATURE_VECS:
            os.makedirs(osp.join(SAVE_PATHS, "val"), exist_ok=True)
            torch.save(val_dt, osp.join(SAVE_PATHS,"val" , f"val_{val_ix}.pth"), pickle_protocol=0)


if SAVE_FEATURE_VECS:

    test_clf_features = True
    test_features = []
    if test_clf_features:
        pbar_test = tqdm(enumerate(test_dl), total=len(test_dl), position=0)
        for test_ix, test_data in pbar_test:
            inr_model_test = models.INR(**inr_config).float().cuda()
            inr_model_test.load_state_dict({k.replace("inr.",""): v.clone().detach() for k,v in deepcopy(best_inr_weights).items()})
            inr_model_test.compile()

            test_img = test_data['img'].float().cuda()
            test_seg = test_data['seg'].float().cuda()
            test_coords = test_data['coords'].float().cuda()

            inr_model_test.fit(test_coords, test_img, epochs=TEST_RUN_STEPS, disable_tqdm=True)
            test_img_output, test_img_features = inr_model_test.forward_w_features(test_coords)

            test_dt = {
                'seg':test_seg.detach().clone().cpu().numpy(),
                'img': test_img.detach().clone().cpu().numpy(),
                # 'coords' : val_coords.detach().clone().cpu().numpy(),
                # 'resolution': val_data['resolution'],
                'features': test_img_features[-2].detach().clone().cpu().numpy(),
            }

            os.makedirs(osp.join(SAVE_PATHS,"test"), exist_ok=True)
            torch.save(test_dt, osp.join(SAVE_PATHS,"test" , f"test_{test_ix}.pth"), pickle_protocol=0)


if USE_SAVED_FEATURE_VECS_TO_TRAIN_CLF:
    data_dir_feature_vecs = SAVE_PATHS


train_clf_features_ds = dataloaders.CLFFeature(data_dir_feature_vecs, mode='train')
val_clf_ds = dataloaders.CLFFeature(data_dir_feature_vecs, mode='val')
test_clf_ds = dataloaders.CLFFeature(data_dir_feature_vecs, mode='test')



train_clf_dl = torch.utils.data.DataLoader(train_clf_features_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
val_clf_dl = torch.utils.data.DataLoader(val_clf_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
test_clf_dl = torch.utils.data.DataLoader(test_clf_ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)



CLASSIFIER_FINETUNE_EPOCHS = 100_000 # until convergence. stop when you see acc no longer decrease.


#### IMPORTANT: this step may have key mismatch based on how the model was saved. simply use the str.replace() function to match your saved keys to the model's named parameters


classifier_model = deepcopy(inr_seg_model.segmentation_head)

classifier_weights = deepcopy(best_classifier_weights)
try:
    classifier_model.load_state_dict(classifier_weights['final_clf_weights']) # check key, if final_clf_weights key does not exist, then just load classifier_weights as shown above.
except:
    classifier_weights = deepcopy({k.replace("segmentation_head.segmentation_head","segmentation_head"):v.clone().detach() for k,v in best_classifier_weights.items()})


if USE_SAVED_FEATURE_VECS_TO_TRAIN_CLF:
    LEARNING_RATE = 5e-5
    FOCAL_LOSS_GAMMA = 3.0
    ZERO_WT = 0.1

    EXPERIMENT_NAME  = f"gamma_{FOCAL_LOSS_GAMMA}_INR_300it_skip_pixels_{SKIP_PIXELS}_continue"

    classifier_opt = torch.optim.Adam(classifier_model.parameters(), lr=LEARNING_RATE)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_opt, CLASSIFIER_FINETUNE_EPOCHS, eta_min=1e-6)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(classifier_opt, step_size=100, gamma=0.95)
    print(classifier_model, list(classifier_weights.keys()), list(classifier_model.state_dict().keys()))

    finetune_classifier_loss_fn = loss_functions.LossFunction({'focal_loss':loss_functions.FocalSemanticLoss(gamma=FOCAL_LOSS_GAMMA)})

    final_classifier_weights = None
    best_val_score = 1e7

    pbar_epochs = tqdm(range(CLASSIFIER_FINETUNE_EPOCHS), position=0)
    for epoch in pbar_epochs:
        
        avg_loss_per_set = 0.0

        for train_ix, train_data in enumerate(train_clf_dl):
            train_seg = train_data['seg'].float().cuda()
            classifier_input = train_data['features'].float().cuda().squeeze(0).squeeze(0) # NC x F
            if NORMALIZE_FEATURES:
                classifier_input = nn.functional.normalize(classifier_input, dim=-1)

            classifier_opt.zero_grad()
            classifier_output = classifier_model(classifier_input)
            classifier_output = classifier_output.unsqueeze(0).unsqueeze(0)
            loss, loss_info = finetune_classifier_loss_fn({'output':{'segmentation_output':classifier_output}, 'seg':train_seg})
            loss.backward()
            classifier_opt.step()
            # lr_scheduler.step()
            avg_loss_per_set += float(loss.item())
            # pbar.set_description(f"Loss: {loss.item():.5f} ce={loss_info['ce_loss']:.5f}, PSNR = {psnr.item():.5f} Dice={loss_info['dice_loss']:.5f}")
        avg_loss_per_set /= len(train_clf_dl)
        pbar_epochs.set_description(f"Loss (clf): {avg_loss_per_set:.5f}. Best Val Loss(clf): {best_val_score:.5f}")
        pbar_epochs.refresh()

        if epoch % VAL_META_STEPS == 0 and epoch > 0:
            with torch.no_grad():
                avg_val_loss = 0.0
                for val_ix, val_data in enumerate(val_clf_dl):
                    val_seg = val_data['seg'].float().cuda()
                    classifier_val_input = val_data['features'].float().cuda().squeeze(0).squeeze(0) # NC x F
                    if NORMALIZE_FEATURES:
                        classifier_val_input = nn.functional.normalize(classifier_val_input, dim=-1)
                    
                    classifier_val_output = classifier_model(classifier_val_input)
                    classifier_val_output = classifier_val_output.unsqueeze(0).unsqueeze(0)
                    val_loss, val_loss_info = finetune_classifier_loss_fn({'output':{'segmentation_output':classifier_val_output}, 'seg':val_seg})

                    avg_val_loss += float(val_loss.item())
                avg_val_loss = avg_val_loss / len(val_clf_dl)
                pbar_epochs.set_description(f"Loss (clf): {avg_loss_per_set:.5f} Val Loss (clf): {avg_val_loss:.5f}")
                pbar_epochs.refresh()

                if avg_val_loss < best_val_score:
                    best_val_score = avg_val_loss
                    final_classifier_weights = deepcopy(classifier_model.state_dict())
                    tqdm.write(f'updated best val score to {best_val_score}')
                    os.makedirs(CLASSIFIER_WEIGHTS_DIR, exist_ok=True)
                    torch.save({'final_clf_weights':final_classifier_weights, 'focal_loss_gamma':FOCAL_LOSS_GAMMA, 'zero_wt':ZERO_WT}, 
                        osp.join(CLASSIFIER_WEIGHTS_DIR, f"classifierfinal_weights_LR_{LEARNING_RATE}_exp_{EXPERIMENT_NAME}.pth"))



