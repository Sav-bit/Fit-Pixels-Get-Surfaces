# coding: utf-8

# ## MetaSeg 3D segmentation STEP 1

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[3]:


import torch
torch.manual_seed(422)


# In[4]:


import os, os.path as osp
import sys

import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import torch.nn as nn
from pathlib import Path


# In[5]:


import numpy as np
import os, os.path as osp
import torch
import alpine
from matplotlib import pyplot as plt

from tqdm.autonotebook import tqdm


# In[6]:


import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.learner import INRMetaLearner
import dataloaders

from modules import models
from modules import loss_functions
from modules import metrics
from modules import utils
from modules import vis


# In[7]:


import json


OUTPUT_DIR = Path(__file__).resolve().parent / "dumps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ## Enter dataset path below!

# In[8]:


dataset_dir = "/projects/thesis-saverio/data"
config_file = str(REPO_ROOT / "config" / "oasis_splits_3d.json")


# In[9]:


files_data = json.load(open(config_file, 'r'))
train_files = files_data['train']
val_files = files_data['val']
test_files = files_data['test']


# In[10]:


# check overlap for any samples.
print(len(train_files), len(val_files), len(test_files))
set_train = set([x['img'] for x in train_files])
set_val = set([x['img'] for x in val_files])
set_test = set([x['img'] for x in test_files])

if len(set_train.intersection(set_val)) > 0 or  len(set_train.intersection(set_test)) > 0 or  len(set_val.intersection(set_test)) > 0:
    print("WARNING: OVERLAPPING DATA SPLITS")
else:
    print("No overlap in data splits. GOOD TO GO!!!!!!!!!!!")


# ## Some hyperparameters for the optimization process

# In[11]:


INNER_STEPS = 2
RANDOM_AUGMENT = False
TEST_RUN_STEPS = VAL_STEPS  = 300 #for full res
SKIP_PIXELS = 2
VAL_META_STEPS = 100
OUTER_LOOP_ITERATIONS =  5000  # 5000
NUM_CLASSES = 4
NUM_CLASSES_AND_ONE = NUM_CLASSES + 1
# RES = (160, 192, 224)

RES = (160,160,200)
VAL_RES = RES = [160//SKIP_PIXELS, 160//SKIP_PIXELS, 200//SKIP_PIXELS]

NORMALIZE_FEATURES = False


# In[12]:


nonlin = 'siren'
inr_config = {"in_features":3, "out_features": 1, "hidden_features": 256, "hidden_layers": 4, }#"first_omega_0":200.0, 'hidden_omega_0':200.0} 
segmentation_config = {'hidden_features':[256,],#[128, 64],
                         'output_features' : NUM_CLASSES_AND_ONE}


# In[13]:


inr_seg_model = models.SirenSegINR(
    inr_type='siren',
    inr_config=inr_config,
    segmentation_config=segmentation_config,
    normalize_features=NORMALIZE_FEATURES,
    ).float().cuda()


# In[14]:


meta_learner = INRMetaLearner(
    model=inr_seg_model,
    inner_steps=INNER_STEPS,
    config={'inner_lr':1e-4, 'outer_lr':1e-4},
    custom_loss_fn = loss_functions.LossFunction(
        {'mse_loss':loss_functions.MSELoss(alpha=1.0, reduction='weighted_mean', zero_weight=0.1), 
         'focal_loss' : loss_functions.FocalSemanticLoss(alpha=1.0, gamma=3.0),
        },
    ),
    outer_optimizer='adam',
    inner_loop_loss_fn=None, # uses default loss fn.
)


# In[15]:


coords_tmp = alpine.utils.coords.get_coords2d(RES[0], RES[1]).float().cuda()[None,...]
print(coords_tmp.shape)


# Set `NUM_VAL_EXAMPLES` variable. For faster evaluation, you can reduce `NUM_VAL_EXAMPLES`

# In[16]:


NUM_VAL_EXAMPLES = 100


# In[17]:


train_ds = dataloaders.TorchMRI3D_Dataloader(json_file=config_file, mode='train', resolution=RES, coords=coords_tmp, config={'augment':RANDOM_AUGMENT}, num_classes=NUM_CLASSES, skip_pixels=SKIP_PIXELS, dataset_dir=dataset_dir)
val_ds = dataloaders.TorchMRI3D_Dataloader(json_file=config_file, mode='val', resolution=RES, coords=coords_tmp, config={'augment':RANDOM_AUGMENT, 'N_samples':NUM_VAL_EXAMPLES}, num_classes=NUM_CLASSES, skip_pixels=SKIP_PIXELS, dataset_dir=dataset_dir)
test_ds = dataloaders.TorchMRI3D_Dataloader(json_file=config_file, mode='test', resolution=RES, coords=coords_tmp, config={'augment':RANDOM_AUGMENT}, num_classes=NUM_CLASSES, skip_pixels=SKIP_PIXELS, dataset_dir=dataset_dir)
print(len(train_ds), len(val_ds), len(test_ds))

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)


# In[18]:


best_weights = deepcopy(meta_learner.model_params)
best_inr_weights = None
best_classifier_weights = None
val_dice_score = []
val_iou_scores = []
val_psnr_scores = []
best_val_psnr = 0
best_val_dice_score = 0


# In[19]:


for i in range(OUTER_LOOP_ITERATIONS//len(train_dl)):
    
    pbar = tqdm(enumerate(train_dl), total=len(train_dl))
    for ix, data in pbar:
        img = data['img'].float().cuda()
        seg = data['seg'].float().cuda()
        coords = data['coords'].float().cuda()
        seg_integer = data['seg_integer'].float().cuda()
        loss, loss_info = meta_learner.forward(coords, {'gt':img, 'seg':seg, 'seg_integer':seg_integer,'resolution' : data['resolution']})
        psnr = -10 * np.log10(loss_info.get('mse_loss',0.01))
        pbar.set_description(f"Loss: {loss.item():.5f} PSNR = {psnr.item():.5f} Dice={loss_info.get('dice_loss',-1):.4f} FL={loss_info.get('focal_loss', -1):.5f} TV={loss_info.get('tv_loss',-1):.5f}")
        pbar.refresh()

        if ix % VAL_META_STEPS == 0:

            val_dice_score = []
            val_iou_scores = []
            val_psnr_scores = []
            
            for val_ix, val in tqdm(enumerate(val_dl), total=len(val_dl), position=1):
                val_img = val['img'].float().cuda()
                val_seg = val['seg'].float().cuda()
                val_coords = val['coords'].float().cuda()
                val_seg_integer = val['seg_integer'].float().cuda()

                render = meta_learner.render_inner_loop(val_coords, val_img, inner_loop_steps=VAL_STEPS)
                segmentation_output = render['output']['segmentation_output'].detach()#[0].detach().reshape(RES, RES, -1).cpu().numpy()
                segmentation_output = nn.functional.softmax(segmentation_output, dim=-1)
                segmentation_output = segmentation_output.argmax(dim=-1).detach().reshape(VAL_RES).cpu().numpy()
                img_recon = render['output']['inr_output'][0].reshape(VAL_RES).detach().cpu().numpy()
                segmentation_output_onehot = torch.nn.functional.one_hot(torch.tensor(segmentation_output), num_classes=NUM_CLASSES_AND_ONE)
                val_reshaped = val_img[0].detach().cpu().numpy().reshape(VAL_RES)
                val_seg_reshaped = val_seg_integer.detach().cpu().numpy().reshape(VAL_RES)
                # for k_x in range(0, VAL_RES[-1], VAL_RES[-1]//4):
                #     plt.figure()
                #     plt.subplot(121)
                #     plt.imshow(np.concatenate([img_recon[...,k_x], val_reshaped[...,k_x]], axis=1))
                #     plt.subplot(122)
                #     plt.imshow(np.concatenate([segmentation_output[...,k_x], val_seg_reshaped[...,k_x]],axis=1))
                #     plt.title(f"Iteration={ix}, Val Iteration={val_ix}")
                #     plt.show()
                mse_val = img_recon[...,40:80].flatten() - val_reshaped[...,40:80].flatten()
                mse_val = np.mean(mse_val**2)
                psnr = psnr = -10 * np.log10(mse_val)
                val_psnr_scores.append(float(psnr))
                dice_score = metrics.multiclass_dice_score_3d(segmentation_output_onehot.cuda(), val_seg.reshape(VAL_RES[0], VAL_RES[1], VAL_RES[2], -1).cuda(), num_classes=NUM_CLASSES_AND_ONE)
                val_dice_score.append(float(dice_score.item()))

            
            if np.mean(val_dice_score) > best_val_dice_score:
                best_val_psnr = np.mean(val_psnr_scores)
                best_val_dice_score = np.mean(val_dice_score)
                best_weights = deepcopy(meta_learner.model_params)
                best_inr_weights = deepcopy(meta_learner.get_inr_parameters())
                best_classifier_weights = deepcopy(meta_learner.get_segmentation_parameters())
                best_idx= ix
                print('updated dice score to ', best_val_dice_score)
                torch.save({'inr_seg_model':inr_seg_model.state_dict(),
                            'best_inr_weights':best_inr_weights,
                            'best_classifier_weights':best_classifier_weights}, 
                            OUTPUT_DIR / f"weights3d_num_classes_{NUM_CLASSES}_IS_{INNER_STEPS}.pth")

            print(f"Mean PSNR={np.mean(val_psnr_scores):.5f} +/- {np.std(val_psnr_scores):.5f}")
            
            print(f"Mean Dice={np.mean(val_dice_score):.5f} +/- {np.std(val_dice_score):.5f}")
        

print("Best weights from Dice=", best_val_dice_score)


# In[ ]:


torch.save({'inr_seg_model':inr_seg_model.state_dict(),'best_inr_weights':best_inr_weights,
                'best_classifier_weights':best_classifier_weights}, OUTPUT_DIR / f"weights3d_num_classes_{NUM_CLASSES}_IS_{INNER_STEPS}.pth")


# In[ ]:




