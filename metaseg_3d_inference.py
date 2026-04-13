import os

from modules import models
os.environ['CUDA_VISIBLE_DEVICES']  = '0'
import torch
torch.manual_seed(422)

INNER_STEPS = 2
RANDOM_AUGMENT = False
TEST_RUN_STEPS = VAL_STEPS  = 100
SKIP_PIXELS = 2
VAL_META_STEPS = 100
OUTER_LOOP_ITERATIONS =  5000  # 5000
NUM_CLASSES = 4
NUM_CLASSES_AND_ONE = NUM_CLASSES + 1
# RES = (160, 192, 224)

RES = (160,160,200)
VAL_RES = RES = [160//SKIP_PIXELS, 160//SKIP_PIXELS, 200//SKIP_PIXELS]

NORMALIZE_FEATURES = False

config_file = "./config/oasis_splits_3d.json"

nonlin = 'siren'
inr_config = {"in_features":3, "out_features": 1, "hidden_features": 256, "hidden_layers": 4, }#"first_omega_0":200.0, 'hidden_omega_0':200.0} 
segmentation_config = {'hidden_features':[256,],#[128, 64],
                         'output_features' : NUM_CLASSES_AND_ONE}

inr_seg_model = models.SirenSegINR(
    inr_type='siren',
    inr_config=inr_config,
    segmentation_config=segmentation_config,
    normalize_features=NORMALIZE_FEATURES, #only change

).float().cuda()