import os
import sys
import json
import argparse
import numpy as np
from pprint import pprint
from os.path import join as opj

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from nucls_model.MiscUtils import load_saved_otherwise_default_model_configs
from nucls_model.ModelRunner import evaluateNucleusModel
from nucls_model.MaskRCNN import MaskRCNN

import nucls_model.PlottingUtils as pu
import nucls_model.torchvision_detection_utils.transforms as tvdt
from GeneralUtils import maybe_mkdir
from nucls_model.DataLoadingUtils import NucleusDataset, get_cv_fold_slides, _crop_all_to_fov, NucleusDatasetMask, NucleusDatasetMask_IMPRECISE
from nucls_model.ModelRunner import trainNucleusModel, load_ckp, evaluateNucleusModel
from nucls_model.MiscUtils import map_bboxes_using_hungarian_algorithm
from nucls_model.DataFormattingUtils import parse_sparse_mask_for_use
from configs.nucleus_model_configs import CoreSetQC, CoreSetNoQC

def test():
    # %%===========================================================================
    # Configs

    TAG = '[test.py]'
    BASEPATH = os.path.realpath('.')
    print(TAG, '[BASEPATH]', BASEPATH)
    model_name = '002_MaskRCNN_tmp'
    dataset_name = CoreSetQC.dataset_name
    all_models_root = opj(BASEPATH, 'results', 'tcga-nucleus', 'models', f'{dataset_name}')
    print(TAG, '[all_models_root]', all_models_root)
    model_root = opj(all_models_root, model_name)
    maybe_mkdir(model_root)

    # load configs
    configs_path = opj(model_root, 'nucleus_model_configs.py')
    cfg = load_saved_otherwise_default_model_configs(configs_path=configs_path)

    print(TAG, '[cfg]')
    pprint(cfg)

    # %%===========================================================================
    # Now test

    fold = 999
    model_folder = opj(model_root, f'fold_{fold}')
    maybe_mkdir(model_folder)
    checkpoint_path = opj(model_folder, f'{model_name}.ckpt')

    _, test_slides = get_cv_fold_slides(train_test_splits_path=CoreSetQC.train_test_splits_path, fold=fold)
    print(TAG, '[test_slides returned from get_cv_fold_slides]')
    model = MaskRCNN(**cfg.MaskRCNNConfigs.maskrcnn_params)
    print(TAG, '[MaskRCNN created as model]')
    model = DataParallel(model)
    test_dataset = NucleusDatasetMask(root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath, slides=test_slides, **cfg.MaskDatasetConfigs.test_dataset)
    print(TAG, '[NucleusDatasetMask created as test_dataset]')
    data_loader_test = DataLoader(dataset=test_dataset, **cfg.MaskDatasetConfigs.test_loader)
    print(TAG, '[DataLoader created as data_loader_test]')
    sample = next(iter(test_dataset))
    print(TAG, '[sample]')
    print(sample)
    output = model(sample)
    print(TAG, '[output]')
    print(output)

    # eval_results = evaluateNucleusModel(model, data_loader_test, checkpoint_path=checkpoint_path)
    # for i, eval_result in enumerate(eval_results):
    #     for key, value in eval_result.items():
    #         # print(TAG, type(eval_results[i][key]), key)
    #         if isinstance(value, np.int32):
    #             eval_results[i][key] = int(value)
    #         if isinstance(value, (np.float32, np.float64)):
    #             eval_results[i][key] = float(value)

    # print(TAG, '[saving eval_results]')
    # # pprint(json.dumps(eval_results, indent=4))
    # with open('eval_results.txt', 'w') as results_file:
    #     json.dump(eval_results, results_file, indent=4)

    # %%===========================================================================

if __name__ == '__main__':
    test()
