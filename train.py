import sys
import os
from os.path import join as opj
import argparse
from pprint import pprint

def train():
    parser = argparse.ArgumentParser(description='Train nucleus model.')
    parser.add_argument('-f', type=int, default=[1], nargs='+', help='fold(s) to run')
    parser.add_argument('-g', type=int, default=[0], nargs='+', help='gpu(s) to use')
    parser.add_argument('--qcd', type=int, default=1, help='use QCd data for training?')
    parser.add_argument('--train', type=int, default=1, help='train?')
    parser.add_argument('--vistest', type=int, default=1, help='visualize results on testing?')
    args = parser.parse_args(['-f', '1', '-g', '0'])
    args.qcd = bool(args.qcd)
    args.train = bool(args.train)
    args.vistest = bool(args.vistest)

    # GPU allocation MUST happen before importing other modules
    from GeneralUtils import save_configs, maybe_mkdir, AllocateGPU
    # AllocateGPU(GPUs_to_use=args.g)

    from nucls_model.MiscUtils import load_saved_otherwise_default_model_configs
    from configs.nucleus_model_configs import CoreSetQC, CoreSetNoQC
    from nucls_model.NucleusWorkflows import run_one_maskrcnn_fold

    # %%===========================================================================
    # Configs

    TAG = '[train.py]'
    BASEPATH = os.path.realpath('.')
    print(TAG, '[BASEPATH]', BASEPATH)
    model_name = '002_MaskRCNN_tmp'
    dataset_name = CoreSetQC.dataset_name if args.qcd else CoreSetNoQC.dataset_name
    all_models_root = opj(BASEPATH, 'results', 'tcga-nucleus', 'models', f'{dataset_name}')
    print(TAG, '[all_models_root]', all_models_root)
    model_root = opj(all_models_root, model_name)
    maybe_mkdir(model_root)

    # load configs
    configs_path = opj(model_root, 'nucleus_model_configs.py')
    cfg = load_saved_otherwise_default_model_configs(configs_path=configs_path)

    # for reproducibility, copy configs & most relevant code file to results
    if not os.path.exists(configs_path):
        save_configs(configs_path=opj(BASEPATH, 'configs', 'nucleus_model_configs.py'), results_path=model_root)
    save_configs(configs_path=os.path.abspath(__file__), results_path=model_root, warn=False)
    save_configs(configs_path=opj(BASEPATH, 'nucls_model', 'NucleusWorkflows.py'), results_path=model_root, warn=False)

    print(TAG, '[cfg]')
    pprint(cfg)

    # %%===========================================================================
    # Now run

    for fold in args.f:
        run_one_maskrcnn_fold(
            fold=fold, cfg=cfg, model_root=model_root, model_name=model_name,
            qcd_training=args.qcd, train=args.train, vis_test=args.vistest)

    # %%===========================================================================

if __name__ == '__main__':
    train()
