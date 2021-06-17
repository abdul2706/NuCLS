import os
import copy
import numpy as np
import pandas as pd

import mmcv
from mmcv import Config
from mmcv.runner import HOOKS
from mmdet.core.evaluation.eval_hooks import EvalHook

from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor

@HOOKS.register_module(name='NuCLSEvalHook', force=True)
class NuCLSEvalHook(EvalHook):
    def __init__(self, eval_interval, fold_number, file_prefix, path_config, base_dir):
        self.hook_name = '[NuCLSEvalHook]'
        self.eval_interval = eval_interval
        self.fold_number = fold_number
        self.file_prefix = file_prefix
        self.cfg = Config.fromfile(path_config)
        self.cfg.data.samples_per_gpu = 1
        self.cfg.data.workers_per_gpu = 1
        self.base_dir = os.path.join(self.cfg.work_dir, base_dir)
        mmcv.mkdir_or_exist(self.base_dir)
        self.dataloader_test, self.dataset_test = self._build_dataloader(self.cfg.data, self.cfg.data.test)
        super().__init__(self.dataloader_test, interval=self.eval_interval, by_epoch=True, metric=['bbox', 'segm'])

    def _build_dataloader(self, cfg_data, data_type):
        if isinstance(data_type, dict):
            data_type.test_mode = True
        elif isinstance(data_type, list):
            for ds_cfg in data_type:
                ds_cfg.test_mode = True
        if cfg_data.samples_per_gpu > 1:
            data_type.pipeline = replace_ImageToTensor(data_type.pipeline)
        dataset = build_dataset(data_type)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg_data.samples_per_gpu,
            workers_per_gpu=cfg_data.workers_per_gpu,
            dist=False,
            shuffle=False)
        return data_loader, dataset
    
    def evaluate(self, runner, results):
        eval_res, eval_res_objectness = self._evaluate(results, runner.logger)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        self._save_to_csv(eval_res, runner.epoch + 1, 'default')
        self._save_to_csv(eval_res_objectness, runner.epoch + 1, 'objectness')

    # def evaluate_specific_epoch(self, model, epoch):
    #     results = single_gpu_test(model, self.dataloader_test)
    #     eval_res, eval_res_objectness = self._evaluate(results)
    #     _save_to_csv(eval_res, epoch)
    #     _save_to_csv(eval_res_objectness, epoch)

    def evaluate_specific_epoch(self, results, epoch):
        eval_res, eval_res_objectness = self._evaluate(results)
        self._save_to_csv(eval_res, epoch, 'default')
        self._save_to_csv(eval_res_objectness, epoch, 'objectness')

    def _evaluate(self, results, logger=None):
        dataset = self.dataloader.dataset
        dataset_objectness = self.prepare_dataset_for_objectness_eval(dataset)
        results_objectness = self.prepare_outputs_for_objectness_eval(results)
        eval_res = dataset.evaluate(results, logger=logger, metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000'], **self.eval_kwargs)
        eval_res_objectness = dataset_objectness.evaluate(results_objectness, logger=logger, metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000'], **self.eval_kwargs)
        
        print('[eval_res_objectness]\n', eval_res_objectness)
        print('[eval_res]\n', eval_res)

        return eval_res, eval_res_objectness
    
    def _save_to_csv(self, eval_results, epoch, label):
        TAG = f'[{label}]'
        columns_order = ['epoch', 'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l', 'bbox_AR@100', 'bbox_AR@300', 'bbox_AR@1000', 'bbox_AR_s@1000', 'bbox_AR_m@1000', 'bbox_AR_l@1000', 'segm_mAP', 'segm_mAP_50', 'segm_mAP_75', 'segm_mAP_s', 'segm_mAP_m', 'segm_mAP_l', 'segm_AR@100', 'segm_AR@300', 'segm_AR@1000', 'segm_AR_s@1000', 'segm_AR_m@1000', 'segm_AR_l@1000']

        del eval_results['bbox_mAP_copypaste']
        del eval_results['segm_mAP_copypaste']
        eval_results['epoch'] = epoch
        print(TAG, self.hook_name, '[eval_results]\n', eval_results)

        csv_metric_path = os.path.join(self.base_dir, f'{self.file_prefix}-fold{self.fold_number}-{label}.csv')
        if not os.path.exists(csv_metric_path):
            df_metric = pd.DataFrame(eval_results, index=[0], columns=eval_results.keys())
        else:
            df_metric = pd.read_csv(csv_metric_path)
            df_metric = df_metric.append(eval_results, ignore_index=True)
        
        df_metric = df_metric[columns_order]
        df_metric.to_csv(csv_metric_path, index=False)
        print(self.hook_name, '[df_metric]\n', df_metric)

    def prepare_dataset_for_objectness_eval(self, dataset):
        dataset2 = copy.deepcopy(dataset)
        dataset2.CLASSES = (('nucleus'), )
        dataset2.coco.dataset['categories'] = [{'id': 0, 'name': 'nucleus'}, ]
        for annotation in dataset2.coco.dataset['annotations']:
            annotation['category_id'] = 0
        dataset2.coco.createIndex()
        dataset2.cat_ids = dataset2.coco.get_cat_ids(cat_names=dataset2.CLASSES)
        dataset2.cat2label = {cat_id: i for i, cat_id in enumerate(dataset2.cat_ids)}
        dataset2.img_ids = dataset2.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in dataset2.img_ids:
            info = dataset2.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = dataset2.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return dataset2

    def prepare_outputs_for_objectness_eval(self, outputs):
        outputs2 = []
        for i in range(len(outputs)):
            output = list(outputs[i])
            output[0] = np.asarray([np.concatenate(output[0], axis=0)])
            output[1] = np.asarray([np.concatenate(output[1], axis=0)])
            output2 = (output[0], output[1])
            outputs2.append(output2)
        return outputs2
