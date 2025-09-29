# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from .indoor_eval import indoor_eval, indoor_layout_eval
from .spatial_lm_eval import spatial_lm_layout_eval
from .transforms_3d import LayoutToBBoxes
from mmdet3d.registry import METRICS
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.structures import get_box_type
from .show_results import show_result_v2
from pathlib import Path
import numpy as np


@METRICS.register_module()
class IndoorLayoutMetric_(BaseMetric):
    def __init__(self,
                 datasets,
                 datasets_classes,
                 vis_dir: str = None,
                 iou_thr: List[float] = [0.25, 0.5],
                 dist_thr: List[float] = [0.4],
                 floor_and_ceiling: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        """
        Initialize the IndoorLayoutMetric_ metric module.

        This class evaluates 3D indoor scene layout estimation and object detection results
        across different datasets. 

        Args:
            datasets (list[str]):  
                List of dataset names to be evaluated (e.g., ``["scannet", "s3dis"]``).
            datasets_classes (list[list[str]]):  
                List of class name lists corresponding to each dataset in ``datasets``.  
            vis_dir (str, optional):  
                Path to the directory where visualizations will be saved.  
            iou_thr (list[float], optional):  
                List of IoU thresholds for evaluating 3D object detection.  
            dist_thr (list[float] or float, optional):  
                Distance threshold(s) used for quad vertex matching in layout evaluation.  
            floor_and_ceiling (bool, optional):  
                Whether to include floor and ceiling quads in layout evaluation.  
            collect_device (str, optional):  
                Device on which to collect results (e.g., ``"cpu"`` or ``"gpu"``).  
            prefix (str, optional):  
                Prefix string for metric names, useful when logging multiple metrics.  
        """
        super(IndoorLayoutMetric_, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.dist_thr = [dist_thr] if isinstance(dist_thr, float) else dist_thr
        self.iou_thr = iou_thr
        self.datasets = datasets
        self.datasets_classes = datasets_classes
        self.vis_dir = vis_dir
        self.layout2bbox = None
        if self.vis_dir is not None:
            self.layout2bbox = LayoutToBBoxes()
        self.floor_and_ceiling = floor_and_ceiling

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            lidar_path = data_sample.get('lidar_path')
            dataset = self.get_dataset(lidar_path)
            pred_3d['dataset'] = dataset
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v

            eval_ann_info['lidar_idx'] = lidar_path.split("/")[-1][:-4]
            predl_numpy = np.array([ds.detach().cpu().numpy()
                                   for ds in pred_3d['layout_verts']])
            gt_layout_verts = eval_ann_info['gt_layout'].detach().cpu().numpy()
            gt_layout = dict(layout_verts=gt_layout_verts,
                             horizontal_quads=np.array(eval_ann_info['horizontal_quads']),
                             overall_quads=eval_ann_info['n_quads_eval'],
                             lidar_idx=lidar_path.split("/")[-1][:-4],
                             points=pred_3d['points'].detach().cpu().numpy())
            if dataset == 'structured3d':
                gt_door_verts = np.array(eval_ann_info['gt_door_verts'])
                gt_window_verts = np.array(eval_ann_info['gt_window_verts'])
                gt_layout.update(dict(door_verts=gt_door_verts,
                                      window_verts=gt_window_verts))

            cpu_pred_3d_layout = dict(layout_verts=predl_numpy,
                                      dataset=dataset)

            self.results.append(
                (eval_ann_info, cpu_pred_3d, gt_layout, cpu_pred_3d_layout))

    def get_dataset(self, lidar_path):
        for dataset in self.datasets:
            if dataset in lidar_path.split('/'):
                return dataset

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """

        logger: MMLogger = MMLogger.get_current_instance()
        gt_obj_ann = [[] for _ in self.datasets]
        pred_obj = [[] for _ in self.datasets]
        gt_layout_ann = [[] for _ in self.datasets]
        pred_layout = [[] for _ in self.datasets]
        for gt_obj, single_pred_results_obj, gt_layout, single_pred_results_layout in results:
            idx = self.datasets.index(single_pred_results_obj['dataset'])
            gt_obj_ann[idx].append(gt_obj)
            pred_obj[idx].append(single_pred_results_obj)
            if self.vis_dir is not None:
                self.vis_results_obj(gt_obj, single_pred_results_obj)
            if len(gt_layout['layout_verts']) > 0:
                gt_layout_ann[idx].append(gt_layout)
                pred_layout[idx].append(single_pred_results_layout)
                if self.vis_dir is not None:
                    self.vis_results_layout(
                        gt_layout, single_pred_results_layout)

        box_type_3d, box_mode_3d = get_box_type(
            self.dataset_meta.get('box_type_3d', 'depth'))

        ret_dict = {}
        for i in range(len(self.datasets)):
            ret_dict[self.datasets[i]] = dict()
            if self.datasets[i] == 'structured3d':
                processed_gt_ann = []
                for scene_ann in gt_layout_ann[i]:
                    curr_processed_ann = dict()
                    curr_processed_ann['wall'] = [validate_quad(v) for v in scene_ann['layout_verts']]
                    curr_processed_ann['door'] = [validate_quad(v) for v in scene_ann['door_verts']]
                    curr_processed_ann['window'] = [validate_quad(v) for v in scene_ann['window_verts']]
                    processed_gt_ann.append(curr_processed_ann)

                processed_pred_ann = self.process_ann_for_stru3d(
                    pred_obj[i], pred_layout[i])

                ret_dict[self.datasets[i]]['layout'] = spatial_lm_layout_eval(
                    processed_gt_ann,
                    processed_pred_ann,
                    logger=logger)
            else:
                ret_dict[self.datasets[i]]['objects'] = indoor_eval(
                    gt_obj_ann[i],
                    pred_obj[i],
                    self.iou_thr,
                    self.datasets_classes[i],
                    logger=logger,
                    box_mode_3d=box_mode_3d)

                ret_dict[self.datasets[i]]['layout'] = indoor_layout_eval(
                    gt_layout_ann[i],
                    pred_layout[i],
                    self.dist_thr,
                    logger=logger,
                    count_floor_and_ceiling=self.floor_and_ceiling)

        return ret_dict

    def process_ann_for_stru3d(self, 
                               obj_ann, 
                               layout_ann, 
                               prefix='',
                               obj_label_mapping=['door', 'window']):
        processed_ann = []
        for curr_obj_ann, curr_layout_ann in zip(obj_ann, layout_ann):
            curr_processed_ann = dict()
            curr_v = curr_layout_ann['layout_verts']
            curr_processed_ann['wall'] = [validate_quad(v) for v in curr_v]
            for obj_label in obj_label_mapping:
                curr_processed_ann[obj_label] = []
            curr_bboxes = curr_obj_ann[prefix + 'bboxes_3d'] 
            curr_labels = curr_obj_ann[prefix + 'labels_3d']
            for box, label in zip(curr_bboxes, curr_labels):
                curr_obj_label = obj_label_mapping[label]
                box = DepthInstance3DBoxes(box[:6][None, ...], with_yaw=False, box_dim=6)
                curr_verts = get_quad_from_bbox(box.corners.numpy()[0])
                curr_processed_ann[curr_obj_label].append(curr_verts)
            processed_ann.append(curr_processed_ann)
        return processed_ann

    def vis_results_obj(self, eval_ann, sinlge_pred_results):
        pts = sinlge_pred_results['points'].numpy()
        pts[:, 3:] *= 127.5
        pts[:, 3:] += 127.5
        show_result_v2(pts, eval_ann['gt_bboxes_3d'].corners,
                       eval_ann['gt_labels_3d'],
                       sinlge_pred_results['bboxes_3d'].corners,
                       sinlge_pred_results['labels_3d'],
                       Path(self.vis_dir) / sinlge_pred_results['dataset'],
                       eval_ann['lidar_idx'])

    def vis_results_layout(self, eval_ann, sinlge_pred_results):
        # pts = sinlge_pred_results['points']
        # pts[:, 3:] *= 127.5
        # pts[:, 3:] += 127.5

        gt_layout_verts = eval_ann['layout_verts']
        pred_layout_verts = sinlge_pred_results['layout_verts']
        gt_bboxes_ann = self.layout2bbox.transform(
            dict(layout_verts=gt_layout_verts))
        pred_bboxes_ann = self.layout2bbox.transform(
            dict(layout_verts=pred_layout_verts))
        show_result_v2(None, gt_bboxes_ann['gt_bboxes_3d'].corners,
                       gt_bboxes_ann['gt_labels_3d'],
                       pred_bboxes_ann['gt_bboxes_3d'].corners,
                       pred_bboxes_ann['gt_labels_3d'],
                       Path(self.vis_dir) / sinlge_pred_results['dataset'],
                       eval_ann['lidar_idx'] + '_layout')


def validate_quad(quad):
    mask_center = quad.mean(axis=0)
    v1 = quad[0] - mask_center
    v2 = quad[1] - mask_center
    v3 = quad[2] - mask_center
    v4 = quad[3] - mask_center

    n_1 = np.cross(v1, v2)
    n_2 = np.cross(v3, v4)
    if (n_1 * n_2).sum() < 0:
        quad[[2, 3]] = quad[[3, 2]]

    return quad


def get_quad_from_bbox(pc_coords):
    min_vert = pc_coords.min(axis=0)
    max_vert = pc_coords.max(axis=0)
    size = max_vert - min_vert
    squeeze_ind = np.argmin(size)  # squeeze bbox in dimension with min size
    size[squeeze_ind] = 0
    m_size = np.copy(size)
    m_size[squeeze_ind - 1] *= -1
    mask_center = (min_vert + max_vert) / 2
    quad = [
        list(mask_center + size / 2),
        list(mask_center + m_size / 2),
        list(mask_center - size / 2),
        list(mask_center - m_size / 2)
    ]
    final_quad = np.array(sorted(quad, key=lambda vert: vert[-1]))

    return validate_quad(final_quad)
