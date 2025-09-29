import numpy as np
from .structures import InstanceData_
from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet3d.datasets.transforms.formating import to_tensor
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints


@TRANSFORMS.register_module()
class Pack3DDetInputs_(Pack3DDetInputs):
    """Modified version of Pack3DDetInputs that keeps layout information.
    """
    INPUTS_KEYS = ['points']  # Removed elastic_coords
    SEG_KEYS = [
        'gt_seg_map',
        'pts_instance_mask',
        'pts_semantic_mask',
        'gt_semantic_seg'
    ]
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]

    def __init__(
        self,
        keys: tuple,
        load_meta=True
    ) -> None:
        super().__init__(keys)
        if load_meta:
            self.meta_keys += ('lidar2img', 'points',
                               'lidar_path', 'lidar_idx')

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths',
                'gt_labels_3d', 'gt_labels_3d_layout'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])

        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])
        if 'layout_verts' in results:
            results['layout_verts'] = to_tensor(results['layout_verts'])
        if 'door_verts' in results:
            results['door_verts'] = to_tensor(results['door_verts'])
        if 'window_verts' in results:
            results['window_verts'] = to_tensor(results['window_verts'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData_()
        gt_instances = InstanceData_()
        gt_pts_seg = PointData()

        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data_sample.set_metainfo(img_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')
        if 'gt_bboxes_3d' in results.keys():
            gt_instances_3d['bboxes_3d'] = results['gt_bboxes_3d']
        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg

        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
            if 'layout_verts' in results:
                data_sample.eval_ann_info['gt_layout'] = results['layout_verts']
            if 'door_verts' in results:
                data_sample.eval_ann_info['gt_door_verts'] = results['door_verts']
            if 'window_verts' in results:
                data_sample.eval_ann_info['gt_window_verts'] = results['window_verts']
            if 'horizontal_quads' in results:
                data_sample.eval_ann_info['horizontal_quads'] = results['horizontal_quads']
        else:
            data_sample.eval_ann_info = None

        # Keep layout-related information
        if 'layout_verts' in results:
            data_sample.gt_instances_3d.gt_layout = results['layout_verts']
            data_sample.gt_instances_3d.gt_labels_3d_layout = results['gt_labels_3d_layout']

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results


@TRANSFORMS.register_module()
class Pack3DDetInputsWithSP_(Pack3DDetInputs):
    """Modified version of Pack3DDetInputs that keeps layout information and superpoint masks.
    """
    INPUTS_KEYS = ['points']
    SEG_KEYS = [
        'gt_seg_map',
        'pts_instance_mask',
        'pts_semantic_mask',
        'gt_semantic_seg',
        'sp_pts_mask',
    ]
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d',
        'gt_sp_masks'
    ]

    def __init__(
        self,
        keys: tuple,
        load_meta=True
    ) -> None:
        super().__init__(keys)
        if load_meta:
            self.meta_keys += ('lidar2img', 'points',
                               'lidar_path', 'lidar_idx', 'sp_pts_mask')

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths',
                # Kept layout-related keys
                'gt_labels_3d', 'gt_labels_3d_layout', 'sp_pts_mask', 'gt_sp_masks'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])

        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])
        if 'layout_verts' in results:
            results['layout_verts'] = to_tensor(results['layout_verts'])
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData_()
        gt_instances = InstanceData_()
        gt_pts_seg = PointData()

        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data_sample.set_metainfo(img_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')
        if 'gt_bboxes_3d' in results.keys():
            gt_instances_3d['bboxes_3d'] = results['gt_bboxes_3d']
        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg

        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
            if 'layout_verts' in results:
                data_sample.eval_ann_info['gt_layout'] = results['layout_verts']
            if 'horizontal_quads' in results:
                data_sample.eval_ann_info['horizontal_quads'] = results['horizontal_quads']
        else:
            data_sample.eval_ann_info = None

        # Keep layout-related information
        if 'layout_verts' in results:
            data_sample.gt_instances_3d.gt_layout = results['layout_verts']
            data_sample.gt_instances_3d.gt_labels_3d_layout = results['gt_labels_3d_layout']

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results
