# Adapted from mmdet3d/datasets/transforms/loading.py
import mmengine
import numpy as np

from mmdet3d.datasets.transforms import LoadAnnotations3D
from mmdet3d.datasets.transforms.loading import get
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AddLayoutLabels:
    """
    Loading labels for ground-truth layout quads.
    """

    def _load_layout_labels(self, results):
        """
        Private function to load labels for ground-truth layout quads.

        Args:
            results (dict): Input dict.
        """
        n_quads = len(results['layout_verts'])
        labels = np.zeros(n_quads).astype('int')
        results['gt_labels_3d_layout'] = labels
        if 'eval_ann_info' in results.keys():
            results['eval_ann_info']['gt_labels_3d_layout'] = labels
            results['eval_ann_info']['n_quads_eval'] = results['n_quads_eval']
        return results

    def __call__(self, results: dict) -> dict:
        return self._load_layout_labels(results)


@TRANSFORMS.register_module()
class LoadAnnotations3D_(LoadAnnotations3D):
    """Just add super point mask loading.

    Args:
        with_sp_mask_3d (bool): Whether to load super point maks. 
    """

    def __init__(self, with_sp_mask_3d, **kwargs):
        self.with_sp_mask_3d = with_sp_mask_3d
        super().__init__(**kwargs)

    def _load_sp_pts_3d(self, results):
        """Private function to load 3D superpoints mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        sp_pts_mask_path = results['super_pts_path']

        try:
            mask_bytes = get(sp_pts_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            sp_pts_mask = np.frombuffer(mask_bytes, dtype=np.int64).copy()
        except ConnectionError:
            mmengine.check_file_exist(sp_pts_mask_path)
            sp_pts_mask = np.fromfile(
                sp_pts_mask_path, dtype=np.int64)

        results['sp_pts_mask'] = sp_pts_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['sp_pts_mask'] = sp_pts_mask
            results['eval_ann_info']['lidar_idx'] = \
                sp_pts_mask_path.split("/")[-1][:-4]
        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_sp_mask_3d:
            results = self._load_sp_pts_3d(results)
        return results
