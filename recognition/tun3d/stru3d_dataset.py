from mmdet3d.registry import DATASETS
from mmdet3d.datasets import S3DISDataset
import numpy as np


@DATASETS.register_module()
class Stru3DDataset(S3DISDataset):
    """
    Structured3D dataset class. We just changed metainfo.
    """
    METAINFO = {
        'classes': ('door', 'window'),
        'seg_valid_class_ids': (0, 1),
        'seg_all_class_ids':
        tuple(range(2)),
        'palette': [(255, 0, 0),
                    (10, 200, 100)]
    }
