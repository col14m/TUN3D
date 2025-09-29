import math
import itertools
from collections import defaultdict
from dataclasses import dataclass
from mmengine.logging import print_log
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely import Polygon
from terminaltables import AsciiTable

import warnings
warnings.filterwarnings("ignore")

"""
Evaluation script for layout estimation on Structured3D adapted for our model outputs. 
Original: https://github.com/manycore-research/SpatialLM/blob/main/eval.py
"""

ZERO_TOLERANCE = 1e-6
LARGE_COST_VALUE = 1e6


@dataclass
class EvalTuple:
    tp: int
    num_pred: int
    num_gt: int

    @property
    def precision(self):
        return self.tp / self.num_pred if self.num_pred > 0 else 0

    @property
    def recall(self):
        return self.tp / self.num_gt if self.num_gt > 0 else 0

    @property
    def f1(self):
        return (
            (2 * self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0
        )

    @property
    def masked(self):
        return self.num_pred == 0 and self.num_gt == 0


def are_planes_parallel_and_close(
    corners_1: np.ndarray,
    corners_2: np.ndarray,
    parallel_tolerance: float,
    dist_tolerance: float,
):
    p1, p2, p3, _ = corners_1
    q1, q2, q3, _ = corners_2
    n1 = np.cross(np.subtract(p2, p1), np.subtract(p3, p1))
    n2 = np.cross(np.subtract(q2, q1), np.subtract(q3, q1))
    n1_length = np.linalg.norm(n1)
    n2_length = np.linalg.norm(n2)
    assert (
        n1_length * n2_length > ZERO_TOLERANCE
    ), f"Invalid plane corners, corners_1: {corners_1}, corners_2: {corners_2}"

    return (
        np.linalg.norm(np.cross(n1, n2)) / (n1_length * n2_length) < parallel_tolerance
        and np.dot(np.subtract(q1, p1), n1) / n1_length < dist_tolerance
    )


def calc_thin_bbox_iou_2d(
    corners_1: np.ndarray,
    corners_2: np.ndarray,
    parallel_tolerance: float,
    dist_tolerance: float,
):
    if are_planes_parallel_and_close(
        corners_1, corners_2, parallel_tolerance, dist_tolerance
    ):
        p1, p2, _, p4 = corners_2
        v1 = np.subtract(p2, p1)
        v2 = np.subtract(p4, p1)
        basis1 = v1 / np.linalg.norm(v1)
        basis1_orth = v2 - np.dot(v2, basis1) * basis1
        basis2 = basis1_orth / np.linalg.norm(basis1_orth)

        projected_corners_1 = [
            [
                np.dot(np.subtract(point, p1), basis1),
                np.dot(np.subtract(point, p1), basis2),
            ]
            for point in corners_1
        ]
        projected_corners_2 = [
            [
                np.dot(np.subtract(point, p1), basis1),
                np.dot(np.subtract(point, p1), basis2),
            ]
            for point in corners_2
        ]
        box1 = Polygon(projected_corners_1)
        box2 = Polygon(projected_corners_2)

        return calc_poly_iou(box1, box2)
    else:
        return 0


def calc_poly_iou(poly1, poly2):
    if poly1.intersects(poly2):
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        poly_iou = inter_area / union_area if union_area > 0 else 0
    else:
        poly_iou = 0
    return poly_iou


def calc_layout_tp(
    pred_entities,
    gt_entities,
    iou_threshold: float = 0.25,
    parallel_tolerance: float = math.sin(math.radians(5)),
    dist_tolerance: float = 0.2,
):
    num_pred = len(pred_entities)
    num_gt = len(gt_entities)
    if num_pred == 0 or num_gt == 0:
        return EvalTuple(0, num_pred, num_gt)

    iou_matrix = torch.as_tensor(
        [
            calc_thin_bbox_iou_2d(
                corners_1, corners_2, parallel_tolerance, dist_tolerance
            )
            for corners_1, corners_2 in itertools.product(
                [entity for entity in pred_entities],
                [entity for entity in gt_entities],
            )
        ]
    ).resize(num_pred, num_gt)

    cost_matrix = torch.full((num_pred, num_gt), LARGE_COST_VALUE)
    cost_matrix[iou_matrix > iou_threshold] = -1

    indices = linear_sum_assignment(cost_matrix.numpy())

    tp_percent = iou_matrix[
        torch.as_tensor(indices[0], dtype=torch.int64),
        torch.as_tensor(indices[1], dtype=torch.int64),
    ]
    tp = torch.sum(tp_percent >= iou_threshold).item()

    return EvalTuple(tp, num_pred, num_gt)


def spatial_lm_layout_eval(gt_annos,
                           dt_annos,
                           logger=None):

    assert len(dt_annos) == len(gt_annos)
    if not len(gt_annos):
        return {}
    classwise_eval_tuples_25 = defaultdict(list)
    classwise_eval_tuples_50 = defaultdict(list)
    LAYOUTS = ['wall', 'door', 'window']
    for img_id in range(len(dt_annos)):
        curr_gt_anno = gt_annos[img_id]
        curr_dt_anno = dt_annos[img_id]
        for class_name in LAYOUTS:
            classwise_eval_tuples_25[class_name].append(
                calc_layout_tp(
                    pred_entities=curr_dt_anno[class_name],
                    gt_entities=curr_gt_anno[class_name],
                    iou_threshold=0.25
                )
            )

            classwise_eval_tuples_50[class_name].append(
                calc_layout_tp(
                    pred_entities=curr_dt_anno[class_name],
                    gt_entities=curr_gt_anno[class_name],
                    iou_threshold=0.50
                )
            )

    headers = ["Layouts", "F1 @.25 IoU", "F1 @.50 IoU"]
    table_data = [headers]
    mean_25 = []
    mean_50 = []
    ret_dict = dict()
    for class_name in LAYOUTS:
        tuples = classwise_eval_tuples_25[class_name]
        f1_25 = np.ma.masked_where(
            [t.masked for t in tuples], [t.f1 for t in tuples]
        ).mean()

        tuples = classwise_eval_tuples_50[class_name]
        f1_50 = np.ma.masked_where(
            [t.masked for t in tuples], [t.f1 for t in tuples]
        ).mean()

        data_arr = [class_name, '{:.4f}'.format(f1_25), '{:.4f}'.format(f1_50)]
        table_data.append(data_arr)
        mean_25.append(f1_25)
        mean_50.append(f1_50)
        ret_dict[f'{class_name}_f1_25'] = f1_25
        ret_dict[f'{class_name}_f1_50'] = f1_25

    overall_f1_25 = np.array(mean_25).mean()
    overall_f1_50 = np.array(mean_50).mean()
    final_arr = ['Overall', '{:.4f}'.format(overall_f1_25), '{:.4f}'.format(overall_f1_50)]
    table_data.append(final_arr)
    ret_dict['f1_25'] = overall_f1_25
    ret_dict['f1_50'] = overall_f1_50
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict
