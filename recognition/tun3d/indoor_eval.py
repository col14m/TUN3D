# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.logging import print_log
from terminaltables import AsciiTable


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def eval_det_cls(pred, gt, iou_thr=None):
    """Generic functions to compute precision/recall for object detection for a
    single class.

    Args:
        pred (dict): Predictions mapping from image id to bounding boxes
            and scores.
        gt (dict): Ground truths mapping from image id to bounding boxes.
        iou_thr (list[float]): A list of iou thresholds.

    Return:
        tuple (np.ndarray, np.ndarray, float): Recalls, precisions and
            average precision.
    """

    # {img_id: {'bbox': box structure, 'det': matched list}}
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, 7], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i].tensor
            bbox = gt[img_id][0].new_box(gt_cur)
        else:
            bbox = gt[img_id]
        det = [[False] * len(bbox) for i in iou_thr]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # construct dets
    image_ids = []
    confidence = []
    ious = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        pred_cur = torch.zeros((cur_num, 7), dtype=torch.float32)
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            pred_cur[box_idx] = box.tensor
            box_idx += 1
        pred_cur = box.new_box(pred_cur)
        gt_cur = class_recs[img_id]['bbox']
        if len(gt_cur) > 0:
            # calculate iou in each image
            iou_cur = pred_cur.overlaps(pred_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]

        if len(BBGT) > 0:
            # compute overlaps
            for j in range(len(BBGT)):
                # iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.

    ret = []
    for iou_idx, thresh in enumerate(iou_thr):
        # compute precision recall
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret.append((recall, precision, ap))

    return ret


def eval_map_recall(pred, gt, ovthresh=None):
    """Evaluate mAP and recall.

    Generic functions to compute precision/recall for object detection
        for multiple classes.

    Args:
        pred (dict): Information of detection results,
            which maps class_id and predictions.
        gt (dict): Information of ground truths, which maps class_id and
            ground truths.
        ovthresh (list[float], optional): iou threshold. Default: None.

    Return:
        tuple[dict]: dict results of recall, AP, and precision for all classes.
    """

    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh)
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]

    for label in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][
                    label] = ret_values[label][iou_idx]
            else:
                recall[iou_idx][label] = np.zeros(1)
                precision[iou_idx][label] = np.zeros(1)
                ap[iou_idx][label] = np.zeros(1)

    return recall, precision, ap


def indoor_eval(gt_annos,
                dt_annos,
                metric,
                label2cat,
                logger=None,
                box_mode_3d=None):
    """Indoor Evaluation.

    Evaluate the result of the detection.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Detection annotations. the dict
            includes the following keys

            - labels_3d (torch.Tensor): Labels of boxes.
            - bboxes_3d (:obj:`BaseInstance3DBoxes`):
                3D bounding boxes in Depth coordinate.
            - scores_3d (torch.Tensor): Scores of boxes.
        metric (list[float]): IoU thresholds for computing average precisions.
        label2cat (tuple): Map from label to category.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Return:
        dict[str, float]: Dict of results.
    """
    print(label2cat)
    assert len(dt_annos) == len(gt_annos)
    pred = {}  # map {class_id: pred}
    gt = {}  # map {class_id: gt}
    for img_id in range(len(dt_annos)):
        # parse detected annotations
        det_anno = dt_annos[img_id]
        for i in range(len(det_anno['labels_3d'])):
            label = det_anno['labels_3d'].numpy()[i]
            bbox = det_anno['bboxes_3d'].convert_to(box_mode_3d)[i]
            score = det_anno['scores_3d'].numpy()[i]
            if label not in pred:
                pred[int(label)] = {}
            if img_id not in pred[label]:
                pred[int(label)][img_id] = []
            if label not in gt:
                gt[int(label)] = {}
            if img_id not in gt[label]:
                gt[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))

        # parse gt annotations
        gt_anno = gt_annos[img_id]

        gt_boxes = gt_anno['gt_bboxes_3d']
        labels_3d = gt_anno['gt_labels_3d']

        for i in range(len(labels_3d)):
            label = labels_3d[i]
            bbox = gt_boxes[i]
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    rec, prec, ap = eval_map_recall(pred, gt, metric)
    ret_dict = dict()
    header = ['classes']
    print(ap[0].keys())
    table_columns = [[label2cat[label]
                      for label in ap[0].keys()] + ['Overall']]

    for i, iou_thresh in enumerate(metric):
        header.append(f'AP_{iou_thresh:.2f}')
        header.append(f'AR_{iou_thresh:.2f}')
        rec_list = []
        for label in ap[i].keys():
            ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                ap[i][label][0])
        ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
            np.nanmean(list(ap[i].values())))

        table_columns.append(list(map(float, list(ap[i].values()))))
        table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_rec_{iou_thresh:.2f}'] = float(
                rec[i][label][-1])
            rec_list.append(rec[i][label][-1])
        ret_dict[f'mAR_{iou_thresh:.2f}'] = float(np.nanmean(rec_list))

        table_columns.append(list(map(float, rec_list)))
        table_columns[-1] += [ret_dict[f'mAR_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict


def same_point(pred, gt, thr):
    """Check if two points are within a distance threshold.
    """

    distance = np.linalg.norm(np.array(pred-gt))
    return (distance <= thr)


def compute_correctness(pred_corner, all_gt, thr):
    """Check if a predicted quad matches any ground-truth quad.
    Args:
        pred_corner (array-like): Predicted quad corners, shape (4, 3).
        all_gt (list[array-like]): List of ground-truth quads, each of shape (4, 3).
        thr (float): Distance threshold for corner matching.

    Returns:
        tuple:
            - bool: True if the predicted quad matches any ground-truth quad, else False.
            - int or None: Index of the matching ground-truth quad, or None if no match.
    """
    for k, gt in enumerate(all_gt):
        correctness = True
        for i in range(0, 4):
            res = contain_point(gt, pred_corner[i], thr)
            if not res[0]:
                correctness = False
                break
            else:
                j = res[1]
                gt = gt[list(set(range(len(gt))) - set([j]))]
        if correctness:
            return True, k
    return False, None


def contain_point(pointlist, point, thr):
    """Check if a point is within a threshold distance of any point in a list.
    """
    for j, p in enumerate(pointlist):
        if same_point(p, point, thr):
            return True, j
    return False, None


def get_ceiling_and_floor(pred_corners, thr):
    """Get floor and ceiling from predicted vertical walls.
    Args:
        pred_corners (list[array-like]): List of predicted quads, each with 4 corners (4, 3).
        thr (float): Distance threshold to consider two points as the same.

    Returns:
        tuple:
            - ceilings (list[array-like]): Unique ceiling points.
            - floors (list[array-like]): Unique floor points.
    """
    ceilings = []
    floors = []
    for quad_corner in pred_corners:
        for i in range(2, 4):
            contain, j = contain_point(ceilings, quad_corner[i], thr)
            p = ceilings[j] if contain else None
            if not contain:
                ceilings.append(quad_corner[i])
        for i in range(0, 2):
            contain, j = contain_point(floors, quad_corner[i], thr)
            p = floors[j] if contain else None
            if not contain:
                floors.append(quad_corner[i])

    return ceilings, floors


def get_tp_fp_single_scan(scan_gt_annos,
                          scan_dt_annos,
                          thr, count_floor_and_ceiling=False):
    """Compute true positives and false positives for a single scan.

    Args:
        scan_gt_annos (dict): Ground-truth annotations containing:
            - 'layout_verts' (list[array-like]): Ground-truth quads, each (4, 3).
            - 'horizontal_quads' (list[array-like]): Floor and ceiling quads.
            - 'overall_quads' (int): Total number of ground-truth quads.
        scan_dt_annos (dict): Predicted annotations containing:
            - 'layout_verts' (list[array-like]): Predicted quads, each (4, 3).
        thr (float): Distance threshold for matching corners.
        count_floor_and_ceiling (bool, optional): Whether to count ceiling
            and floor quads. Default: False.

    Returns:
        dict: Counts for the scan:
            - 'tp' (int): Number of true positives.
            - 'fp' (int): Number of false positives.
            - 'npos' (int): Total number of ground-truth quads.
    """
    all_pred_corners = scan_dt_annos['layout_verts']
    all_gt_corners = scan_gt_annos['layout_verts']
    horizontal_quads = scan_gt_annos['horizontal_quads']

    tp = 0
    fp = 0
    npos = scan_gt_annos['overall_quads']
    for pred_corner in all_pred_corners:
        res = compute_correctness(pred_corner, all_gt_corners, thr)
        if res[0]:
            tp = tp + 1
            all_gt_corners = all_gt_corners[list(
                set(range(len(all_gt_corners))) - set([res[1]]))]
        else:
            fp = fp + 1

    if count_floor_and_ceiling:  # calculate horizontal quads
        ceilings, floors = get_ceiling_and_floor(all_pred_corners, thr)
        if len(ceilings) == 4:
            if compute_correctness(ceilings, horizontal_quads, thr)[0]:
                tp = tp + 1
        if len(floors) == 4:
            if compute_correctness(floors, horizontal_quads, thr)[0]:
                tp = tp + 1

    return {'tp': tp, 'fp': fp, 'npos': npos}


def compute_F1(tp, fp, npos):
    """Compute precision, recall, and F1 score.
    """
    p = tp/max((tp+fp), 1e-6)

    r = tp/npos

    f1 = 2.0*p*r/max((p+r), 1e-6)

    return {'p': p, 'r': r, 'f1': f1}


def indoor_layout_eval(gt_annos,
                       dt_annos,
                       dist_thr,
                       logger=None,
                       count_floor_and_ceiling=False):
    """Indoor Evaluation for layout.

    Evaluate the result of the layout estimation.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Predicted annotations.
        dist_thr (list[float]): distance thresholds for computing precision and recall.
        logger (logging.Logger | str, optional): The way to print the metrics
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        count_floor_and_ceiling (bool): Include floor and ceiling in tp/fp computation or not  

    Return:
        dict[str, float]: Dict of results.
    """
    assert len(dt_annos) == len(gt_annos)
    if not len(gt_annos):
        return {}
    tp_fp_accum = dict()
    for img_id in range(len(dt_annos)):
        curr_gt_anno = gt_annos[img_id]
        curr_dt_anno = dt_annos[img_id]
        for thr in dist_thr:
            if thr not in tp_fp_accum:
                tp_fp_accum[thr] = {'tp': 0, 'fp': 0,
                                    'npos': 0, 'metrics': None}

            quantities = get_tp_fp_single_scan(curr_gt_anno, curr_dt_anno, thr,
                                               count_floor_and_ceiling=count_floor_and_ceiling)
            tp_fp_accum[thr]['tp'] += quantities['tp']
            tp_fp_accum[thr]['fp'] += quantities['fp']
            tp_fp_accum[thr]['npos'] += quantities['npos']

    for thr in dist_thr:
        accum = tp_fp_accum[thr]
        print(accum['npos'])
        metrics = compute_F1(accum['tp'], accum['fp'], accum['npos'])
        tp_fp_accum[thr]['metrics'] = metrics

    header = ['Threshold', 'Precision', 'Recall', 'F1 score']
    data = [header]
    ret_dict = dict()
    for thr in dist_thr:
        precision = tp_fp_accum[thr]['metrics']['p']
        recall = tp_fp_accum[thr]['metrics']['r']
        f1_score = tp_fp_accum[thr]['metrics']['f1']
        data.append([thr, f"{precision:.4f}",
                    f"{recall:.4f}", f"{f1_score:.4f}"])
        ret_dict[f'precision_{thr}'] = precision
        ret_dict[f'recall_{thr}'] = recall
        ret_dict[f'f1_{thr}'] = f1_score
    table = AsciiTable(data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict
