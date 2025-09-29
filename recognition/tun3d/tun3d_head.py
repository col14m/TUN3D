# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/tr3d/blob/master/mmdet3d/models/dense_heads/tr3d_head.py # noqa
from typing import List, Optional, Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

import torch
from mmcv.ops import nms3d, nms3d_normal
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor, nn
from .structures import InstanceData_

from mmdet3d.models import Base3DDenseHead
from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes, DepthInstance3DBoxes
from torch_scatter import scatter_mean
from mmdet3d.utils import InstanceList, OptInstanceList
from mmdet3d.structures import rotation_3d_in_axis
import itertools
from .transforms_3d import LayoutToBBoxes


@MODELS.register_module()
class LayoutHead(nn.Module):
    """Head for predicting the layout vertices from backbone features.

    This module projects 3D features into 2D and outputs both regression
    and classification predictions for layout quads.

    Args:
        input_dim (int): Number of input channels for sparse convolution.
        num_layout_reg_outs (int): Output regression dimension (e.g., coords).
        n_spconv2d (int): Number of intermediate 2D sparse conv blocks.
        n_q_feats (int): Number of z-quantiles.
        voxel_size (float): Voxel size in meters.

    Forward Inputs:
        x (ME.SparseTensor): Sparse 3D voxelized input tensor.
        ceiling (Tensor): z-quantiles tensor of shape (N, n_q_feats).

    Forward Outputs:
        tuple:
            - cls_out (Tensor): Classification logits of shape (N, 1).
            - reg_out (Tensor): Regression outputs of shape (N, num_layout_reg_outs).
            - x_proj_w_z_feat (ME.SparseTensor): 2D sparse tensor with projected feats.
    """
    def __init__(self, input_dim, num_layout_reg_outs, n_spconv2d, n_q_feats, voxel_size):
        super(LayoutHead, self).__init__()

        self.n_q_feats = n_q_feats
        conv = ME.MinkowskiConvolution
        hidden_dim = 32
        self.add_feats_encoder = nn.Sequential(
            nn.Linear(n_q_feats, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.out_reg_conv = nn.Sequential(
            conv(
                input_dim + hidden_dim, 
                input_dim + hidden_dim, 
                kernel_size=1, 
                bias=True, 
                dimension=2),
            ME.MinkowskiReLU(inplace=True),
            conv(
                input_dim + hidden_dim, 
                num_layout_reg_outs, 
                kernel_size=1, 
                bias=True, 
                dimension=2)
        )

        self.out_cls_conv = conv(
            input_dim, 1, kernel_size=1, bias=True, dimension=2)

        nn.init.normal_(self.out_reg_conv[0].kernel, std=.01)
        nn.init.normal_(self.out_reg_conv[2].kernel, std=.01)
        nn.init.normal_(self.out_cls_conv.kernel, std=.01)
        nn.init.constant_(self.out_cls_conv.bias, bias_init_with_prob(.01))

        self.conv_blocks = nn.Sequential(
            *[
                self._make_block(input_dim, input_dim)
                for _ in range(n_spconv2d)
            ]
        )

        self.z_fusion_block = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim)
        )

        self.voxel_size = voxel_size

    def _make_block(self, in_ch, out_ch):
        conv = ME.MinkowskiConvolution(in_ch,
                                       out_ch,
                                       kernel_size=3,
                                       stride=1,
                                       dimension=2)
        batch_norm = ME.MinkowskiBatchNorm(out_ch)
        nn.init.constant_(batch_norm.bn.weight, 1)
        nn.init.constant_(batch_norm.bn.bias, 0)
        ME.utils.kaiming_normal_(
            conv.kernel, mode='fan_out', nonlinearity='relu')

        act = ME.MinkowskiReLU(inplace=True)
        return nn.Sequential(
            conv, batch_norm, act
        )

    def project_3d_to_2d(self, input_tensor):
        coords = input_tensor.coordinates
        feats = input_tensor.features
        coords_2d = coords[:, [0, 1, 2]]
        z_coord = coords[:, 3].float().reshape(-1, 1)
        z_feats = self.z_fusion_block(z_coord * self.voxel_size)
        fused_feats = feats + z_feats
        output_tensor = ME.SparseTensor(
            features=fused_feats,
            coordinates=coords_2d,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=input_tensor.device
        )

        return output_tensor

    def forward(self, x, ceiling):
        x_proj = self.project_3d_to_2d(x)
        x_proj = self.conv_blocks(x_proj)
        cls_out = self.out_cls_conv(x_proj)

        z_feat = ceiling[x_proj.coordinates[:, 0]]
        z_feat = self.add_feats_encoder(z_feat)

        x_proj_w_z_feat = ME.SparseTensor(
            features=torch.hstack((x_proj.features, z_feat)),
            coordinate_manager=x_proj.coordinate_manager,
            coordinates=x_proj.coordinates,
            device=x_proj.device
        )

        reg_out = self.out_reg_conv(x_proj_w_z_feat)
        return cls_out.features, reg_out.features, x_proj_w_z_feat


@MODELS.register_module()
class TUN3DHead(Base3DDenseHead):
    r"""Bbox head of `TUN3D <link>`. TODO: paste the link of paper

    Args:
        in_channels (int): Number of channels in input tensors.
        num_reg_outs (int): Number of regression layer channels.
        voxel_size (float): Voxel size in meters.
        pts_center_threshold (int): Box to location assigner parameter.
            After feature level for the box is determined, assigner selects
            pts_center_threshold locations closest to the box center.
        bbox_loss (dict): Config of bbox loss. Defaults to
            dict(type='AxisAlignedIoULoss', mode='diou', reduction=None).
        cls_loss (dict): Config of classification loss. Defaults to
            dict = dict(type='mmdet.FocalLoss', reduction=None).
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 datasets: List,
                 datasets_classes: List[List[str]],
                 angles: List,
                 datasets_weights: List,
                 in_channels: int,
                 num_reg_outs: int,
                 voxel_size: int,
                 pts_center_threshold: int,
                 pts_center_threshold_layout: int,
                 label2level: Tuple[int],
                 loss_weights: List,
                 layout_level: int,
                 layout_head=None,
                 layout_loss: dict = dict(
                     type='L1Loss',
                     reduction='none'),
                 bbox_loss: dict = dict(
                     type='TUN3DAxisAlignedIoULoss',
                     mode='diou',
                     reduction='none'),
                 bbox_rotated_loss: dict = dict(
                     type='TUN3DRotatedIoU3DLoss',
                     mode='diou',
                     reduction='none'),
                 cls_loss: dict = dict(
                     type='mmdet.FocalLoss', reduction='none'),
                 cls_layout_loss: dict = dict(
                     type='mmdet.FocalLoss', reduction='none'),
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(TUN3DHead, self).__init__(init_cfg)
        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to ' \
                'install MinkowskiEngine.`'  
            )
        self.datasets = datasets
        self.angles = angles
        self.dataset_weights = datasets_weights
        self.voxel_size = voxel_size
        self.pts_center_threshold = pts_center_threshold
        self.pts_center_threshold_layout = pts_center_threshold_layout
        self.label2level = label2level
        self.layout_level = layout_level
        self.loss_weights = loss_weights
        self.layout_loss = MODELS.build(layout_loss)
        self.cls_layout_loss = MODELS.build(cls_layout_loss)
        self.bbox_loss = MODELS.build(bbox_loss)
        self.bbox_rotated_loss = MODELS.build(bbox_rotated_loss)
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.layout_head = MODELS.build(layout_head)

        n_q_feats = self.layout_head.n_q_feats
        if n_q_feats == 1:
            self.q_values = torch.tensor([0.995])
        else:
            self.q_values = torch.linspace(0, 1, n_q_feats)
            self.q_values[0] = 0.005
            self.q_values[-1] = 0.995

        unique_cls = sorted(list(set(itertools.chain.from_iterable(
                            datasets_classes))))
        self.datasets_cls_idxs = []
        for dataset_classes in datasets_classes:
            dataset_cls_idxs = []
            for cls in dataset_classes:
                dataset_cls_idxs.append(unique_cls.index(cls))
            self.datasets_cls_idxs.append(dataset_cls_idxs)

        self._init_layers(len(unique_cls), in_channels, num_reg_outs)

    def _init_layers(self, num_classes: int, in_channels: int,
                     num_reg_outs: int):
        """Build detection and layout heads.

        Initializes convolutional layers for bounding box regression,
        classification, and head for layout estimation.

        Args:
            num_classes (int): Number of classes for classification.
            in_channels (int): Input feature channels.
            num_reg_outs (int): Output channels for box regression.
            num_layout_reg_outs (int): Output channels for layout regression.
            n_conv (int): Number of 2D sparse convolutions used in the layout head.
        """

        # Detection heads
        self.conv_reg = ME.MinkowskiConvolution(
            in_channels, num_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.conv_cls = ME.MinkowskiConvolution(
            in_channels, num_classes, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.conv_reg.kernel, std=.01)
        nn.init.normal_(self.conv_cls.kernel, std=.01)
        nn.init.constant_(self.conv_cls.bias, bias_init_with_prob(.01))

    def get_dataset(self, lidar_path):
        for dataset in self.datasets:
            if dataset in lidar_path.split('/'):
                return dataset

    def get_dataset_idx(self, lidar_path):
        return self.datasets.index(self.get_dataset(lidar_path))

    def _forward_single(self, 
                        x: SparseTensor, 
                        batch_input_metas, 
                        is_layout: bool) -> Tuple[Tensor, ...]:
        """Forward pass for one feature level.

        Performs forward pass through model and computes 3D bounding boxes, 
        its class scores, and optionally layout predictions.

        Args:
            x (SparseTensor): Per level neck output tensor.
            batch_input_metas (list[dict]): Sample metadata.
            is_layout (bool): If True, compute layout predictions.

        Returns:
            - bbox_preds (list[Tensor]): Per-sample box predictions.
            - cls_preds (list[Tensor]): Per-sample class predictions.
            - layout_preds (list[Tensor | list]): Layout regressions or
            empty lists if ``is_layout=False``.
            - cls_layout_preds (list[Tensor | list]): Layout class scores
            or empty lists if ``is_layout=False``.
            - points (list[Tensor]): Point coordinates.
            - points_layout (list[Tensor | list]): Layout point coordinates
            or empty lists if ``is_layout=False``.
        """

        reg_final = self.conv_reg(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle),
                              dim=1)
        cls_pred = self.conv_cls(x).features

        bbox_preds, cls_preds, layout_preds, cls_layout_preds, points, \
            points_layout = [], [], [], [], [], []
        for i, permutation in enumerate(x.decomposition_permutations):
            bbox_preds.append(bbox_pred[permutation])
            merged_cls_preds = cls_pred[permutation]
            dataset_idx = self.get_dataset_idx(
                batch_input_metas[i]['lidar_path'])
            cls_idxs = self.datasets_cls_idxs[dataset_idx]
            cls_preds.append(merged_cls_preds[..., cls_idxs])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        if is_layout:
            ceiling = x.features.new_zeros(
                (len(batch_input_metas), self.q_values.shape[0]))
            for i in range(len(batch_input_metas)):
                pcd = batch_input_metas[i]['points']
                ceiling[i] = torch.quantile(pcd[..., 2], q=self.q_values)

            cls_pred_layout, reg_layout, x_layout = self.layout_head(
                x, ceiling=ceiling)
            height = torch.exp(reg_layout[..., 4])
            reg_final_layout = torch.cat((reg_layout[..., :4], height.unsqueeze(1)),
                                         dim=1)
            for permutation in x_layout.decomposition_permutations:
                layout_preds.append(reg_final_layout[permutation])
                cls_layout_preds.append(cls_pred_layout[permutation])
                points_layout.append(
                    x_layout.coordinates[permutation][:, 1:] * self.voxel_size)
        else:
            for _ in range(len(x.decomposition_permutations)):
                layout_preds.append([])
                cls_layout_preds.append([])
                points_layout.append([])

        return bbox_preds, cls_preds, layout_preds, cls_layout_preds, points, points_layout

    def forward(self, x: List[Tensor], batch_input_metas: List[dict]) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Args:
            x (list[Tensor]): Features from the backbone.
            batch_input_metas (list[dict]): Sample metadata.

        Returns:
            Tuple[List[Tensor], ...]: Predictions of the head.
        """
        bbox_preds, cls_preds, layout_preds, cls_layout_preds, points, \
            points_layout = [], [], [], [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, layout_pred, cls_layout_pred, point, point_layout = (
                self._forward_single(
                    x[i],
                    batch_input_metas,
                    is_layout=i == self.layout_level)
            )
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
            layout_preds.append(layout_pred)
            cls_layout_preds.append(cls_layout_pred)
            points_layout.append(point_layout)
        return bbox_preds, cls_preds, layout_preds, cls_layout_preds, \
                points, points_layout

    def loss(self, x: Tuple[Tensor], batch_data_samples,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x, batch_input_metas)

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (batch_gt_instances_3d, batch_input_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def _loss_by_feat_single(self, bbox_preds: List[Tensor],
                             cls_preds: List[Tensor], 
                             layout_preds: List[Tensor], 
                             cls_layout_preds: List[Tensor], 
                             points: List[Tensor], 
                             points_layout: List[Tensor],
                             gt_bboxes: BaseInstance3DBoxes, 
                             gt_labels: Tensor,
                             gt_layout: Tensor, 
                             layout_labels: Tensor,
                             input_meta: dict) -> Tuple[Tensor, ...]:
        """Loss function of single sample.

        Args:
            bbox_preds (list[Tensor]): Box predictions from all levels.
            cls_preds (list[Tensor]): Classification predictions from all levels.
            layout_preds (list[Tensor]): Layout regression predictions.
            cls_layout_preds (list[Tensor]): Layout classification predictions.
            points (list[Tensor]): Final 3D locations for all levels (bboxes).
            points_layout (list[Tensor]): Final 2D locations for all levels (layout).
            gt_bboxes (BaseInstance3DBoxes): Ground-truth 3D boxes.
            gt_labels (Tensor): Ground-truth class labels.
            gt_layout (Tensor): Ground-truth layout annotations.
            layout_labels (Tensor): Ground-truth layout class labels.
            input_meta (dict): Scene metadata.

        Returns:
            tuple[Tensor, ...]: Regression and classification loss
                values for objects/layout and a boolean mask of assigned points.
        """

        num_classes = cls_preds[0].shape[1]
        num_classes_layout = cls_layout_preds[self.layout_level].shape[1]
        dataset_idx = self.get_dataset_idx(input_meta['lidar_path'])
        dataset_loss_weight = self.dataset_weights[dataset_idx]
        det_weight = self.loss_weights[dataset_idx][0] * dataset_loss_weight
        layout_weight = self.loss_weights[dataset_idx][1] * dataset_loss_weight

        bbox_targets, cls_targets = self.get_targets(points, gt_bboxes,
                                                     gt_labels, num_classes, dataset_idx)
        layout_preds = layout_preds[self.layout_level]
        cls_layout_preds = cls_layout_preds[self.layout_level]
        points_for_layout = points_layout[self.layout_level]
        layout_targets, matched_layout_preds, cls_layout_targets = (
            self.get_targets_layout_2d(
                points_for_layout, 
                gt_layout, 
                layout_labels, 
                layout_preds)
        )

        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        cls_bbox_loss = det_weight * \
            self.cls_loss(cls_preds, cls_targets).sum(dim=-1, keepdim=True)
        if len(layout_labels) > 0:
            cls_layout_loss = layout_weight * \
                self.cls_layout_loss(cls_layout_preds, cls_layout_targets)
        else:
            cls_layout_loss = torch.zeros(cls_layout_preds.shape[0], 1).to(points.device)
            cls_layout_loss.requires_grad = True

        # bbox loss
        pos_mask = cls_targets < num_classes
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            pos_bbox_targets = bbox_targets[pos_mask]
            if not self.angles[dataset_idx]:
                bbox_loss_fn = self.bbox_loss
                pos_bbox_preds = pos_bbox_preds[..., :6]
            else:
                bbox_loss_fn = self.bbox_rotated_loss
            bbox_loss = det_weight * bbox_loss_fn(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = pos_bbox_preds

        pos_layout_mask = cls_layout_targets < num_classes_layout
        pos_layout_preds = matched_layout_preds[pos_layout_mask]
        if pos_layout_mask.sum() > 0:
            pos_points_layout = points_for_layout[pos_layout_mask]
            pos_layout_preds = matched_layout_preds[pos_layout_mask]
            pos_layout_targets = layout_targets[pos_layout_mask]
            gt_targets = self._quad_to_bev_loss(pos_points_layout, 
                                                pos_layout_targets)
            layout_loss = layout_weight * \
                self.layout_loss(pos_layout_preds, gt_targets)

        else:
            layout_loss = pos_layout_preds

        return bbox_loss, cls_bbox_loss, layout_loss, cls_layout_loss, \
                pos_mask, pos_layout_mask

    def loss_by_feat(self,
                     bbox_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     layout_preds: List[List[Tensor]],
                     cls_layout_preds: List[List[Tensor]],
                     points: List[List[Tensor]],
                     points_layout,
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:
        """Loss function about feature.

        Args:
           bbox_preds (list[list[Tensor]]): Bounding box predictions for
            each level and batch sample.
            cls_preds (list[list[Tensor]]): Classification predictions for
                each level and batch sample.
            layout_preds (list[list[Tensor]]): Layout regression predictions
                for each level and batch sample.
            cls_layout_preds (list[list[Tensor]]): Layout classification
                predictions for each level and batch sample.
            points (list[list[Tensor]]): Final 3D location coordinates for all scenes. 
            The first list contains predictions from different levels. 
            The second list contains predictions in a mini-batch.
            points_layout (list[list[Tensor]]): Final 2D location coordinates for all scenes. 
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each scene.

        Returns:
            dict: Regression and classification losses for objects/layout.
        """
        bbox_losses, cls_bbox_losses, layout_losses, cls_layout_losses, \
            pos_masks, pos_layout_masks = [], [], [], [], [], []
        for i in range(len(batch_input_metas)):
            bbox_loss, cls_bbox_loss, layout_loss, cls_layout_loss, pos_mask, pos_layout_mask = (
                self._loss_by_feat_single(
                    bbox_preds=[x[i] for x in bbox_preds],
                    cls_preds=[x[i] for x in cls_preds],
                    layout_preds=[x[i] for x in layout_preds],
                    cls_layout_preds=[x[i] for x in cls_layout_preds],
                    points=[x[i] for x in points],
                    points_layout=[x[i] for x in points_layout],
                    gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                    gt_labels=batch_gt_instances_3d[i].labels_3d,
                    gt_layout=batch_gt_instances_3d[i].gt_layout,
                    layout_labels=batch_gt_instances_3d[i].gt_labels_3d_layout,
                    input_meta=batch_input_metas[i])
            )

            if len(bbox_loss) > 0:
                bbox_losses.append(bbox_loss)
            cls_bbox_losses.append(cls_bbox_loss)
            pos_masks.append(pos_mask)
            if len(layout_loss) > 0:
                layout_losses.append(layout_loss)
                cls_layout_losses.append(cls_layout_loss)
                pos_layout_masks.append(pos_layout_mask)
        
        return dict(
            bbox_loss=torch.mean(torch.cat(bbox_losses)) \
                if len(bbox_losses) > 0 else torch.zeros(1).to(cls_bbox_losses[0].device),
            cls_bbox_loss=torch.sum(torch.cat(cls_bbox_losses)) / torch.sum(torch.cat(pos_masks)) \
                if len(cls_bbox_losses) > 0 else torch.zeros(1).to(cls_bbox_losses[0].device),
            layout_loss=torch.mean(torch.cat(layout_losses)) \
                if len(layout_losses) > 0 else torch.zeros(1).to(cls_bbox_losses[0].device),
            cls_layout_loss=torch.sum(torch.cat(cls_layout_losses)) / torch.sum(torch.cat(pos_layout_masks)) \
                if len(pos_layout_masks) > 0 else torch.zeros(1).to(cls_bbox_losses[0].device))

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x, batch_input_metas)
        predictions = self.predict_by_feat(
            *outs, batch_input_metas=batch_input_metas, rescale=rescale)
        return predictions

    def _predict_by_feat_single(self, bbox_preds: List[Tensor],
                                cls_preds: List[Tensor], points: List[Tensor],
                                input_meta: dict) -> InstanceData:
        """Generate boxes for single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            input_meta (dict): Scene meta info.

        Returns:
            InstanceData: Predicted bounding boxes, scores and labels.
        """
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        dataset = self.get_dataset(input_meta.get('lidar_path'))
        dataset_idx = self.datasets.index(dataset)
        if not self.angles[dataset_idx]:
            bbox_preds = bbox_preds[..., :6]
        bboxes = self._bbox_pred_to_bbox(points, bbox_preds)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, input_meta)

        if 'sp_pts_mask' not in input_meta:
            bboxes = input_meta['box_type_3d'](
                bboxes,
                box_dim=bboxes.shape[1],
                with_yaw=bboxes.shape[1] == 7,
                origin=(.5, .5, .5))
            results = dict(bboxes_3d=bboxes,
                           labels_3d=labels, scores_3d=scores)
        else:
            sp_pts_mask = input_meta['sp_pts_mask'].to(bboxes.device)
            pcd = input_meta['points'][..., :3].to(bboxes.device)
            results = self.trim_bboxes_by_superpoints(
                sp_pts_mask, pcd, bboxes, labels, scores)
        return results

    def trim_bboxes_by_superpoints(self, sp_pts_mask, point,
                                   bboxes, labels, scores):
        """Trim bounding boxes based on superpoint masks.

        Args:
            sp_pts_mask (Tensor): A boolean tensor indicating the valid points 
                for each superpoint.
            point (Tensor): A tensor of shape (n_points, 3) representing the 
                3D coordinates of the points.
            bboxes (Tensor): A tensor of predicted bounding boxes, with shape 
                (n_boxes, 6) or (n_boxes, 7) if yaw is included.
            labels (Tensor): A tensor of shape (n_boxes,) containing the 
                predicted labels for each bounding box.
            scores (Tensor): A tensor of shape (n_boxes,) containing the 
                classification scores for each bounding box.

        Returns:
            List[Tuple[DepthInstance3DBoxes, Tensor, Tensor]]: A list 
                containing a tuple of trimmed bounding boxes, 
                labels, and scores.
        """
        n_points = point.shape[0]
        n_boxes = bboxes.shape[0]
        point = point.unsqueeze(1).expand(n_points, n_boxes, 3)
        if bboxes.shape[1] == 6:
            bboxes = torch.cat(
                (bboxes, torch.zeros_like(bboxes[:, :1])),
                dim=1)
        bboxes = bboxes.unsqueeze(0).expand(n_points, n_boxes,
                                            bboxes.shape[1])
        face_distances = get_face_distances(point, bboxes)

        inside_bbox = face_distances.min(dim=-1).values > 0
        inside_bbox = inside_bbox.T
        sp_inside = scatter_mean(inside_bbox.float(),
                                 sp_pts_mask, dim=-1)
        sp_del = sp_inside < self.test_cfg.low_sp_thr
        inside_bbox[sp_del[:, sp_pts_mask]] = False

        sp_add = sp_inside > self.test_cfg.up_sp_thr
        inside_bbox[sp_add[:, sp_pts_mask]] = True

        points_for_max = point.clone()
        points_for_min = point.clone()
        points_for_max[~inside_bbox.T.bool()] = float('-inf')
        points_for_min[~inside_bbox.T.bool()] = float('inf')
        bboxes_max = points_for_max.max(axis=0)[0]
        bboxes_min = points_for_min.min(axis=0)[0]
        bboxes_sizes = bboxes_max - bboxes_min
        bboxes_centers = (bboxes_max + bboxes_min) / 2
        bboxes = torch.hstack((bboxes_centers, bboxes_sizes))
        bboxes = DepthInstance3DBoxes(bboxes, with_yaw=False,
                                      box_dim=6, origin=(0.5, 0.5, 0.5))

        return dict(bboxes_3d=bboxes, labels_3d=labels, scores_3d=scores)

    def _predict_by_feat_single_layout(self, 
                                       layout_preds, 
                                       cls_layout_preds, 
                                       points, 
                                       input_meta=None):
        scores = torch.nn.functional.sigmoid(
            cls_layout_preds[self.layout_level])[:, 0]
        points = points[self.layout_level]
        layout_preds = layout_preds[self.layout_level]

        layout_preds = self._layout_rel_to_abs(points, layout_preds)

        if len(scores) > self.test_cfg.nms_pre_layout > 0:
            _, ids = scores.topk(self.test_cfg.nms_pre_layout)
            layout_preds = layout_preds[ids]
            scores = scores[ids]

        nms_quads, nms_scores = self._layout_nms(layout_preds, scores)

        if self.test_cfg.enable_double_layout_nms:
            nms_quads, nms_scores = self._layout_nms_bboxes(nms_quads, nms_scores)

        result = dict(layout_verts=nms_quads)
        return result

    def predict_by_feat(self, 
                        bbox_preds: List[List[Tensor]], 
                        cls_preds,
                        layout_preds, cls_layout_preds,
                        points: List[List[Tensor]],
                        points_layout,
                        batch_input_metas: List[dict],
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[InstanceData]: Predicted bboxes, scores, and labels for
            all scenes.
        """
        results = []
        for i in range(len(batch_input_metas)):
            result_detection = self._predict_by_feat_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=batch_input_metas[i])
            result_layout = self._predict_by_feat_single_layout(
                layout_preds=[x[i] for x in layout_preds],
                cls_layout_preds=[x[i] for x in cls_layout_preds],
                points=[x[i] for x in points_layout],
                input_meta=batch_input_metas[i]
            )
            result = InstanceData_()
            result.bboxes_3d = result_detection['bboxes_3d']
            result.labels_3d = result_detection['labels_3d']
            result.scores_3d = result_detection['scores_3d']
            layout_verts = result_layout['layout_verts']
            result.layout_verts = layout_verts
            result.points = batch_input_metas[i]['points']
            results.append(result)
        return results

    def get_dataset(self, lidar_path):
        for dataset in self.datasets:
            if dataset in lidar_path.split('/'):
                return dataset

    @staticmethod
    def _layout_rel_to_abs(points, quad_pred):
        """Transform relative layout parameters to absolute coordinates of layout verts.

        Args:
            points (Tensor): Final locations of shape (N, 2)
            bbox_pred (Tensor): Predicted layout parameters of shape (N, 5)

        Returns:
            Tensor: Transformed layout absolute coordinates of shape (N, 4, 3).
        """
        points = points.unsqueeze(1)
        height = quad_pred[..., 4]
        quad_pred = quad_pred[..., :4].reshape(shape=(-1, 2, 2))
        final_pred = quad_pred + points
        final_pred_quads_2d = torch.cat([final_pred, final_pred], dim=1)
        floor_ceiling_tensor = torch.zeros(
            quad_pred.shape[0], 4, 1).to(quad_pred.device)
        floor_ceiling_tensor[:, 2:, :] = height[:, None, None]
        return torch.cat([final_pred_quads_2d, floor_ceiling_tensor], dim=-1)

    @staticmethod
    def _quad_to_bev_loss(points, quads):
        """Transform absolute coordinates of layout verts to 5-param representation.

        Args:
            points (Tensor): Final locations of shape (N, 2)
            quads (Tensor): Layout absolute coordinates of shape (N, 4, 3)

        Returns:
            Tensor: Transformed layout representation of shape (N, 5).
        """
        points = points.unsqueeze(1)
        height = 0.5 * (quads[:, 2, 2] + quads[:, 3, 2])
        quads_2d = quads[..., :-1]
        quads_2d_rel = quads_2d - points
        quad_2d_vert_0 = 0.5 * (quads_2d_rel[:, 0, :] + quads_2d_rel[:, 2, :])
        quad_2d_vert_1 = 0.5 * (quads_2d_rel[:, 1, :] + quads_2d_rel[:, 3, :])
        return torch.cat([quad_2d_vert_0, quad_2d_vert_1, height.unsqueeze(1)], dim=-1)

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center, y_center, z_center, bbox_pred[:, 3], bbox_pred[:, 4],
            bbox_pred[:, 5]
        ], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    @torch.no_grad()
    def get_targets(self, 
                    points: Tensor, 
                    gt_bboxes: BaseInstance3DBoxes,
                    gt_labels: Tensor, 
                    num_classes: int, 
                    dataset_idx: int) -> Tuple[Tensor, ...]:
        """Compute object targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            num_classes (int): Number of classes.

        Returns:
            tuple[Tensor, ...]: Regression and classification targets for all
                locations.
        """
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([
            points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
            for i in range(len(points))
        ])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if len(gt_labels) == 0:
            return points.new_tensor([]), \
                gt_labels.new_full((n_points,), num_classes)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                          dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level[dataset_idx])
        label_levels = label2level[gt_labels].unsqueeze(0).expand(
            n_points, n_boxes)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(level_condition, center_distances,
                                       float_max)
        topk_distances = torch.topk(
            center_distances,
            min(self.pts_center_threshold + 1, len(center_distances)),
            largest=False,
            dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances,
                                       float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)

        bbox_targets = boxes[0][min_inds]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = torch.where(min_inds >= 0, gt_labels[min_inds],
                                  num_classes)
        return bbox_targets, cls_targets

    @torch.no_grad()
    def get_targets_layout_2d(self, 
                              points: Tensor, 
                              gt_layout: Tensor, 
                              gt_labels: Tensor, 
                              layout_preds: Tensor):
        """Compute layout targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_layout (Tensor): Ground truth layout predictions.
            gt_labels (Tensor): Ground truthlayout  labels.
            num_classes (int): Number of classes.

        Returns:
            tuple[Tensor, ...]: Regression and classification targets for all locations.
        """
        n_points = len(points)
        if len(gt_layout) == 0:
            return points.new_tensor([]), layout_preds, \
                gt_labels.new_full((n_points,), 1)
        float_max = points[0].new_tensor(1e8)
        n_quads = len(gt_layout)
        quad_centers = gt_layout.mean(dim=1)[..., :-1]
        quad_centers = quad_centers.expand(
            n_points, n_quads, quad_centers.shape[1])
        points = points.unsqueeze(1).expand(n_points, n_quads, 2)

        center_distances = torch.sum(
            torch.pow(quad_centers - points, 2), dim=-1)

        topk_distances = torch.topk(
            center_distances,
            min(self.pts_center_threshold_layout + 1, len(center_distances)),
            largest=False,
            dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)
        center_distances = torch.where(topk_condition, center_distances,
                                       float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds_targets = torch.where(min_values < float_max, min_ids, -1)
        layout_targets = gt_layout[min_inds_targets]
        cls_targets = torch.where(min_inds_targets >= 0, 0, 1)

        return layout_targets, layout_preds, cls_targets

    def _single_scene_multiclass_nms(self, 
                                     bboxes: Tensor, 
                                     scores: Tensor,
                                     input_meta: dict) -> Tuple[Tensor, ...]:
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor, ...]: Predicted bboxes, scores and labels.
        """
        num_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(num_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if with_yaw:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if not with_yaw:
            nms_bboxes = nms_bboxes[:, :6]

        return nms_bboxes, nms_scores, nms_labels

    def _layout_nms(self, layout_verts, scores):
        """Perform non-maximum suppression (NMS) on layout quads.

        Filters overlapping layout predictions based on their scores and a
        distance threshold defined in ``self.test_cfg.nms_radius``.

        Args:
            layout_verts (Tensor): Predicted layout quads of shape (N, 4, 3).
            scores (Tensor): Confidence scores for each quad, shape (N,).

        Returns:
            tuple:
                - nms_quads (list[Tensor]): Filtered layout quads after NMS.
                - nms_scores (list[Tensor]): Corresponding scores of the filtered quads.
        """
        nms_quads = []
        nms_scores = []
        ids = scores > self.test_cfg.score_thr_layout
        if not ids.any():
            return [], []
        scores = scores[ids]
        layout_verts = layout_verts[ids]

        idxs = torch.argsort(scores, descending=True)
        layout_verts = layout_verts[idxs]

        nms_quads.append(layout_verts[0])
        nms_scores.append(scores[0])
        for i in range(1, layout_verts.shape[0]):
            quad_to_check = layout_verts[i]
            flag = True
            for quad in nms_quads:
                if self._same_quad_with_permutations(quad, quad_to_check):
                    flag = False
                    break
            if flag:
                nms_quads.append(quad_to_check)
                nms_scores.append(scores[i])

        return nms_quads, nms_scores

    def _layout_nms_bboxes(self, layout_verts, scores):
        """Apply 3D bounding box NMS to layout quads.

        Converts layout quads to 3D bounding boxes and suppresses 
        overlapping predictions.

        Args:
            layout_verts (Tensor): Predicted layout quads, shape (N, 4, 3).
            scores (list[Tensor]): Confidence scores for each quad.

        Returns:
            tuple:
                - nms_quads (list[Tensor]): Filtered layout quads after NMS.
                - nms_scores (list[float]): Corresponding scores of filtered quads.
        """

        if not len(layout_verts):
            return [], []
        scores = torch.stack(scores)
        layout_verts = torch.stack(layout_verts)
        nms_idxs = []
        nms_corners = []
        idxs = torch.argsort(scores, descending=True)
        layout_verts = layout_verts[idxs]
        scores = scores[idxs]
        quad2corners = LayoutToBBoxes()
        input_dict = {'layout_verts': layout_verts.to('cpu')}
        all_corners = quad2corners.transform(input_dict)['gt_bboxes_3d'].corners
        all_corners = all_corners.to(layout_verts.device)
        nms_idxs.append(0)
        nms_corners.append(all_corners[0])
        for i in range(1, layout_verts.shape[0]):
            quad_to_check = all_corners[i]
            flag = True
            for quad in nms_corners:
                if iou_3d_from_vertices(quad, quad_to_check) > self.test_cfg.iou_thr_layout:
                    flag = False
                    break
            if flag:
                nms_idxs.append(i)
                nms_corners.append(quad_to_check)

        nms_quads = layout_verts[nms_idxs]
        nms_scores = scores[nms_idxs]
        return [v for v in nms_quads], list(torch.unbind(nms_scores))

    def _same_quad_with_permutations(self, quad_1, quad_2):
        """Check if two layout quads are the same within a distance threshold.

        Args:
            quad_1 (Tensor): First quad vertices, shape (4, 3).
            quad_2 (Tensor): Second quad vertices, shape (4, 3).
            radius (float): Maximum allowed distance between corresponding vertices.

        Returns:
            bool: True if all vertex distances are below the radius, False otherwise.
        """
        vert_dist = torch.linalg.norm(quad_1 - quad_2, dim=-1)
        return (vert_dist < self.test_cfg.nms_radius).all()


def get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
    """Calculate distances from point to box faces.

    Args:
        points (Tensor): Final locations of shape (N_points, N_boxes, 3).
        boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

    Returns:
        Tensor: Face distances of shape (N_points, N_boxes, 6),
        (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
    """
    # If boxes have 6 dimensions, add a zero in the 7th column
    if boxes.shape[-1] == 6:
        boxes = torch.cat([boxes, torch.zeros(
            *boxes.shape[:-1], 1, dtype=boxes.dtype, device=boxes.device)], dim=-1)
    shift = torch.stack(
        (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
            points[..., 2] - boxes[..., 2]),
        dim=-1).permute(1, 0, 2)
    shift = rotation_3d_in_axis(
        shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
    centers = boxes[..., :3] + shift
    dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
    dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
    dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
    dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
    dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
    dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
    return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                       dim=-1)


def iou_3d_from_vertices(box1_vertices: torch.Tensor, box2_vertices: torch.Tensor) -> float:
    """
    Compute IoU between two axis-aligned 3D bounding boxes from their vertices using PyTorch.

    Parameters:
        box1_vertices (torch.Tensor): shape (8, 3), vertices [(x,y,z), ...]
        box2_vertices (torch.Tensor): shape (8, 3), vertices [(x,y,z), ...]

    Returns:
        float: IoU value between 0 and 1
    """
    v1 = box1_vertices.float()
    v2 = box2_vertices.float()

    x1_min, y1_min, z1_min = v1.min(dim=0).values
    x1_max, y1_max, z1_max = v1.max(dim=0).values

    x2_min, y2_min, z2_min = v2.min(dim=0).values
    x2_max, y2_max, z2_max = v2.max(dim=0).values

    x_overlap = min(x1_max.item(), x2_max.item()) - max(x1_min.item(), x2_min.item())
    x_overlap = max(0.0, x_overlap)
    y_overlap = min(y1_max.item(), y2_max.item()) - max(y1_min.item(), y2_min.item())
    y_overlap = max(0.0, y_overlap)
    z_overlap = min(z1_max.item(), z2_max.item()) - max(z1_min.item(), z2_min.item())
    z_overlap = max(0.0, z_overlap)

    inter_vol = x_overlap * y_overlap * z_overlap

    vol1 = (x1_max - x1_min).item() * (y1_max - y1_min).item() * (z1_max - z1_min).item()
    vol2 = (x2_max - x2_min).item() * (y2_max - y2_min).item() * (z2_max - z2_min).item()

    union_vol = vol1 + vol2 - inter_vol

    if union_vol == 0.0:
        return 0.0

    return inter_vol / union_vol
