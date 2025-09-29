# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch

from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import RandomFlip3D, GlobalRotScaleTrans
from mmdet3d.datasets import PointSample
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.structures.points import BasePoints, DepthPoints


@TRANSFORMS.register_module()
class TUN3DPointSample(PointSample):
    """The only difference with PointSample is the support of float num_points
    parameter.

    In this case we sample random fraction of points from num_points to 100%
    points. These classes should be merged in the future.
    """

    def _points_random_sampling(
        self,
        points: BasePoints,
        num_samples: Union[int, float],
        sample_range: Optional[float] = None,
        replace: bool = False,
        return_choices: bool = False
    ) -> Union[Tuple[BasePoints, np.ndarray], BasePoints]:
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool): Sampling with or without replacement.
                Defaults to False.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if isinstance(num_samples, float):
            assert num_samples < 1
            num_samples = int(
                np.random.uniform(self.num_points, 1.) * points.shape[0])
        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            dist = np.linalg.norm(points.coord.numpy(), axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(
                    far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]


@TRANSFORMS.register_module()
class RandomFlip3DLayout(RandomFlip3D):
    """Randomly flip 3D points and layout quads for data augmentation.

    Extends `RandomFlip3D` to handle layout vertices by applying the same
    flip transformation to layout quads as to the point cloud.
    """

    def call_parent(self, input_dict):
        return super(RandomFlip3D, self).__call__(input_dict)

    def __call__(self, input_dict):
        layout_verts = torch.FloatTensor(input_dict['layout_verts'])
        if not layout_verts.numel():
            return self.call_parent(input_dict)

        init_shape = layout_verts.shape
        points = input_dict['points'].tensor
        layout_verts = layout_verts.reshape((-1, 3))
        layout_verts = torch.hstack((layout_verts, torch.ones_like(layout_verts)))
        N = layout_verts.shape[0]
        points = DepthPoints(torch.vstack((points, layout_verts)), points_dim=6)
        input_dict['points'] = points
        res_dict = self.call_parent(input_dict)
        res_points = res_dict['points'].tensor
        res_dict['layout_verts'] = res_points[-N:, :3].reshape(init_shape)
        res_dict['points'] = DepthPoints(res_points[:-N, :3])
        res_dict['points'].color = res_points[:-N, 3:]
        return res_dict


@TRANSFORMS.register_module()
class GlobalRotScaleTransLayout(GlobalRotScaleTrans):
    """Apply global rotation, scaling, and translation to 3D points and layout quads.

    Extends `GlobalRotScaleTrans` to also transform layout vertices consistently
    with the point cloud during data augmentation.
    """

    def call_parent(self, input_dict):
        return super(GlobalRotScaleTrans, self).__call__(input_dict)

    def __call__(self, input_dict):
        layout_verts = torch.FloatTensor(input_dict['layout_verts'])
        if not layout_verts.numel():
            return self.call_parent(input_dict)
        init_shape = layout_verts.shape
        points = input_dict['points'].tensor
        layout_verts = layout_verts.reshape((-1, 3))
        layout_verts = torch.hstack((layout_verts, torch.ones_like(layout_verts)))
        N = layout_verts.shape[0]
        points = DepthPoints(torch.vstack((points, layout_verts)), points_dim=6)
        input_dict['points'] = points
        res_dict = self.call_parent(input_dict)
        res_points = res_dict['points'].tensor
        res_dict['layout_verts'] = res_points[-N:, :3].reshape(init_shape)
        res_dict['points'] = DepthPoints(res_points[:-N, :3])
        res_dict['points'].color = res_points[:-N, 3:]
        return res_dict


@TRANSFORMS.register_module()
class LayoutOrientation(BaseTransform):
    """Ensure correct orientation of layout vertices.

    Adjusts floor and ceiling vertex ordering to maintain a consistent orientation of layout quads.

    Example of right vertex ordering:

    2 ----- 3 (z = z_max)
    |       |
    |       |
    |       |
    0 ----- 1 (z = 0)

   - Vertices `0` and `1` are always on the floor.  
   - Vertices `2` and `3` are always on the ceiling.  
   - Vertex `2` is directly above vertex `0`, and vertex `3` is directly above vertex `1`.
    """

    def transform(self, input_dict):
        if isinstance(input_dict, dict):
            return self.orientate(input_dict)
        elif isinstance(input_dict, list):
            for i in range(len(input_dict)):
                input_dict[i] = self.orientate(input_dict[i])
            return input_dict
        else:
            raise Exception('Wrong input type in LayoutOrientation transform!')

    def orientate(self, input_dict):
        layout_verts = torch.tensor(input_dict['layout_verts'])
        if not layout_verts.numel():
            return input_dict
        quad_centers = layout_verts.mean(dim=1)

        points = input_dict['points']
        pc_center = points.coord.mean(dim=0).reshape((1, 3))
        view_dirs = pc_center - quad_centers
        z = layout_verts[:, :, 2]
        indices = torch.sort(z, dim=1)[1]
        arange_idxs = torch.arange(layout_verts.shape[0])[:, None]
        sorted_layout_verts = layout_verts[arange_idxs, indices, :]
        floor_verts = sorted_layout_verts[:, :2, :]
        ceiling_verts = sorted_layout_verts[:, 2:, :]

        floor_vectors = floor_verts - quad_centers[:, None, :]
        cross_floor = torch.cross(floor_vectors[:, 0, :], 
                                  floor_vectors[:, 1, :], dim=1)  # cross product
        dot_floor = torch.sum(cross_floor * view_dirs, dim=1)  # dot product
        mask_floor = dot_floor < 0
        new_0 = floor_verts[mask_floor, 1, :].clone()
        new_1 = floor_verts[mask_floor, 0, :].clone()
        floor_verts[mask_floor, 0, :], floor_verts[mask_floor, 1, :] = new_0, new_1

        ceiling_vectors = ceiling_verts - quad_centers[:, None, :]
        cross_ceiling = torch.cross(ceiling_vectors[:, 0, :], 
                                    ceiling_vectors[:, 1, :], dim=1)  # cross product
        dot_ceiling = torch.sum(cross_ceiling * view_dirs, dim=1)  # dot product
        mask_ceiling = dot_ceiling > 0
        new_0 = ceiling_verts[mask_ceiling, 1, :].clone()
        new_1 = ceiling_verts[mask_ceiling, 0, :].clone()
        ceiling_verts[mask_ceiling, 0, :], ceiling_verts[mask_ceiling, 1, :] = new_0, new_1

        input_dict['layout_verts'] = torch.cat((floor_verts, ceiling_verts), dim=1)
        return input_dict


@TRANSFORMS.register_module()
class LayoutToBBoxes(BaseTransform):
    """Convert layout quads to 3D bounding boxes.
    """

    def transform(self, input_dict):
        layout_verts = input_dict['layout_verts']
        boxes = []
        for quad in layout_verts:
            quad = np.array(quad)
            center, size, angle = self.quad_to_bbox(quad, layout_bbox_size=0.1)
            box = torch.cat((torch.from_numpy(center),
                            torch.from_numpy(size), torch.from_numpy(angle)))
            boxes.append(box)
        n_boxes = torch.stack(boxes) if len(boxes) > 0 else torch.tensor([])
        bboxes = DepthInstance3DBoxes(
            n_boxes, with_yaw=True, box_dim=7, origin=(0.5, 0.5, 0.5))
        bboxes_np = bboxes.corners.numpy()
        input_dict['gt_bboxes_3d'] = bboxes
        input_dict['gt_labels_3d'] = np.zeros(bboxes_np.shape[0]).astype('int')
        if 'eval_ann_info' in input_dict.keys():
            input_dict['eval_ann_info']['gt_bboxes_3d'] = bboxes
            input_dict['eval_ann_info']['gt_labels_3d'] = np.zeros(
                bboxes_np.shape[0]).astype('int')
        return input_dict

    def quad_to_bbox(self, verts, layout_bbox_size=0.1):
        """Convert a layout quad to a 3D bounding box.

        Args:
            verts (np.ndarray): Quad vertices of shape (4, 3).
            layout_bbox_size (float): Bbox size along the smalestl axis, default 0.1.

        Returns:
            tuple with center, size and heading angle of resulting bbox.
        """
        verts = np.array(verts)  # shape (4,3)
        centroid = np.mean(verts, axis=0)
        pts_centered = verts - centroid
        u, s, vh = np.linalg.svd(pts_centered)
        normal_vector = vh[-1, :]

        if normal_vector[2] < 0:
            normal_vector = -normal_vector

        normal_vector /= np.linalg.norm(normal_vector)

        normal_vector[2] = 0
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        quad_center = np.mean(verts, axis=0)
        sorted_verts_inds = np.argsort(verts[..., 2])
        sorted_verts = verts[sorted_verts_inds]
        cos_theta = normal_vector[1]
        heading_angle = np.arccos(cos_theta)
        cos_theta1 = normal_vector[0]
        if cos_theta1 > 0:
            heading_angle = np.pi*2 - heading_angle
        rot_mat = self.get_rotation_around_z_axis(-heading_angle)
        quad_center = np.mean(sorted_verts, axis=0)
        shifted_verts = sorted_verts - quad_center[None, ...]
        rotated_verts = shifted_verts @ rot_mat.T + quad_center[None, ...]
        normal_vector = rot_mat @ normal_vector
        x_max = np.max(rotated_verts[:, 1], axis=0)
        x_min = np.min(rotated_verts[:, 1], axis=0)
        size_y = x_max - x_min
        if size_y < layout_bbox_size:
            vec1 = (rotated_verts[0] + rotated_verts[1]) / 2
            vec2 = (rotated_verts[2] + rotated_verts[3]) / 2
            v = vec2 - vec1
            L = np.sqrt(v[0] ** 2 + v[1] ** 2)
            if L > 1e-5:
                v_proj = v / L
                v_proj[2] = 0
                sign_bottom_verts = np.sign(np.dot(v_proj, normal_vector))
            else:
                sign_bottom_verts = 1
            scale = 0.5 * (layout_bbox_size - size_y) / normal_vector[1]
            normal_vector *= scale
            rotated_verts[:2] -= sign_bottom_verts * normal_vector
            rotated_verts[2:] += sign_bottom_verts * normal_vector
        xyz_max = np.max(rotated_verts, axis=0)
        xyz_min = np.min(rotated_verts, axis=0)
        center = quad_center
        size = xyz_max - xyz_min
        return center, size, np.array([heading_angle])

    def get_rotation_around_z_axis(self, angle):
        """Compute rotation matrix for rotation around Z-axis.

        Args:
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        return np.array([[np.cos(angle), np.sin(-angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])
