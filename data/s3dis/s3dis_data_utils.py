from os import path as osp
import numpy as np

class_names = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter', 'unlabeled']

WALL_LABEL = class_names.index('wall')
FLOOR_LABEL = class_names.index('floor')
CEILING_LABEL = class_names.index('ceiling')


def load_scan(pcd_path, dtype):
    pcd_data = np.fromfile(pcd_path, dtype)
    return pcd_data


def get_quad_from_pc(pc_coords, z_min, z_max):
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

    for i in range(len(quad)):
        z = quad[i][2]
        z = min(z, z_max)
        z = max(z, z_min)
        quad[i][2] = z

    return sorted(quad, key=lambda vert: vert[-1])


def get_quads_from_seg(scan_name, data_dir, semantic_mask_dir, instance_mask_dir):
    points = load_scan(osp.join(data_dir, scan_name + '.bin'),
                       dtype=np.float32).reshape(-1, 6)
    semantic_mask = load_scan(osp.join(
        semantic_mask_dir, scan_name + '.bin'), dtype=np.int32).reshape(-1, 2)[:, 0]
    instance_mask = load_scan(osp.join(
        instance_mask_dir, scan_name + '.bin'), dtype=np.int32).reshape(-1, 2)[:, 0]

    vertical_quads = []
    horizontal_quads = []
    floor_z_coord = np.quantile(points[..., 2], q=0.005)
    ceiling_z_coord = np.quantile(points[..., 2], q=0.995)
    for label in [WALL_LABEL, FLOOR_LABEL, CEILING_LABEL]:
        # obtain instances of specific class
        class_instances = np.unique(instance_mask[semantic_mask == label])
        for instance_label in class_instances:
            curr_pc = points[instance_mask == instance_label]
            quad_verts = get_quad_from_pc(
                curr_pc[:, :3], floor_z_coord, ceiling_z_coord)
            if label == WALL_LABEL:
                vertical_quads.append(quad_verts)
            else:
                horizontal_quads.append(quad_verts)

    return vertical_quads, horizontal_quads, len(vertical_quads) + len(horizontal_quads)
