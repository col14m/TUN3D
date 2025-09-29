from typing import Dict, List, Tuple
from entity import Wall, Door, Window, get_corners
import numpy as np
import open3d as o3d
import pickle
import csv
import os
from tqdm import tqdm
import argparse

LABEL_MAPPING = {
    'door': 0,
    'window': 1
}


def quad_to_aabb(verts, layout_bbox_size=0.1):
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

    rotated_verts = sorted_verts
    x_max = np.max(rotated_verts, axis=0)
    x_min = np.min(rotated_verts, axis=0)
    size_y = x_max - x_min
    min_dim = np.argmin(size_y)
    min_size = size_y[min_dim]
    if min_size < layout_bbox_size:
        sign_bottom_verts = 1
        scale = 0.5 * (layout_bbox_size - min_size) / normal_vector[min_dim]
        normal_vector *= scale
        rotated_verts[:2] -= sign_bottom_verts * normal_vector
        rotated_verts[2:] += sign_bottom_verts * normal_vector
    xyz_max = np.max(rotated_verts, axis=0)
    xyz_min = np.min(rotated_verts, axis=0)
    center = quad_center
    size = xyz_max - xyz_min
    return center, size, 0


def ply_to_bin(ply_path, bin_path):
    """
    Convert PLY file to binary format (x, y, z, r, g, b)
    """
    try:
        # Read PLY file
        pcd = o3d.io.read_point_cloud(str(ply_path))

        # Get points and colors
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        colors = (colors - colors.min()) / (colors.max() - colors.min())
        colors *= 255
        combined = np.zeros((len(points), 6), dtype=np.float32)
        combined[:, :3] = points.astype(np.float32)
        combined[:, 3:6] = colors.astype(np.float32)

        combined.tofile(bin_path)

        return True

    except Exception as e:
        print(e)
        return False


def parse_annotation_file(file_path: str) -> Tuple[Dict[int, 'Wall'], List['Door'], List['Window']]:

    walls_dict: Dict[int, Wall] = {}
    doors_list: List[Door] = []
    windows_list: List[Window] = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split at '='
        key, value = line.split('=', 1)
        entity_type = key.split('_')[0]
        entity_id = int(key.split('_')[1])

        # Extract arguments from parentheses
        args_str = value[value.find('(')+1: value.rfind(')')]
        args = [arg.strip() for arg in args_str.split(',')]

        if entity_type == 'wall':
            wall = Wall(
                id=entity_id,
                ax=float(args[0]),
                ay=float(args[1]),
                az=float(args[2]),
                bx=float(args[3]),
                by=float(args[4]),
                bz=float(args[5]),
                height=float(args[6]),
                thickness=float(args[7]) if len(args) > 7 else 0.0
            )
            walls_dict[wall.id] = wall

        elif entity_type == 'door':
            # Simplified wall_id extraction
            wall_id = int(args[0].split('_')[1])
            door = Door(
                id=entity_id,
                wall_id=wall_id,
                position_x=float(args[1]),
                position_y=float(args[2]),
                position_z=float(args[3]),
                width=float(args[4]),
                height=float(args[5])
            )
            doors_list.append(door)

        elif entity_type == 'window':
            wall_id = int(args[0].split('_')[1])
            window = Window(
                id=entity_id,
                wall_id=wall_id,
                position_x=float(args[1]),
                position_y=float(args[2]),
                position_z=float(args[3]),
                width=float(args[4]),
                height=float(args[5])
            )
            windows_list.append(window)

    return walls_dict, doors_list, windows_list


def parse_stru3d_scene(ann_path, split):
    layout_verts = []
    instances = []
    door_verts = []
    window_verts = []
    walls_dict, doors_list, windows_list = parse_annotation_file(ann_path)

    for wall_id, wall_entity in walls_dict.items():
        corners = get_corners(wall_entity)
        layout_verts.append(corners)

    for door_entity in doors_list:
        corners = get_corners(door_entity, walls_dict)
        center, size, _ = quad_to_aabb(corners)
        bbox = np.hstack([center, size]).reshape(-1)
        label = LABEL_MAPPING['door']
        instances.append({'bbox_3d': list(bbox), 'bbox_label_3d': label})
        if split != 'train':
            door_verts.append(corners.tolist())

    for window_entity in windows_list:
        corners = get_corners(window_entity, walls_dict)
        center, size, _ = quad_to_aabb(corners)
        bbox = np.hstack([center, size]).reshape(-1)
        label = LABEL_MAPPING['window']
        instances.append({'bbox_3d': list(bbox), 'bbox_label_3d': label})
        if split != 'train':
            window_verts.append(corners.tolist())

    if split == 'train':
        return layout_verts, instances
    else:
        return layout_verts, instances, door_verts, window_verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert PLY files to binary format')
    parser.add_argument('destination', help='Destination directory')
    args = parser.parse_args()
    target_folder = args.destination
    scene_info_train_new = dict()
    scene_info_val_new = dict()

    ann_dict = dict(train=scene_info_train_new, val=scene_info_val_new)
    all_scenes_csv = open('split.csv', newline='')
    csv_reader = csv.reader(all_scenes_csv)
    for row in tqdm(csv_reader, ncols=70):
        if row[0] == 'id':
            continue
        scene_name, split = row[0], row[1]
        if split == 'test':
            split = 'val'
        else:
            continue
        source_pcd_path = os.path.join('pcd', scene_name + '.ply')
        target_pcd_path = os.path.join(
            target_folder, 'points', scene_name + '.bin')
        success = ply_to_bin(source_pcd_path, target_pcd_path)
        if not success:
            continue
        txt_ann_path = os.path.join('layout', scene_name + '.txt')
        if split == 'train':
            layout_verts, instances = parse_stru3d_scene(txt_ann_path, split)
            ann_dict[split][scene_name] = {
                'layout_verts': layout_verts,
                'n_quads_eval': len(layout_verts),
                'horizontal_quads': [],
                'instances': instances
            }
        else:
            layout_verts, instances, door_verts, window_verts = parse_stru3d_scene(txt_ann_path, split)
            ann_dict[split][scene_name] = {
                'layout_verts': layout_verts,
                'door_verts': door_verts,
                'window_verts': window_verts,
                'n_quads_eval': len(layout_verts),
                'horizontal_quads': [],
                'instances': instances
            }

    all_scenes_csv.close()
    print('Parsing completed, now adding to annotations...')
    for split in ['train', 'val']:
        print(f'Adding to {split} split...')
        new_info = dict()
        new_data_list = []
        for scene_name, scene_ann in tqdm(ann_dict[split].items(), ncols=70):
            new_data_list_item = dict(
                lidar_points=dict(
                    lidar_path=scene_name + '.bin',
                    num_pts_feats=6
                ),
                pts_instance_mask_path='',
                pts_semantic_mask_path=''
            )
            new_data_list_item.update(scene_ann)
            new_data_list.append(new_data_list_item)

        new_info['data_list'] = new_data_list
        save_ann_path = os.path.join(
            target_folder, f'structured3d_infos_{split}.pkl')
        with open(save_ann_path, 'wb') as f:
            pickle.dump(new_info, f)
