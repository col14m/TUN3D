import pickle
import os
import argparse
import numpy as np
from s3dis_data_utils import *

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(FILE_DIR), 'data/s3dis')
DATA_DIR = os.path.join(BASE_DIR, 'points')
SEMANTIC_MASK_DIR = os.path.join(BASE_DIR, 'semantic_mask')
INSTANCE_MASK_DIR = os.path.join(BASE_DIR, 'instance_mask')

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input prefix")
parser.add_argument("--output", type=str, required=True, help="Output prefix")
parser.add_argument("--split", type=str, required=True,
                    help="S3DIS split name")

area = {
    "train": [1, 2, 3, 4, 6],
    "val": [5]
}

args = parser.parse_args()

if __name__ == "__main__":
    split = args.split.lower()
    for area_idx in area[split]:
        pkl_name = f'{args.input}_{area_idx}.pkl'
        with open(pkl_name, 'rb') as f:
            scene_info = pickle.load(f)

        new_data = []
        N = len(scene_info['data_list'])
        for j in range(N):
            curr_info = scene_info['data_list'][j]
            scene_name = curr_info['lidar_points']['lidar_path'].split('.')[0]
            vert_quads, horizontal_quads, overall_n_quads = get_quads_from_seg(
                scene_name, data_dir=DATA_DIR, semantic_mask_dir=SEMANTIC_MASK_DIR, instance_mask_dir=INSTANCE_MASK_DIR)
            scene_info['data_list'][j]['layout_verts'] = vert_quads
            scene_info['data_list'][j]['horizontal_quads'] = horizontal_quads
            scene_info['data_list'][j]['n_quads_eval'] = overall_n_quads
            if len(vert_quads) > 0:
                new_data.append(scene_info['data_list'][j])

        scene_info['data_list'] = new_data
        with open(f'{args.output}_{area_idx}.pkl', 'wb') as f:
            pickle.dump(scene_info, f)
