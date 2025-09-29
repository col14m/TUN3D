import pickle
import os
import argparse
import numpy as np
from scannet_data_utils import *

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(FILE_DIR), 'data/scannet')
LAYOUT_ANNOTATION_DIR = os.path.join(BASE_DIR, 'scannet_planes/')
DATA_DIR = os.path.join(BASE_DIR, 'points')

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input file name")
parser.add_argument("--output", type=str, required=True,
                    help="Output file name")
parser.add_argument("--split", type=str, required=True,
                    help="ScanNet split name")
args = parser.parse_args()

if __name__ == "__main__":
    scan_names_layout = list(set([os.path.basename(x)[0:12]
                                  for x in os.listdir(LAYOUT_ANNOTATION_DIR) if x.startswith('scene')]))

    split = args.split
    approx_non_quad_walls = True if split == 'train' else False
    with open(args.input, 'rb') as f:
        scene_info = pickle.load(f)

    new_data = []
    N = len(scene_info['data_list'])
    print(N)
    for j in range(N):
        curr_info = scene_info['data_list'][j]
        scene_name = curr_info['lidar_points']['lidar_path'].split('.')[0]
        if scene_name in scan_names_layout:
            vert_quads, horizontal_quads, overall_n_quads = get_quads(
                scene_name, approx_non_quads=approx_non_quad_walls)
            scene_info['data_list'][j]['layout_verts'] = vert_quads
            scene_info['data_list'][j]['horizontal_quads'] = horizontal_quads
            scene_info['data_list'][j]['n_quads_eval'] = overall_n_quads
        else:
            scene_info['data_list'][j]['layout_verts'] = []
            scene_info['data_list'][j]['horizontal_quads'] = []
            scene_info['data_list'][j]['n_quads_eval'] = 0

        new_data.append(scene_info['data_list'][j])

    scene_info['data_list'] = new_data
    with open(args.output, 'wb') as f:
        pickle.dump(scene_info, f)
