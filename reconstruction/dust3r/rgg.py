import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm


def read_bin(path):
    pcd = np.fromfile(str(path), dtype=np.float32).reshape(-1, 6)
    return pcd[:, :3], pcd[:, 3:]

def write_bin(path, xyz, rgb):
    rgb = np.clip(rgb, 0, 255)[:, :3]
    if rgb.max() <= 1:
        rgb = (rgb * 255)

    points = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    points.tofile(str(path))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True)
    parser.add_argument('-o', "--output", type=str, required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    for scene_id in tqdm([*input_path.glob("*.bin")]):
        vertices, colors = read_bin(scene_id)
        colors[:, 2] = colors[:, 1]
        write_bin(output_path / f"{scene_id.stem}.bin", vertices, colors)