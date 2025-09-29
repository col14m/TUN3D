import argparse
from pathlib import Path
import trimesh
import numpy as np
from tqdm.contrib.concurrent import thread_map, process_map


def save_bin(path, vertices, colors):
    rgb = np.clip(colors, 0, 255)[:, :3]

    # if rgb [0, 1] then change to [0, 255]
    if rgb.max() <= 1.0:
        rgb = rgb * 255

    points = np.concatenate([vertices, rgb], axis=1).astype(np.float32)
    points.tofile(path)


def process_scene_scannet(data):
    rec_path, output_dir = data
    scene_id = rec_path.stem
    point_cloud = trimesh.load(rec_path)

    xyz = np.array(point_cloud.vertices)
    rgb = np.array(point_cloud.colors)

    output_path = output_dir / f"{scene_id}.bin"
    save_bin(output_path, xyz, rgb)

    return output_path


def process_scene_s3dis(data):
    rec_path, output_dir, scenes_path = data
    scene_id = rec_path.stem
    point_cloud = trimesh.load(rec_path)

    xyz = np.array(point_cloud.vertices)
    rgb = np.array(point_cloud.colors)

    output_path = output_dir / f"{scene_id}.bin"

    transform_path = scenes_path / f"{scene_id}" / f"transform.txt"

    transform = np.loadtxt(transform_path)
    transformed_xyz = np.hstack([xyz, np.ones((xyz.shape[0], 1))]) @ transform.T
    transformed_xyz = transformed_xyz[:, :3]

    save_bin(output_path, transformed_xyz, rgb)

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["s3dis", "scannet"]
    )
    parser.add_argument("-sp", "--scenes_path", type=str, default=None)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-n", "--num_workers", type=int, default=16)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob("*.ply"))
    if args.dataset == "scannet":
        data = [*zip(input_files, [output_dir] * len(input_files))]
        process_map(process_scene_scannet, data, max_workers=args.num_workers)
    elif args.dataset == "s3dis":
        if args.scenes_path is None:
            raise ValueError("--scenes_path argument is required for s3dis")
        scenes_path = Path(args.scenes_path)
        data = [
            *zip(
                input_files,
                [output_dir] * len(input_files),
                [scenes_path] * len(input_files),
            )
        ]
        process_map(process_scene_s3dis, data, max_workers=args.num_workers)


if __name__ == "__main__":
    main()
