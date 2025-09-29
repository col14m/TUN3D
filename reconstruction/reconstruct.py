from argparse import ArgumentParser
from pathlib import Path
import open3d as o3d
import numpy as np
import trimesh as tm
import torch
from tqdm import tqdm
import cv2


def load_dust3r_scene(scene_dir):
    output_params = torch.load(scene_dir / "output_params.pt")
    scene_params = torch.load(scene_dir / "scene_params.pt")

    # load glb
    scene_mesh = tm.load(str(scene_dir / "scene.glb"))
    return output_params, scene_params, scene_mesh


def get_pcd_min_max(path):
    try:
        data = np.loadtxt(path)
    except Exception as e:
        print("Can't load txt file, trying to read line by line")
        data = []
        with open(path, "r") as fin:
            lines = fin.readlines()
            for i, line in enumerate(lines):
                try:
                    data.append(list(map(float, line.split(" "))))
                except Exception as e:
                    print(f"Error at {i}: {line} ({e}), skipping")
        data = np.array(data)
    vertices = data[:, :3]
    return vertices.min(axis=0), vertices.max(axis=0)


def process_scene_s3dis(data):
    scene_id, recs_path, out_path, confidence_trunc = data
    if Path(out_path / f"{scene_id}.ply").exists():
        return

    output_params, scene_params, scene_mesh = load_dust3r_scene(recs_path / scene_id)

    poses = scene_params["poses"]
    pts3d = scene_params["pts3d"]
    Ks = scene_params["Ks"]
    image_names = scene_params["image_files"]
    confs = scene_params["im_conf"]

    vertices = np.empty((0, 3))
    colors = np.empty((0, 3))

    for i in range(len(image_names)):

        name = image_names[i]
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ch, cw = confs[i].shape
        image = cv2.resize(image, (cw, ch))
        if confidence_trunc > 0:
            conf = confs[i].numpy().astype(np.float32)
            conf = conf > confidence_trunc
        else:
            conf = np.ones_like(confs[i], dtype=bool)

        pts3d_i = pts3d[i][conf].numpy()
        colors_i = image[conf, :].astype(np.float32)
        vertices = np.concatenate([vertices, pts3d_i], axis=0)
        colors = np.concatenate([colors, colors_i], axis=0)

    mins, maxs = get_pcd_min_max(Path(image_names[0]).parent / "pcd.txt")
    colors = colors[np.all(vertices >= mins, axis=1) & np.all(vertices <= maxs, axis=1)]
    vertices = vertices[
        np.all(vertices >= mins, axis=1) & np.all(vertices <= maxs, axis=1)
    ]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    pcd = pcd.voxel_down_sample(voxel_size=0.025)
    o3d.io.write_point_cloud(str(out_path / f"{scene_id}.ply"), pcd)


def process_scene_align_s3dis(data):

    scene_id, recs_path, out_path, depth_trunc, confidence_trunc = data
    if Path(out_path / f"{scene_id}.ply").exists():
        return
    output_params, scene_params, scene_mesh = load_dust3r_scene(recs_path / scene_id)
    poses = scene_params["poses"]
    depths = scene_params["depths"]
    Ks = scene_params["Ks"]
    image_names = scene_params["image_files"]

    idx = 0
    while True:
        if idx >= len(image_names):
            raise ValueError(f"No depth map found for {scene_id}")
        image_file = Path(image_names[idx])
        gt_depth = image_file.parent / (
            "_".join(image_file.stem.split("_")[:-1] + ["depth"]) + ".png"
        )
        if not gt_depth.exists():
            print(f"Depth map not found for {image_file} ({gt_depth})")
            idx += 1
            continue

        gt_depth = cv2.imread(str(gt_depth), -1)
        gt_depth = np.where(gt_depth < 65535, gt_depth / 512.0, gt_depth)

        h, w = gt_depth.shape
        depth = depths[idx].numpy().astype(np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        valid_mask = (gt_depth > 1e-10) & (depth > 1e-10) & (gt_depth < 127.0)
        depth = depth[valid_mask]
        gt_depth = gt_depth[valid_mask]

        scale = np.median(gt_depth / depth)
        break
    image_file = Path(image_names[0])
    pose_path = image_file.parent / (
        "_".join(image_file.stem.split("_")[:-1] + ["pose"]) + ".txt"
    )
    gt_pose = np.loadtxt(str(pose_path))
    first_pose = np.linalg.inv(poses[0].numpy())
    tsdffusion = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.025,
        sdf_trunc=0.1,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(len(image_names)):
        name = image_names[i]
        color = o3d.io.read_image(name)

        fx, fy = Ks[i][0, 0], Ks[i][1, 1]
        cx, cy = Ks[i][0, 2], Ks[i][1, 2]
        depth = depths[i].numpy().astype(np.float32) * scale
        h, w = np.asarray(color).shape[:2]
        dh, dw = depth.shape

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        if confidence_trunc > 0:
            confidence = scene_params["im_conf"][i]
            confidence = confidence.numpy().astype(np.float32)
            confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
            depth[confidence < confidence_trunc] = 0
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_o3d,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        camera_o3d = o3d.camera.PinholeCameraIntrinsic(
            w, h, fx / dw * w, fy / dh * h, cx / dw * w, cy / dh * h
        )

        pose = poses[i].numpy()
        pose = first_pose @ pose
        pose[:3, 3] *= scale
        pose = gt_pose @ pose

        tsdffusion.integrate(
            rgbd,
            camera_o3d,
            np.linalg.inv(pose),
        )
    pc = tsdffusion.extract_point_cloud()
    pc.voxel_down_sample(voxel_size=0.025)

    scene_gt_path = image_file.parent / "pcd.txt"
    mins, maxs = get_pcd_min_max(scene_gt_path)

    new_points = np.asarray(pc.points)
    new_colors = np.asarray(pc.colors)

    new_colors = new_colors[
        np.all(new_points >= mins, axis=1) & np.all(new_points <= maxs, axis=1)
    ]
    new_points = new_points[
        np.all(new_points >= mins, axis=1) & np.all(new_points <= maxs, axis=1)
    ]

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    o3d.io.write_point_cloud(str(out_path / f"{scene_id}.ply"), new_pcd)


def process_scene_scannet(data):
    scene_id, recs_path, out_path, confidence_trunc = data
    if Path(out_path / f"{scene_id}.ply").exists():
        return

    output_params, scene_params, scene_mesh = load_dust3r_scene(recs_path / scene_id)

    poses = scene_params["poses"]
    pts3d = scene_params["pts3d"]
    Ks = scene_params["Ks"]
    image_names = scene_params["image_files"]
    confs = scene_params["im_conf"]

    vertices = np.empty((0, 3))
    colors = np.empty((0, 3))

    for i in range(len(image_names)):

        name = image_names[i]
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ch, cw = confs[i].shape
        image = cv2.resize(image, (cw, ch))
        if confidence_trunc > 0:
            conf = confs[i].numpy().astype(np.float32)
            conf = conf > confidence_trunc
        else:
            conf = np.ones_like(confs[i], dtype=bool)

        pts3d_i = pts3d[i][conf].numpy()
        colors_i = image[conf, :].astype(np.float32)
        vertices = np.concatenate([vertices, pts3d_i], axis=0)
        colors = np.concatenate([colors, colors_i], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)

    o3d.io.write_point_cloud(str(out_path / f"{scene_id}.ply"), pcd)


def process_scene_align_scannet(data):

    scene_id, recs_path, out_path, depth_trunc, confidence_trunc = data
    if Path(out_path / f"{scene_id}.ply").exists():
        return
    output_params, scene_params, scene_mesh = load_dust3r_scene(recs_path / scene_id)
    poses = scene_params["poses"]
    depths = scene_params["depths"]
    Ks = scene_params["Ks"]
    image_names = scene_params["image_files"]

    image_file = Path(image_names[0])

    gt_depth = image_file.parent / (image_file.stem + ".png")
    gt_pose = image_file.parent / (image_file.stem + ".txt")

    gt_depth = cv2.imread(str(gt_depth), -1) / 1000.0
    gt_pose = np.loadtxt(str(gt_pose))

    h, w = gt_depth.shape
    depth = depths[0].numpy().astype(np.float32)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    valid_mask = (gt_depth > 1e-10) & (depth > 1e-10)
    depth = depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    scale = np.median(gt_depth / depth)

    first_pose = np.linalg.inv(poses[0].numpy())
    tsdffusion = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.025,
        sdf_trunc=0.1,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(len(image_names)):
        name = image_names[i]
        color = o3d.io.read_image(name)

        fx, fy = Ks[i][0, 0], Ks[i][1, 1]
        cx, cy = Ks[i][0, 2], Ks[i][1, 2]
        depth = depths[i].numpy().astype(np.float32) * scale
        if confidence_trunc > 0:
            confidence = scene_params["im_conf"][i]
            confidence = confidence.numpy().astype(np.float32)
            depth[confidence < confidence_trunc] = 0

        h, w = np.asarray(color).shape[:2]
        dh, dw = depth.shape

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_o3d,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        camera_o3d = o3d.camera.PinholeCameraIntrinsic(
            w, h, fx / dw * w, fy / dh * h, cx / dw * w, cy / dh * h
        )

        pose = poses[i].numpy()
        pose = first_pose @ pose
        pose[:3, 3] *= scale
        pose = gt_pose @ pose

        tsdffusion.integrate(
            rgbd,
            camera_o3d,
            np.linalg.inv(pose),
        )
    pc = tsdffusion.extract_point_cloud()
    pc.voxel_down_sample(voxel_size=0.025)
    o3d.io.write_point_cloud(str(out_path / f"{scene_id}.ply"), pc)


def check_valid_input(path):
    if not (path / "output_params.pt").exists():
        return False
    if not (path / "scene_params.pt").exists():
        return False
    if not (path / "scene.glb").exists():
        return False
    return True


def process_scannet(parser):
    args = parser.parse_args()

    out_path = Path(args.out_path)
    if args.align:
        out_path = out_path / "aligned"
    else:
        out_path = out_path / "unaligned"
    out_path.mkdir(parents=True, exist_ok=True)
    recs_path = Path(args.recs_path)
    scene_ids = sorted([*recs_path.glob("*")])
    scene_ids = list(map(lambda x: Path(x).stem, scene_ids))

    if args.align:
        data = [
            (scene_id, recs_path, out_path, args.depth_trunc, args.confidence_trunc)
            for scene_id in scene_ids[args.job_id : args.job_id_upper : args.num_jobs]
            if check_valid_input(recs_path / scene_id)
        ]
        [*tqdm(map(process_scene_align_scannet, data), total=len(data))]
    else:
        data = [
            (scene_id, recs_path, out_path, args.confidence_trunc)
            for scene_id in scene_ids[args.job_id : args.job_id_upper : args.num_jobs]
            if check_valid_input(recs_path / scene_id)
        ]
        [*tqdm(map(process_scene_scannet, data), total=len(data))]


def process_s3dis(parser):
    args = parser.parse_args()

    out_path = Path(args.out_path)
    if args.align:
        out_path = out_path / "aligned"
    else:
        out_path = out_path / "unaligned"

    out_path.mkdir(parents=True, exist_ok=True)
    recs_path = Path(args.recs_path)
    scenes = sorted([*recs_path.glob("*")])
    scenes = list(map(lambda x: Path(x).stem, scenes))
    scenes = scenes[args.job_id : args.job_id_upper : args.num_jobs]
    if args.align:
        data = [
            (scene_id, recs_path, out_path, args.depth_trunc, args.confidence_trunc)
            for scene_id in scenes
            if check_valid_input(recs_path / scene_id)
        ]
        [*tqdm(map(process_scene_align_s3dis, data), total=len(data))]
    else:
        data = [
            (scene_id, recs_path, out_path, args.confidence_trunc)
            for scene_id in scenes
            if check_valid_input(recs_path / scene_id)
        ]
        [*tqdm(map(process_scene_s3dis, data), total=len(data))]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-rp", "--recs_path", type=str, required=True)
    parser.add_argument(
        "-o", "--out_path", type=str, default="res/scannet/reconstructions"
    )
    parser.add_argument("-a", "--align", action="store_true")
    parser.add_argument("-dt", "--depth_trunc", type=float, default=3.0)
    parser.add_argument("-ct", "--confidence_trunc", type=float, default=0.0)
    parser.add_argument("-i", "--job_id", type=int, default=0)
    parser.add_argument("-u", "--job_id_upper", type=int, default=int(1e10))
    parser.add_argument("-n", "--num_jobs", type=int, default=1)
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["s3dis", "scannet"]
    )

    args = parser.parse_args()

    if args.dataset == "s3dis":
        parser.set_defaults(out_path="res/s3dis/reconstructions")
        process_s3dis(parser)

    elif args.dataset == "scannet":
        parser.set_defaults(out_path="res/scannet/reconstructions")
        process_scannet(parser)
