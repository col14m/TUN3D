# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import math
import builtins
import datetime
import gradio
import torch
import numpy as np
import trimesh
import copy
from scipy.spatial.transform import Rotation
import gc
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from pathlib import Path
import matplotlib.pyplot as pl


def _convert_scene_output_to_glb(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    silent=False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            None if transparent_cams else imgs[i],
            focals[i],
            imsize=imgs[i].shape[1::-1],
            screen_width=cam_size,
        )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = outdir / "scene.glb"
    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(
    outdir,
    silent,
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(
        outdir,
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
        silent=silent,
    )


def get_reconstructed_scene(
    outdir,
    model,
    device,
    silent,
    image_size,
    filelist,
    schedule,
    niter,
    min_conf_thr,
    as_pointcloud,
    mask_sky,
    clean_depth,
    transparent_cams,
    cam_size,
    scenegraph_type,
    winsize,
    refid,
    poses=None,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    filelist = [str(file) for file in filelist]
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(
        imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True
    )
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    if poses is not None:
        scene.preset_pose(poses)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )

    outfile = get_3D_model_from_scene(
        outdir,
        silent,
        scene,
        min_conf_thr,
        as_pointcloud,
        mask_sky,
        clean_depth,
        transparent_cams,
        cam_size,
    )

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap("jet")
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs, output


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files - 1) / 2))
    if scenegraph_type == "swin":
        winsize = gradio.Slider(
            label="Scene Graph: Window Size",
            value=max_winsize,
            minimum=1,
            maximum=max_winsize,
            step=1,
            visible=True,
        )
        refid = gradio.Slider(
            label="Scene Graph: Id",
            value=0,
            minimum=0,
            maximum=num_files - 1,
            step=1,
            visible=False,
        )
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(
            label="Scene Graph: Window Size",
            value=max_winsize,
            minimum=1,
            maximum=max_winsize,
            step=1,
            visible=False,
        )
        refid = gradio.Slider(
            label="Scene Graph: Id",
            value=0,
            minimum=0,
            maximum=num_files - 1,
            step=1,
            visible=True,
        )
    else:
        winsize = gradio.Slider(
            label="Scene Graph: Window Size",
            value=max_winsize,
            minimum=1,
            maximum=max_winsize,
            step=1,
            visible=False,
        )
        refid = gradio.Slider(
            label="Scene Graph: Id",
            value=0,
            minimum=0,
            maximum=num_files - 1,
            step=1,
            visible=False,
        )
    return winsize, refid


def save_scene_params(
    output_dir,
    scene,
    image_files,
    min_conf_thr,
    as_pointcloud,
    mask_sky,
    clean_depth,
    transparent_cams,
    cam_size,
    output,
    silent,
    device,
    outfile,
):

    saved_files = {
        "scene_params": output_dir / "scene_params.pt",
        "output_params": output_dir / "output_params.pt",
    }

    scene_params = {}

    try:
        scene_params["image_files"] = image_files
        if hasattr(scene, "get_im_poses"):
            scene_params["poses"] = scene.get_im_poses().detach().cpu()
        elif hasattr(scene, "poses"):
            scene_params["poses"] = (
                scene.poses.detach().cpu()
                if torch.is_tensor(scene.poses)
                else scene.poses
            )

        if hasattr(scene, "get_intrinsics"):
            scene_params["Ks"] = scene.get_intrinsics().detach().cpu()
        elif hasattr(scene, "Ks"):
            scene_params["Ks"] = (
                scene.Ks.detach().cpu() if torch.is_tensor(scene.Ks) else scene.Ks
            )

        if hasattr(scene, "get_depthmaps"):
            depthmaps = scene.get_depthmaps()
            scene_params["depths"] = [
                d.detach().cpu() if torch.is_tensor(d) else d for d in depthmaps
            ]

        if hasattr(scene, "get_pts3d"):
            scene_params["pts3d"] = [
                p.detach().cpu() if torch.is_tensor(p) else p for p in scene.get_pts3d()
            ]

        if hasattr(scene, "im_conf"):
            scene_params["im_conf"] = [
                c.detach().cpu() if torch.is_tensor(c) else c for c in scene.im_conf
            ]

        if hasattr(scene, "imshapes"):
            scene_params["imshapes"] = scene.imshapes

        if hasattr(scene, "imgs"):
            scene_params["imgs_metadata"] = [
                {
                    "path": img.get("path", ""),
                    "idx": img.get("idx", -1),
                    "size": img.get("size", None),
                }
                for img in scene.imgs
            ]

        scene_params["min_conf_thr"] = min_conf_thr
        scene_params["as_pointcloud"] = as_pointcloud
        scene_params["mask_sky"] = mask_sky
        scene_params["clean_depth"] = clean_depth
        scene_params["transparent_cams"] = transparent_cams
        scene_params["cam_size"] = cam_size

    except Exception as e:
        if not silent:
            print(f"Warning while extracting scene parameters: {e}")

    torch.save(scene_params, saved_files["scene_params"])

    output_params = {}
    if isinstance(output, dict):
        for key in ["depthmaps", "confidence", "points"]:
            if key in output:
                try:
                    value = output[key]
                    if torch.is_tensor(value):
                        output_params[key] = value.detach().cpu()
                    else:
                        output_params[key] = value
                except Exception:
                    continue

    torch.save(output_params, saved_files["output_params"])

    if not silent:
        print(f"Results saved to:")
        for name, path in saved_files.items():
            print(f"  - {name}: {path}")
        print(f"3D model saved as: {outfile}")
    if device.startswith("cuda"):
        del scene
        del output
        del scene_params
        del output_params

        gc.collect()

        torch.cuda.empty_cache()

        if not silent:
            print("GPU memory freed")

    return {"saved_files": saved_files, "model_file": outfile}


def process_image_directory_scannet(
    input_dir,
    output_dir,
    model,
    device,
    tmpdirname=None,
    silent=True,
    image_size=512,
    schedule="linear",
    niter=300,
    min_conf_thr=3.0,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=True,
    transparent_cams=False,
    cam_size=0.05,
    scenegraph_type="complete",
    winsize=1,
    refid=0,
    use_poses=False,
    force=False,
    num_images=40,
):
    """
    Applies the grs method to a directory of images and saves the results.
    For the scene, only the parameters required for reconstruction are saved,
    using the methods of the PointCloudOptimizer class.

    Args:
        input_dir (Path): Path to the directory with input images
        output_dir (Path): Path to the directory for saving results
        model: DUSt3R model
        device (str): PyTorch device
        tmpdirname (str, optional): Temporary directory
        silent (bool, optional): Disable verbose output
        image_size (int, optional): Image size for processing
        schedule (str): Schedule for global alignment
        niter (int): Number of iterations for global alignment
        min_conf_thr (float): Minimum confidence threshold
        as_pointcloud (bool): Export as a point cloud
        mask_sky (bool): Mask the sky
        clean_depth (bool): Clean depth maps
        transparent_cams (bool): Transparent cameras
        cam_size (float): Camera size in the output point cloud
        scenegraph_type (str): Scene graph type
        winsize (int): Window size for the scene graph
        refid (int): Reference image ID
        use_poses (bool): Use poses for reconstruction
        force (bool): Force overwrite existing results
        num_images (int): Maximum number of images to process

    Returns:
        dict: Paths to the saved files and the 3D model
    """

    if (Path(output_dir) / "scene_params.pt").exists() and not force:
        print(
            f"Results already exist in {output_dir}. Use the --force flag to overwrite."
        )
        return

    if tmpdirname is None:
        tmpdirname = output_dir

    # scannet
    image_extensions = ["jpg", "jpeg"]
    image_files = []
    for ext in image_extensions:
        image_files.extend([*input_dir.glob(f"*.{ext}")])

    if not image_files:
        print(f"No images found in directory {input_dir}")
        return

    if not silent:
        print(f"Found {len(image_files)} images in {input_dir}")

    # scannet
    poses = [Path(image).parent / (Path(image).stem + ".txt") for image in image_files]
    poses = [np.loadtxt(pose) for pose in poses]

    image_files = [
        image_files[i] for i in range(len(image_files)) if np.isfinite(poses[i]).all()
    ]
    poses = [poses[i] for i in range(len(poses)) if np.isfinite(poses[i]).all()]

    max_image_num = num_images
    if len(image_files) > max_image_num:
        print(
            f"More than {max_image_num} images found in {input_dir}. Only the first {max_image_num} will be processed."
        )
        image_files = [
            image_files[i]
            for i in np.linspace(0, len(image_files) - 1, max_image_num).astype(int)
        ]
        poses = [
            poses[i] for i in np.linspace(0, len(poses) - 1, max_image_num).astype(int)
        ]
        poses = torch.from_numpy(np.array(poses))

    if not use_poses:
        poses = None

    scene, outfile, imgs, output = get_reconstructed_scene(
        output_dir,
        model,
        device,
        silent,
        image_size,
        image_files,
        schedule,
        niter,
        min_conf_thr,
        as_pointcloud,
        mask_sky,
        clean_depth,
        transparent_cams,
        cam_size,
        scenegraph_type,
        winsize,
        refid,
        poses,
    )

    save_scene_params(
        output_dir,
        scene,
        image_files,
        min_conf_thr,
        as_pointcloud,
        mask_sky,
        clean_depth,
        transparent_cams,
        cam_size,
        output,
        silent,
        device,
        outfile,
    )


def process_image_directory_s3dis(
    input_dir,
    output_dir,
    model,
    device,
    tmpdirname=None,
    silent=True,
    image_size=512,
    schedule="linear",
    niter=300,
    min_conf_thr=3.0,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=True,
    transparent_cams=False,
    cam_size=0.05,
    scenegraph_type="complete",
    winsize=1,
    refid=0,
    use_poses=False,
    force=False,
    num_images=40,
):
    """
    Applies the grs function to a directory of images and saves the results.
    For the scene, only the parameters required for reconstruction are saved,
    using the methods of the PointCloudOptimizer class.

    Args:
        input_dir (Path): Path to the directory with input images
        output_dir (Path): Path to the directory for saving results
        model: DUSt3R model
        device (str): PyTorch device
        tmpdirname (str, optional): Temporary directory
        silent (bool, optional): Disable verbose output
        image_size (int, optional): Image size for processing
        schedule (str): Schedule for global alignment
        niter (int): Number of iterations for global alignment
        min_conf_thr (float): Minimum confidence threshold
        as_pointcloud (bool): Export as a point cloud
        mask_sky (bool): Mask the sky
        clean_depth (bool): Clean depth maps
        transparent_cams (bool): Transparent cameras
        cam_size (float): Camera size in the output point cloud
        scenegraph_type (str): Scene graph type
        winsize (int): Window size for the scene graph
        refid (int): Reference image ID
        force (bool): Force overwrite existing results
        num_images (int): Maximum number of images to process
        use_poses (bool): Use poses for reconstruction

    Returns:
        dict: Paths to the saved files and the 3D model
    """
    try:

        if (Path(output_dir) / "scene_params.pt").exists() and not force:
            print(
                f"Results already exist in {output_dir}. Use the --force flag to overwrite."
            )
            return

        if tmpdirname is None:
            tmpdirname = output_dir

        image_extensions = ["jpg", "jpeg"]
        image_files = []
        for ext in image_extensions:
            image_files.extend([*input_dir.glob(f"*.{ext}")])

        if not image_files:
            print(f"No images found in directory {input_dir}")
            return

        if not silent:
            print(f"Found {len(image_files)} images in {input_dir}")

        poses = [
            Path(image).parent
            / ("_".join(Path(image).stem.split("_")[:-1] + ["pose"]) + ".txt")
            for image in image_files
        ]
        poses = [np.loadtxt(pose) for pose in poses]

        mask = [
            1 if np.isfinite(poses[i]).all() else 0 for i in range(len(image_files))
        ]

        image_files = [image_files[i] for i in range(len(image_files)) if mask[i]]
        poses = [poses[i] for i in range(len(poses)) if mask[i]]

        max_image_num = num_images
        if len(image_files) > max_image_num:
            print(
                f"More than {max_image_num} images found in {input_dir}. Only {max_image_num} will be processed."
            )
            image_files = [
                image_files[i]
                for i in np.linspace(0, len(image_files) - 1, max_image_num).astype(int)
            ]
            poses = [
                poses[i]
                for i in np.linspace(0, len(poses) - 1, max_image_num).astype(int)
            ]
            poses = torch.from_numpy(np.array(poses))

        if not use_poses:
            poses = None
        scene, outfile, imgs, output = get_reconstructed_scene(
            tmpdirname,
            model,
            device,
            silent,
            image_size,
            image_files,
            schedule,
            niter,
            min_conf_thr,
            as_pointcloud,
            mask_sky,
            clean_depth,
            transparent_cams,
            cam_size,
            scenegraph_type,
            winsize,
            refid,
            poses,
        )
        save_scene_params(
            output_dir,
            scene,
            image_files,
            min_conf_thr,
            as_pointcloud,
            mask_sky,
            clean_depth,
            transparent_cams,
            cam_size,
            output,
            silent,
            device,
            outfile,
        )
    except Exception as e:
        print(f"Error: {e}")
        return None
