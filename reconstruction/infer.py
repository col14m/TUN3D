import torch

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.infer_utils import (
    process_image_directory_scannet,
    process_image_directory_s3dis,
)
from pathlib import Path
import argparse

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def infer_scannet(parser):
    parser.set_defaults(output_path="res/scannet")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if args.use_poses:
        output_path = output_path / "posed"
    else:
        output_path = output_path / "unposed"

    weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # dust3r will write the 3D model inside tmpdirname
    with open(f"scannet_splits/scannet_{args.split}.txt", "r") as f:
        scenes = f.read().splitlines()
    scenes = scenes[args.job_index : args.job_upper : args.job_num]
    for scene in scenes:
        scene_path = Path(args.scenes_path) / scene
        out_path = output_path / scene
        out_path.mkdir(parents=True, exist_ok=True)
        process_image_directory_scannet(
            scene_path,
            out_path,
            model,
            args.device,
            silent=args.silent,
            use_poses=args.use_poses,
            force=args.force,
            num_images=args.num_images,
        )


def infer_s3dis(parser):

    parser.set_defaults(output_path="res/s3dis")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if args.use_poses:
        output_path = output_path / "posed"
    else:
        output_path = output_path / "unposed"

    weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    scenes = sorted([s.name for s in Path(args.scenes_path).glob("*")])

    scenes = scenes[args.job_index : args.job_upper : args.job_num]
    for scene in scenes:
        scene_path = Path(args.scenes_path) / scene
        out_path = output_path / Path(scene).stem
        out_path.mkdir(parents=True, exist_ok=True)
        process_image_directory_s3dis(
            scene_path,
            out_path,
            model,
            args.device,
            silent=args.silent,
            use_poses=args.use_poses,
            force=args.force,
            num_images=args.num_images,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scannet", "s3dis"],
        help="Dataset to process: 'scannet' or 's3dis'.",
    )
    parser.add_argument(
        "-sp",
        "--scenes_path",
        type=str,
        required=True,
        help="Path to the root directory containing scene images/poses.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--tmp_dir", type=str, default=None, help="value for tempfile.tempdir"
    )
    parser.add_argument(
        "--silent", action="store_true", default=False, help="silence logs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        help="name of the model weights",
    )
    parser.add_argument(
        "-i",
        "--job_index",
        type=int,
        default=0,
        help="Zero-based index of this job in a multi-job split.",
    )
    parser.add_argument(
        "-n",
        "--job_num",
        type=int,
        default=1,
        help="Total number of parallel jobs; used as stride when slicing scenes.",
    )
    parser.add_argument(
        "-N",
        "--num_images",
        type=int,
        default=45,
        help="Maximum number of images to sample per scene.",
    )
    parser.add_argument(
        "-u",
        "--job_upper",
        type=int,
        default=int(1e10),
        help="Exclusive upper bound when slicing the scene list (for debugging/batching).",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    parser.add_argument(
        "-p",
        "--posed",
        action="store_true",
        help="Use camera poses if available (produces 'posed' outputs).",
        dest="use_poses",
    )
    parser.add_argument(
        "-s",
        "--scannet_split",
        type=str,
        default="all",
        dest="split",
        help="ScanNet split name to use; reads scannet_splits/scannet_<split>.txt.",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="res", help="Output directory."
    )

    args = parser.parse_args()

    if args.dataset == "scannet":
        infer_scannet(parser)
    elif args.dataset == "s3dis":
        infer_s3dis(parser)
