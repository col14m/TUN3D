import pickle
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm


def read_pkl(pkl_path):
    if not pkl_path.exists():
        raise FileNotFoundError(f"File {pkl_path} not found.")
    with open(pkl_path, "rb") as f:
        scene = pickle.load(f)
    return scene


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--points_path",
        type=str,
        required=True,
        help="Path to the directory where .bin files are stored.",
    )
    parser.add_argument(
        "--pkl_path", type=str, required=True, help="Path to the .pkl file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output .pkl file."
    )
    args = parser.parse_args()

    bins_path = Path(args.points_path)
    pkl_path = Path(args.pkl_path)
    bins = bins_path.glob("*.bin")
    bins = [bin.name for bin in bins]

    scene = read_pkl(pkl_path)
    data_list = scene["data_list"]
    new_data_list = []
    for data in tqdm(data_list, desc="Processing data"):
        scene_id = data["lidar_points"]["lidar_path"]
        if scene_id not in bins:
            print("-" * 10 + f"Scene {scene_id} not found in {bins_path}, skipping...")
        else:
            print(f"Scene {scene_id} found in {bins_path}, adding to new data list...")
            new_data_list.append(data)

        scene["data_list"] = new_data_list
        with open(args.output_path, "wb") as f:
            pickle.dump(scene, f)
