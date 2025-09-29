import pickle
import os
from pathlib import Path
import numpy as np
import trimesh as tm
from tqdm import tqdm

p1 = Path("res_rec_posed_p1_ct3_dt3")
p2 = Path("res_rec_posed_p2_ct3_dt3")
out_path = Path("res_rec_posed_halves_ct3_dt3")


for scene in tqdm([*p1.glob("*.ply")]):
    scene_id = scene.stem
    scene_p2 = p2 / f"{scene_id}.ply"
    scene_out_path = out_path / f"{scene_id}.ply"
    scene_out_path.parent.mkdir(parents=True, exist_ok=True)
    scene_p1 = tm.load(scene)
    scene_p2 = tm.load(scene_p2)

    scene_out = tm.PointCloud(np.concatenate([
        scene_p1.vertices, scene_p2.vertices
        ], axis=0), colors=np.concatenate([
            scene_p1.colors, scene_p2.colors
            ], axis=0))
    scene_out.export(str(scene_out_path))