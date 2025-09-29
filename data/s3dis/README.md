# 🏢 Preparing S3DIS Data for TUN3D Model

## 1. Ground-Truth Point Clouds

For ground-truth point clouds, we follow the procedure from [UniDet3D](https://github.com/filaPro/unidet3d/tree/master/data/s3dis).

### Step 1. Get Preprocessed S3DIS Detection Data

* Follow the [official procedure](https://github.com/open-mmlab/mmdetection3d/tree/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/data/s3dis),
  **or** simply download the preprocessed version of S3DIS from [this link](https://huggingface.co/datasets/maksimko123/UniDet3D/blob/main/s3dis.tar.gz).
* Unzip it in the **current directory**.

### Step 2. Add Layout Information to Annotations

Run the following commands to create `.pkl` annotations with layout data:

```bash
python add_layout_to_s3dis.py --input s3dis_infos_Area --output s3dis_layout_infos_Area --split train
python add_layout_to_s3dis.py --input s3dis_infos_Area --output s3dis_layout_infos_Area --split val
```

> ⚠️ You should pass the **common prefix** as input/output path,
> e.g. `s3dis_infos_Area` (not `s3dis_infos_Area_1.pkl`).

### Alternative: Download Preprocessed PKL

* You can also **directly download preprocessed `.pkl` files** from [here](https://huggingface.co/datasets/maksimko123/TUN3D-pkl/tree/main/pkl_s3dis_gt).

---

## 2. Posed and Unposed Point Clouds

To generate posed and unposed point clouds, you first need to download preprocessed 2D images.

### Step 1. Get Preprocessed S3DIS Posed Images

* Simply download the preprocessed version of S3DIS from [this link](https://huggingface.co/datasets/bulatko/tun3d_s3dis_posed_images/tree/main).
* Unzip it in the **current directory**.

### Step 2. Reconstruction

* Navigate to the `../reconstruction` folder.
* Follow the instructions provided there,
  **or** simply download the preprocessed version ([posed](https://huggingface.co/datasets/bulatko/tun3d_s3dis_detection/blob/main/s3dis_points_posed.tar.gz), [unposed](https://huggingface.co/datasets/bulatko/tun3d_s3dis_detection/blob/main/s3dis_points_unposed.tar.gz)) and pkl files ([posed](https://huggingface.co/datasets/maksimko123/TUN3D-pkl/tree/main/pkl_s3dis_posed), [unposed](https://huggingface.co/datasets/maksimko123/TUN3D-pkl/tree/main/pkl_s3dis_unposed)).

⚠️ Be careful: The pkl files for ground-truth point clouds are incompatible with those from posed/unposed images, due to missing source images for some scenes.

The directory structure after pre-processing should be as below

```
s3dis
├── meta_data/
├── indoor3d_util.py
├── collect_indoor3d_data.py
├── README.md
├── Stanford3dDataset_v1.2_Aligned_Version/
├── s3dis_data/
├── posed_images/
│   ├── Area_[AREA_ID]_[ROOM_ID]/
│   │   ├── camera_[CAMERA_ID]_[ROOM_ID]_frame_[FRAME_ID]_domain_depth.png
│   │   ├── camera_[CAMERA_ID]_[ROOM_ID]_frame_[FRAME_ID]_domain_pose.txt
│   │   ├── camera_[CAMERA_ID]_[ROOM_ID]_frame_[FRAME_ID]_domain_rgb.jpg
│   │   ├── transform.txt
│   │   └── pcd.txt
├── points/
│   ├── xxxxx.bin
├── instance_mask/
│   ├── xxxxx.bin
├── semantic_mask/
│   ├── xxxxx.bin
├── seg_info/
│   ├── Area_1_label_weight.npy
│   ├── Area_1_resampled_scene_idxs.npy
│   ├── Area_2_label_weight.npy
│   ├── Area_2_resampled_scene_idxs.npy
│   ├── Area_3_label_weight.npy
│   ├── Area_3_resampled_scene_idxs.npy
│   ├── Area_4_label_weight.npy
│   ├── Area_4_resampled_scene_idxs.npy
│   ├── Area_5_label_weight.npy
│   ├── Area_5_resampled_scene_idxs.npy
│   ├── Area_6_label_weight.npy
│   ├── Area_6_resampled_scene_idxs.npy
├── s3dis_infos_Area_1.pkl
├── s3dis_infos_Area_2.pkl
├── s3dis_infos_Area_3.pkl
├── s3dis_infos_Area_4.pkl
├── s3dis_infos_Area_5.pkl
└── s3dis_infos_Area_6.pkl
├── s3dis_layout_infos_Area_1.pkl
├── s3dis_layout_infos_Area_2.pkl
├── s3dis_layout_infos_Area_3.pkl
├── s3dis_layout_infos_Area_4.pkl
├── s3dis_layout_infos_Area_5.pkl
└── s3dis_layout_infos_Area_6.pkl
├── s3dis_layout_infos_Area_1_posed.pkl
├── s3dis_layout_infos_Area_2_posed.pkl
├── s3dis_layout_infos_Area_3_posed.pkl
├── s3dis_layout_infos_Area_4_posed.pkl
├── s3dis_layout_infos_Area_5_posed.pkl
└── s3dis_layout_infos_Area_6_posed.pkl
├── s3dis_layout_infos_Area_1_unposed.pkl
├── s3dis_layout_infos_Area_2_unposed.pkl
├── s3dis_layout_infos_Area_3_unposed.pkl
├── s3dis_layout_infos_Area_4_unposed.pkl
├── s3dis_layout_infos_Area_5_unposed.pkl
└── s3dis_layout_infos_Area_6_unposed.pkl
```