- [Datasets structure](#datasets-structure)
  - [ScanNet](#datasets-scannet)
  - [S3DIS](#datasets-s3dis)
- [Install requirements](#install)
- [Full Reconstruction pipeline](#pipeline)
  - [ScanNet posed mode](#pipeline-scannet-posed)
  - [ScanNet unposed mode](#pipeline-scannet-unposed)
  - [S3DIS posed mode](#pipeline-s3dis-posed)
  - [S3DIS unposed mode](#pipeline-s3dis-unposed)

<a id="datasets-structure"></a>
# Datasets structure
After preprocessing datasets (**[`Scannet`](./../data/scannet/)**, **[`S3DIS`](./../data/s3dis/)**), the following data structure should be obtained:

## ScanNet

```
../data/scannet/posed_images/
    scene0000_00/
        0.jpg # color
        0.txt # pose
        0.png # depth
        1.jpg
        1.txt
        1.png
        ...
    scene0000_01/
        0.jpg # color
        0.txt # pose
        0.png # depth
        1.jpg
        1.txt
        1.png
        ...
    ...

```


## S3DIS

```
../data/s3dis/posed_images/
    Area_1_WC_1/
        camera_03eb3fa2e1524ee887ba22d1a4896f3c_WC_1_frame_0_domain_depth.png # depth
        camera_03eb3fa2e1524ee887ba22d1a4896f3c_WC_1_frame_0_domain_pose.txt # pose
        camera_03eb3fa2e1524ee887ba22d1a4896f3c_WC_1_frame_0_domain_rgb.jpg # color
        camera_03eb3fa2e1524ee887ba22d1a4896f3c_WC_1_frame_10_domain_depth.png
        camera_03eb3fa2e1524ee887ba22d1a4896f3c_WC_1_frame_10_domain_pose.txt
        camera_03eb3fa2e1524ee887ba22d1a4896f3c_WC_1_frame_10_domain_rgb.jpg
        ...
        camera_ce0a7ef47557461d97f72a45720d37c9_WC_1_frame_9_domain_depth.png
        camera_ce0a7ef47557461d97f72a45720d37c9_WC_1_frame_9_domain_pose.txt
        camera_ce0a7ef47557461d97f72a45720d37c9_WC_1_frame_9_domain_rgb.jpg
        transform.txt # S3DIS transform matrix (shift center to (0, 0, 0) & axis align)
        pcd.txt # GT point cloud
    Area_1_conferenceRoom_1/
        camera_[CAMERA_ID]_conferenceRoom_1_frame_[FRAME_ID]_domain_depth.png # depth
        camera_[CAMERA_ID]_conferenceRoom_1_frame_[FRAME_ID]_domain_pose.txt # pose
        camera_[CAMERA_ID]_conferenceRoom_1_frame_[FRAME_ID]_domain_rgb.jpg # color
        ...
        transform.txt # S3DIS transform matrix
        pcd.txt # GT point cloud
    ...
    Area_[AREA_ID]_[ROOM_ID]/
        camera_[CAMERA_ID]_[ROOM_ID]_frame_[FRAME_ID]_domain_depth.png # depth
        camera_[CAMERA_ID]_[ROOM_ID]_frame_[FRAME_ID]_domain_pose.txt # pose
        camera_[CAMERA_ID]_[ROOM_ID]_frame_[FRAME_ID]_domain_rgb.jpg # color
        ...
        transform.txt # S3DIS transform matrix
        pcd.txt # GT point cloud
    ...


```


<a id="install"></a>
# Install requirements
Please follow installation process from [original DUSt3R repository](https://github.com/naver/dust3r?tab=readme-ov-file#installation) (to create environment)

Also we use Open3D to apply TSDF Fusion while converting DUSt3R outputs to `.ply` files.
```
pip install open3d==0.19.0
```

<a id="pipeline"></a>
# Full Reconstruction pipeline

<a id="pipeline-scannet-posed"></a>
## ScanNet posed mode

### 1. Run DUSt3R

```
python infer.py \
--dataset scannet \
--scenes_path ../data/scannet/posed_images/ \
--posed \
--num_images 45
```
**Note** If you have got CUDA out of memory error - you can reduce number of images (`--num_images`)

### 2. Transform DUSt3R outputs to `*.ply` files

```
python reconstruct.py \
--dataset scannet \
--recs_path res/scannet/posed \
--confidence_trunc 4.0
```

### 3. Convert `.ply` files to `*.bin` files

```
python convert_to_bin.py \
--dataset scannet \
--input res/scannet/reconstruntions/unaligned \
--output ../data/scannet/points_posed
```

<a id="pipeline-scannet-unposed"></a>
## ScanNet unposed mode


```
python infer.py \
--dataset scannet \
--scenes_path ../data/scannet/posed_images/ \
--num_images 45
```
**Note** If you have got CUDA out of memory error - you can reduce number of images (`--num_images`)

### 2. Transform DUSt3R outputs to `*.ply` files

```
python reconstruct.py \
--dataset scannet \
--recs_path res/scannet/unposed \
--confidence_trunc 4.0 \
--align
```

**Note** Here we use alignment procedure to find transform and scale of reconstruction. 1 GT pose and 1 GT depth map are used.
### 3. Convert `.ply` files to `*.bin` files
```
python convert_to_bin.py \
--dataset scannet \
--input res/scannet/reconstruntions/aligned \
--output ../data/scannet/points_unposed
```
<a id="pipeline-s3dis-posed"></a>
## S3DIS posed mode

### 1. Run DUSt3R

```
python infer.py \
--dataset s3dis \
--scenes_path ../data/s3dis/posed_images/ \
--posed \
--num_images 45
```
**Note** If you have got CUDA out of memory error - you can reduce number of images (`--num_images`)

### 2. Transform DUSt3R outputs to `*.ply` files

```
python reconstruct.py \
--dataset s3dis \
--recs_path res/s3dis/posed \
--confidence_trunc 4.0
```

### 3. Convert `.ply` files to `*.bin` files

```
python convert_to_bin.py \
--dataset s3dis \
--input res/s3dis/reconstruntions/unaligned \
--output ../data/s3dis/points_posed \
--scenes_path ../data/s3dis/posed_images
```

**Note** Here `--scenes_path` is used to get precomputed transform matrices.


### 4. Build `.pkl` file to exclude scenes that haven't been reconstructed
```

for i in 1 2 3 4 5 6; do
  python build_pkl.py \
    --points_path ../data/s3dis/points_posed \
    --pkl_path ../data/s3dis/s3dis_layout_infos_Area_${i}.pkl \
    --output_path ../data/s3dis/s3dis_layout_infos_Area_${i}_posed.pkl
done
```

<a id="pipeline-s3dis-unposed"></a>
## S3DIS unposed mode


```
python infer.py \
--dataset s3dis \
--scenes_path ../data/s3dis/posed_images/ \
--num_images 45
```
**Note** If you have got CUDA out of memory error - you can reduce number of images (`--num_images`)

### 2. Transform DUSt3R outputs to `*.ply` files

```
python reconstruct.py \
--dataset s3dis \
--recs_path res/s3dis/unposed \
--confidence_trunc 4.0 \
--align
```

**Note** Here we use alignment procedure to find transform and scale of reconstruction. 1 GT pose and 1 GT depth map are used.
### 3. Convert `.ply` files to `*.bin` files
```
python convert_to_bin.py \
--dataset s3dis \
--input res/s3dis/reconstruntions/aligned \
--output ../data/s3dis/points_unposed \
--scenes_path ../data/s3dis/posed_images
```

**Note** Here `--scenes_path` is used to get precomputed transform matrices.

### 4. Build `.pkl` file to exclude scenes that haven't been reconstructed
```
for i in 1 2 3 4 5 6; do
  python build_pkl.py \
    --points_path ../data/s3dis/points_unposed \
    --pkl_path ../data/s3dis/s3dis_layout_infos_Area_${i}.pkl \
    --output_path ../data/s3dis/s3dis_layout_infos_Area_${i}_unposed.pkl
done
```
