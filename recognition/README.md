### Installation
We use the same [environment](https://github.com/filaPro/unidet3d/blob/master/Dockerfile) as in [Unidet3D](https://github.com/filaPro/unidet3d). For working with the Structured3D dataset, you also need the `shapely` package:
```
pip install shapely==2.1.1
```
If you are not using Docker, please follow the instructions in [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/get_started.md).

### Getting started
Please see [train_test.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/user_guides/train_test.md) for basic usage examples.

#### Training
We provide configs for joint training on ScanNet and S3DIS. To train TUN3D on point clouds from posed images, run the following command:
```
python tools/train.py configs/tun3d_1xb16_scannet_s3dis_posed.py
```
TUN3D can also be trained on [ground-truth point clouds](./configs/tun3d_1xb16_scannet_s3dis.py), [point clouds from unposed images](./configs/tun3d_1xb16_scannet_s3dis_unposed.py), or on [individual datasets](./configs/tun3d_1xb16_scannet.py).

#### Testing
To test a trained model, run the testing script with the path to the checkpoint file:
```
python tools/test.py configs/tun3d_1xb16_scannet_s3dis_posed.py \
    work_dirs/tun3d_1xb16_scannet_s3dis_posed/epoch_12.pth
```

#### Visualization
To visualize ground truth and predicted bounding boxes, run the testing script with the `--show` and `--show-dir` arguments:
```
python tools/test.py configs/tun3d_1xb16_scannet_s3dis_posed.py \
    work_dirs/tun3d_1xb16_scannet_s3dis_posed/epoch_12.pth \
    --show --show-dir work_dirs/tun3d_1xb16_scannet_s3dis_posed
```
For better visualizations, set `model.test_cfg.score_thr` to `0.3` in the config file.


### Trained Models
Metrics for all available pretrained models are given below (they might slightly deviate from the values reported in the paper due to the randomized training/testing procedure).
#### ScanNet & S3DIS
| Modality         | ScanNet Layout F1 | ScanNet AP@0.25 | ScanNet AP@0.50 | S3DIS Layout F1 | S3DIS AP@0.25 | S3DIS AP@0.50 | Download      |
|:----------------:|:----------------:|:---------------:|:---------------:|:---------------:|:--------------:|:--------------:|:-------------:|
| [GT points clouds](./configs/tun3d_1xb16_scannet_s3dis.py) | 66.6             | 72.7            | 60.2            | 52.0            | 74.4           | 58.6           | [model](https://huggingface.co/maksimko123/TUN3D/blob/main/tun3d_scannet_s3dis_gt.pth) / [log](https://huggingface.co/maksimko123/TUN3D/blob/main/tun3d_scannet_s3dis_gt.log)   |
| [Posed RGB](./configs/tun3d_1xb16_scannet_s3dis_posed.py)        | 52.2             | 58.3            | 36.7            | 34.4            | 33.8           | 15.6           | [model](https://huggingface.co/maksimko123/TUN3D/blob/main/tun3d_scannet_s3dis_posed.pth) / [log](https://huggingface.co/maksimko123/TUN3D/blob/main/tun3d_scannet_s3dis_posed.log)   |
| [Unposed RGB](./configs/tun3d_1xb16_scannet_s3dis_unposed.py)      | 47.2             | 43.0            | 20.4            | 21.5            | 11.7           | 1.4            | [model](https://huggingface.co/maksimko123/TUN3D/blob/main/tun3d_scannet_s3dis_unposed.pth) / [log](https://huggingface.co/maksimko123/TUN3D/blob/main/tun3d_scannet_s3dis_unposed.log)   |

#### Structured3D
| Layouts | F1@0.25 IoU | F1@0.50 IoU |
|:-------:|:-----------:|:-----------:|
| Wall    | 88.1       | 86.9        |
| Door    | 93.9        | 93.7        |
| Window  | 89.6        | 88.2        |
| **Overall** | **90.5** | **89.6** |

Download: [model](https://huggingface.co/maksimko123/TUN3D/blob/main/tun3d_structured3d.pth), [log](https://huggingface.co/maksimko123/TUN3D/blob/main/tun3d_structured3d.log)