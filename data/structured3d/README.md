# 🏠 Preparing Structured3D Data for TUN3D Model

To prepare the **Structured3D** dataset for use with the **TUN3D model**, follow the steps below or simply download the preprocessed version from [here](https://huggingface.co/datasets/drozdgk/structured3d/tree/main).

## 1. Download Preprocessed Dataset

* Obtain the preprocessed version of **Structured3D** provided by the [SpatialLM authors](https://huggingface.co/datasets/ysmao/structured3d-spatiallm).
* Place the downloaded content into the **current directory**.
* Extract the point clouds along with layout annotations according to the [instructions](https://huggingface.co/datasets/ysmao/structured3d-spatiallm#data-extraction) included with the dataset.

After this step, the `/layout` and `/pcd` directories (with layout annotations in `.txt` format and point clouds in `.ply` format) should be present in the current folder.

## 2. Generate Annotation Files

Run the following script to create `.pkl` annotation files:

```bash
python parse_layout_annotations.py .
```

The directory structure after pre-processing should be as below
```
structured3d
├── layout
│   ├── scene_xxxx.txt
├── pcd
│   ├── scene_xxxx.txt
├── points
│   ├── scene_xxxx.bin
├── parse_layout_annotations.py
├── structured3d_infos_train.pkl
├── structured3d_infos_val.pkl
```

