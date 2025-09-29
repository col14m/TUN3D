# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['tun3d'])

classes_stru3d = ['door', 'window']
model = dict(
    type='MinkSingleStage3DDetector',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='TUN3DMinkResNet',
        in_channels=3,
        depth=34,
        norm='batch',
        num_planes=(64, 128, 128, 128)),
    neck=dict(
        type='TUN3DNeck', in_channels=(64, 128, 128, 128), out_channels=128),
    bbox_head=dict(
        type='TUN3DHead',
        datasets=['structured3d'],
        datasets_classes=[classes_stru3d],
        angles=[False],
        datasets_weights=[1.0],
        in_channels=128,
        voxel_size=0.01,
        pts_center_threshold=6,
        pts_center_threshold_layout=6,
        layout_level=1,
        num_reg_outs=6,
        layout_head=dict(
            type='LayoutHead',
            input_dim=128,
            num_layout_reg_outs=5,
            n_spconv2d=3,
            n_q_feats=10,
            voxel_size=0.01
        ),
        loss_weights=[
            [0.75, 0.25]
        ],
        label2level=[
            [0, 0]
        ]),
    train_cfg=dict(),
    test_cfg=dict(
        enable_double_layout_nms=True,
        nms_pre=1000,
        iou_thr=0.15,
        score_thr=0.35,
        nms_pre_layout=150,
        score_thr_layout=0.5,
        nms_radius=0.1,
        iou_thr_layout=0.1))

# Dataset settings
dataset_type = 'Stru3DDataset'
data_root = '../data/structured3d'
metainfo = dict(
    classes=classes_stru3d)

backend_args = None

# Training pipeline
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=False,
        with_seg_3d=False,
        backend_args=backend_args),
    dict(type='AddLayoutLabels'),
    dict(type='TUN3DPointSample', num_points=0.33),
    dict(
        type='RandomFlip3DLayout',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTransLayout',
        rot_range=[0, 0],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor',
         color_mean=[127.5, 127.5, 127.5]),
    dict(type='LayoutOrientation'),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# Testing pipeline
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args),
    dict(type='AddLayoutLabels'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor',
                 color_mean=[127.5, 127.5, 127.5]),
        ]),
    dict(type='LayoutOrientation'),
    dict(type='Pack3DDetInputs_', keys=['points'], load_meta=True)
]

# Data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=7,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='structured3d_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            box_type_3d='Depth',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='structured3d_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth',
        backend_args=backend_args))

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(type='IndoorLayoutMetric_',
                     datasets=['structured3d'],
                     datasets_classes=[classes_stru3d],
                     floor_and_ceiling=False)
test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Training schedule
EPOCHS = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCHS, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimization
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2))


param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=EPOCHS,
    by_epoch=True,
    milestones=[8, 11],
    gamma=0.1)

# Custom hooks
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2))
