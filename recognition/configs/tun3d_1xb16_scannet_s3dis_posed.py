_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['tun3d'])

classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                   'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                   'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                   'garbagebin']
classes_s3dis = ['table', 'chair', 'sofa', 'bookcase', 'board']

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
        datasets=['scannet', 's3dis'],
        datasets_classes=[classes_scannet, classes_s3dis],
        datasets_weights=[1.0, 1.0],
        angles=[False, False],
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
            [1.0, 0.25],
            [1.0, 0.2]
        ],  # first -- detection, second -- layout
        label2level=[
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 0]
        ]),
    train_cfg=dict(),
    test_cfg=dict(
        enable_double_layout_nms=False,
        low_sp_thr=0.18,
        up_sp_thr=0.81,
        nms_pre=1000,
        iou_thr=0.5,
        score_thr=0.01,
        nms_pre_layout=70,
        score_thr_layout=0.5,
        nms_radius=0.75))

classes_s3dis = ['table', 'chair', 'sofa', 'bookcase', 'board']
dataset_type = 'ScanNetDataset'
data_root = '../data/scannet'

metainfo = dict(classes=classes_scannet)

backend_args = None

dataset_type_s3dis = 'S3DISDataset'
data_root_s3dis = '../data/s3dis'
metainfo_s3dis = dict(classes=classes_s3dis)
train_area = [1, 2, 3, 4, 6]
test_area = 5

# Training pipeline
train_pipeline_scannet = [
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
        backend_args=backend_args),
    dict(type='AddLayoutLabels'),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='TUN3DPointSample', num_points=0.33),
    dict(
        type='RandomFlip3DLayout',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTransLayout',
        rot_range=[-0.02, 0.02],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='LayoutOrientation'),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

train_pipeline_s3dis = [
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
        backend_args=backend_args),
    dict(type='AddLayoutLabels'),
    dict(type='PointSample', num_points=100000),
    dict(
        type='RandomFlip3DLayout',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTransLayout',
        rot_range=[0, 0],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='LayoutOrientation'),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline_scannet = [
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
        with_bbox_3d=False,
        with_label_3d=False,
        backend_args=backend_args),
    dict(type='AddLayoutLabels'),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor', color_mean=None)]),
    dict(type='LayoutOrientation'),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

test_pipeline_s3dis = [
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
        with_bbox_3d=False,
        with_label_3d=False,
        backend_args=backend_args),
    dict(type='AddLayoutLabels'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor', color_mean=None)]),
    dict(type='LayoutOrientation'),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

scannet_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(pts='points_posed'),
    ann_file='scannet_layout_infos_train.pkl',
    pipeline=train_pipeline_scannet,
    filter_empty_gt=False,
    metainfo=metainfo,
    box_type_3d='Depth',
    backend_args=backend_args
)

s3dis_train_dataset = dict(
    type='ConcatDataset',
    datasets=[dict(
        type=dataset_type_s3dis,
        data_root=data_root_s3dis,
        data_prefix=dict(pts='points_posed'),
        ann_file=f's3dis_layout_infos_Area_{i}_posed.pkl',
        pipeline=train_pipeline_s3dis,
        filter_empty_gt=False,
        metainfo=metainfo_s3dis,
        box_type_3d='Depth',
        backend_args=backend_args) for i in train_area]
)

scannet_val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(pts='points_posed'),
    ann_file='scannet_layout_infos_val.pkl',
    pipeline=test_pipeline_scannet,
    metainfo=metainfo,
    test_mode=True,
    box_type_3d='Depth',
    backend_args=backend_args
)

s3dis_val_dataset = dict(
    type=dataset_type_s3dis,
    data_root=data_root_s3dis,
    data_prefix=dict(pts='points_posed'),
    pipeline=test_pipeline_s3dis,
    ann_file=f's3dis_layout_infos_Area_{test_area}_posed.pkl',
    metainfo=metainfo_s3dis,
    test_mode=True,
    box_type_3d='Depth',
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type='ConcatDataset_',
            datasets=[scannet_train_dataset, s3dis_train_dataset]
        )))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset_',
        datasets=[scannet_val_dataset, s3dis_val_dataset]
    ))

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(type='IndoorLayoutMetric_',
                     datasets=['scannet', 's3dis'],
                     datasets_classes=[classes_scannet, classes_s3dis],
                     dist_thr=[0.4])

test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

EPOCHS = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCHS, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
