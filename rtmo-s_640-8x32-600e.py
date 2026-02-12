_base_ = ['mmpose::_base_/default_runtime.py']
# 参考https://github.com/open-mmlab/mmpose/issues/2920
# 数据集类型及路径
dataset_type = 'CocoDataset'
data_mode = 'topdown'

data_root = r'D:\\毕业论文\\正面图片1\\split_dataset6.11/'
# data_root = r'/home/featurize/work/split_dataset6.11/'
work_dir = './work_dirs/rtmo-s_640-8x32-600e'
data_mode = 'bottomup'

# data_root = '/home/featurize/work/split_dataset6.11/Train/'
# val_root = '/home/featurize/work/split_dataset6.11/val/'
# annotation_root = '/home/featurize/work/split_dataset6.11/split_train.json'
# annotation_root_val = '/home/featurize/work/split_dataset6.11/split_val.json'

# 竞技反曲弓关键点检测数据集-元数据
dataset_info = {
    'dataset_name':'recurve_bow_425',
    'classes':'bow',
    'paper_info':{
        'author':'chq',
        'title':'Bow Keypoints Detection',
        'container':'OpenMMLab',
        'year':'2025'
    },
    'keypoint_info':{
        0:{'name':'up','id':0,'color':[255,0,0],'type': '','swap': ''},# swap指其在水平翻转（horizontal flip）数据增强时与哪个关键点交换位置
        1:{'name':'down','id':1,'color':[0,255,0],'type': '','swap': ''},
        2:{'name':'fe','id': 2,'color':[0,0,255],'type': '','swap': ''},
        3:{'name':'weight','id':3,'color':[0,0,255],'type': '','swap': ''},
        4:{'name':'bowstring','id':4,'color':[0,255,255],'type': '','swap': ''}
    },  # 这里缺少了一个闭合括号
    'skeleton_info': {                  # 骨骼连接定义
        0: {'link': ('up', 'bowstring'), 'id': 0, 'color': [100, 150, 200]},  # 'top'改为'up'
        1: {'link': ('bowstring', 'down'), 'id': 1, 'color': [200, 100, 150]},
        2: {'link': ('up', 'down'), 'id': 2, 'color': [150, 120, 100]},  # 'top'改为'up'
        3: {'link': ('bowstring', 'fe'), 'id': 3, 'color': [200, 200, 0]},
        4: {'link': ('bowstring', 'weight'), 'id': 4, 'color': [100, 200, 100]}  
    },
    'joint_weights': [
        1.0, 1.1, 1.5, 1.0, 1.0
    ],
    'sigmas': [
        0.25, 0.25, 0.25, 0.25, 0.25
    ]
}

# 获取关键点个数
NUM_KEYPOINTS = len(dataset_info['keypoint_info'])

# # 训练超参数
train_cfg = dict(max_epochs=600, val_interval=20, dynamic_intervals=[(580, 1)])

auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=40, max_keep_ckpts=3))

optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({'neck.encoder': dict(lr_mult=0.05)})),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=5,
        T_max=280,
        end=280,
        by_epoch=True,
        convert_to_iter_based=True),
    # this scheduler is used to increase the lr from 2e-4 to 5e-4
    dict(type='ConstantLR', by_epoch=True, factor=2.5, begin=280, end=281),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=281,
        T_max=300,
        end=580,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=580, end=600),
]

# data
input_size = (640, 640)
metafile = r'D:\yolo\mmpose\MMPose_Tutorials-main\2023\0523\mmpose\configs\_base_\datasets\coco-bow.py'
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type='YOLOXMixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(type='YOLOXHSVRandomAug'),
    #dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    #dict(type='RandomFlip'),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# # train datasets
# dataset_coco = dict(
#     type='CocoDataset',
#     data_root=data_root,
#     data_mode=data_mode,
#     ann_file='split_train.json',
#     ann_file=annotation_root, #  修改
#     data_prefix=dict(img='images/'),
#     pipeline=train_pipeline_stage1,
# )

train_dataloader = dict(
    batch_size=28,# 改
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,  # 使用统一的根目录
        data_mode=data_mode,
        ann_file='split_train.json',  # 相对于 data_root 的路径
        data_prefix=dict(img='train/images/'),  # 相对于 data_root 的训练图片路径
        pipeline=train_pipeline_stage1,
        metainfo=dataset_info  # 关键：添加 metainfo
    )
)

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='split_val.json',
        # ann_file=annotation_root_val,
        data_prefix=dict(img='val/images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dataset_info
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    # ann_file=f'{data_root}/split_val.json',
    ann_file=data_root + 'split_val.json',  # 使用拼接好的完整路径
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator

# hooks
custom_hooks = [
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=20,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(
        type='RTMOModeSwitchHook',
        epoch_attributes={
            280: {
                'proxy_target_cc': True,
                'loss_mle.loss_weight': 5.0,
                'loss_oks.loss_weight': 10.0
            },
        },
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

# model
widen_factor = 0.5
deepen_factor = 0.33

model = dict(
    type='BottomupPoseEstimator',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1),
        ]),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/'
            'yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_'
            '20211121_095711-4592a793.pth',
            prefix='backbone.',
        )),
    neck=dict(
        type='HybridEncoder',
        in_channels=[128, 256, 512],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[1, 2],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU'))),
        projector=dict(
            type='ChannelMapper',
            in_channels=[256, 256],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=2)),
    head=dict(
        type='RTMOHead',
        num_keypoints=5,# 关键点个数
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=256,
            cls_feat_channels=256,
            channels_per_group=36,
            pose_vec_channels=256,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish')),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='oks',
            oks_calculator=dict(type='PoseOKS', metainfo=metafile),  # 使用元信息文件路径
            use_keypoints_for_center=True),
        prior_generator=dict(
            type='MlvlPointGenerator',
            centralize_points=True,
            strides=[16, 32]),
        dcc_cfg=dict(
            in_channels=256,
            feat_channels=128,
            num_bins=(192, 256),
            spe_channels=128,
            gau_cfg=dict(
                s=128,
                expansion_factor=2,
                dropout_rate=0.0,
                drop_path=0.0,
                act_fn='SiLU',
                pos_enc='add')),
        overlaps_power=0.5,
        loss_cls=dict(
            type='VariFocalLoss',
            reduction='sum',
            use_target_weight=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo=metafile,
            loss_weight=30.0),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mle=dict(
            type='MLECCLoss',
            use_target_weight=True,
            loss_weight=1.0,
        ),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    test_cfg=dict(
        input_size=input_size,
        score_thr=0.1,
        nms_thr=0.65,
    ))
