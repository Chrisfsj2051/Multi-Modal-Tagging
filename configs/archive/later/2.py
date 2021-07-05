checkpoint_config = dict(interval=1000)
evaluation = dict(interval=1000)
seed = 1
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = []
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.01, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8000, 9000])
runner = dict(type='IterBasedRunner', max_iters=10000)
modal_used = ['image', 'video', 'text', 'audio']
model = dict(
    type='SingleBranchModel',
    pretrained='torchvision://resnet50',
    key='image',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    head=dict(
        type='SingleSEHead',
        in_dim=2048,
        gating_reduction=8,
        out_dim=1024,
        dropout_p=0.8,
        cls_head_config=dict(
            type='ClsHead',
            in_dim=1024,
            out_dim=82,
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8)),
        norm_cfg=dict(type='SyncBN')))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='CutOut',
        n_holes=3,
        cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.07, 0.07)]),
    dict(
        type='CutOut',
        n_holes=1,
        cutout_ratio=[(0.2, 0.2), (0.15, 0.15), (0.13, 0.13)]),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Shear',
            'prob': 0.5,
            'level': 1
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 2
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 3
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 4
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 5
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 6
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 7
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 8
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 9
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 10
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 1
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 2
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 3
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 4
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 5
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 6
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 7
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 8
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 9
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 10
        }]]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['image', 'meta_info', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['image', 'meta_info'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='CutOut',
                n_holes=3,
                cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.07, 0.07)]),
            dict(
                type='CutOut',
                n_holes=1,
                cutout_ratio=[(0.2, 0.2), (0.15, 0.15), (0.13, 0.13)]),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 1
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 2
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 3
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 4
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 5
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 6
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 7
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 8
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 9
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 10
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 1
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 2
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 3
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 4
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 5
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 6
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 7
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 8
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 9
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 10
                }]]),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='BertTokenize',
                bert_path='pretrained/bert',
                max_length=256),
            dict(
                type='Pad',
                video_pad_size=(300, 1024),
                audio_pad_size=(300, 128)),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['image', 'meta_info', 'gt_labels'])
        ]),
    val=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='BertTokenize',
                bert_path='pretrained/bert',
                max_length=256),
            dict(
                type='Pad',
                video_pad_size=(300, 1024),
                audio_pad_size=(300, 128)),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['image', 'meta_info'])
        ]),
    test=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/test_2nd.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='BertTokenize',
                bert_path='pretrained/bert',
                max_length=256),
            dict(
                type='Pad',
                video_pad_size=(300, 1024),
                audio_pad_size=(300, 128)),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['image', 'meta_info'])
        ]))
norm_cfg = dict(type='SyncBN')
work_dir = './work_dirs/_image_id38_2gpu'
gpu_ids = range(0, 2)
