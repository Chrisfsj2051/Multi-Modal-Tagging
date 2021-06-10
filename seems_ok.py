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
    type='Adam',
    amsgrad=True,
    lr=0.01,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            image_branch=dict(lr_mult=0.01, decay_mult=1.0),
            text_branch=dict(lr_mult=0.001, decay_mult=1.0),
            video_branch=dict(lr_mult=0.01, decay_mult=1.0),
            audio_branch=dict(lr_mult=0.01, decay_mult=1.0),
            fusion=dict(weight_decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3333, 6666])
runner = dict(type='IterBasedRunner', max_iters=10000)
modal_used = ['image', 'video', 'text', 'audio']
model = dict(
    type='MultiBranchesFusionModel',
    pretrained=dict(
        image='pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth'),
    modal_dropout_p=dict(text=0.3, video=0.3, image=0.3, audio=0.3),
    use_batch_norm=False,
    mode=3,
    modal_used=['image', 'video', 'text', 'audio'],
    branch_config=dict(
        video=dict(
            type='NeXtVLAD',
            feature_size=1024,
            max_frames=300,
            cluster_size=128),
        audio=dict(
            type='NeXtVLAD', feature_size=128, max_frames=300,
            cluster_size=64),
        image=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch'),
        text=dict(
            type='TwoStreamTextCNN',
            vocab_size=9906,
            ebd_dim=300,
            channel_in=256,
            channel_out=1024,
            filter_size=(2, 3, 4),
            dropout_p=0.0)),
    head_config=dict(
        video=dict(
            type='SEHead',
            in_dim=16384,
            gating_reduction=8,
            out_dim=1024,
            dropout_p=0.8,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss'))),
        image=dict(
            type='SEHead',
            in_dim=2048,
            gating_reduction=8,
            out_dim=1024,
            dropout_p=0.5,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss'))),
        text=dict(
            type='SEHead',
            in_dim=1024,
            gating_reduction=8,
            out_dim=1024,
            dropout_p=0.5,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss'))),
        audio=dict(
            type='SEHead',
            in_dim=1024,
            dropout_p=0.5,
            gating_reduction=8,
            out_dim=1024,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss'))),
        fusion=dict(
            type='SEHead',
            in_dim=20480,
            gating_reduction=8,
            dropout_p=0.8,
            out_dim=1024,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss')))))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(
        type='Tokenize', vocab_root='dataset/vocab_small.txt', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect', keys=['video', 'image', 'text', 'audio', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(
        type='Tokenize', vocab_root='dataset/vocab_small.txt', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='Tokenize',
                vocab_root='dataset/vocab_small.txt',
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
            dict(
                type='Collect',
                keys=['video', 'image', 'text', 'audio', 'gt_labels'])
        ]),
    val=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='Tokenize',
                vocab_root='dataset/vocab_small.txt',
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
            dict(type='Collect', keys=['video', 'image', 'text', 'audio'])
        ]),
    test=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/test.txt',
        label_id_file='dataset/tagging/label_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='Tokenize',
                vocab_root='dataset/vocab_small.txt',
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
            dict(type='Collect', keys=['video', 'image', 'text', 'audio'])
        ]))
train_total_iters = 10000
work_dir = './work_dirs/mode3_difflr'
gpu_ids = range(0, 2)
