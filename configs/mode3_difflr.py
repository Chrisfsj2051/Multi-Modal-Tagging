_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

train_total_iters = 10000

optimizer = dict(
    _delete_=True,
    type='AdamW',
    amsgrad=True,
    lr=0.01,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'image': dict(lr_mult=0.01, decay_mult=1.0),
                     'text': dict(lr_mult=0.001, decay_mult=1.0),
                     'video': dict(lr_mult=0.01, decay_mult=1.0),
                     'audio': dict(lr_mult=0.01, decay_mult=1.0),
                     'fusion': dict(weight_decay_mult=0.0)})
)

model = dict(
    mode=3,
    modal_dropout_p=dict(text=0.3, video=0.3, image=0.3, audio=0.3),
    # attn_config=dict(
    #     in_dim=20480,
    #     input_dropout_p=0.3,
    # ),
    # head_config=dict(
    #     fusion=dict(
    #         _delete_=True,
    #         type='ClsHead',
    #         out_dim=82,
    #         in_dim=1024,
    #         loss=dict(type='MultiLabelBCEWithLogitsLoss'))
        # video=dict(loss=dict(loss_weight=0.1)),
        # image=dict(loss=dict(loss_weight=0.1)),
        # audio=dict(loss=dict(loss_weight=0.1)),
        # text=dict(loss=dict(loss_weight=0.1))
    # )
)

optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[train_total_iters // 3, 2 * train_total_iters // 3]
)

runner = dict(type='IterBasedRunner', max_iters=train_total_iters)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations'),
    # dict(type='PhotoMetricDistortion',
    #      brightness_delta=16,
    #      contrast_range=(0.75, 1.25),
    #      saturation_range=(0.75, 1.25),
    #      hue_delta=9),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Tokenize', vocab_root='dataset/vocab_small.txt',
         max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    # dict(type='FrameRandomErase',
    #      key_fields=['video'],
    #      aug_num_frame=30,
    #      aug_max_len=10,
    #      aug_num_block=10,
    #      aug_max_size=100),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'gt_labels'])
]

data = dict(
    workers_per_gpu=0,
    train=dict(pipeline=train_pipeline))
