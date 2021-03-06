_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'pretrained/text0.7178_audio0.6702_video0.7139_image0.7039.pth'

train_total_iters = 10000

optimizer = dict(
    _delete_=True,
    type='Adam',
    amsgrad=True,
    lr=0.01,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                     'text_branch': dict(lr_mult=0.001, decay_mult=1.0),
                     'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                     'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                     'fusion': dict(weight_decay_mult=1.0)})
)

model = dict(
    mode=3,
    modal_dropout_p=dict(text=0.3, video=0.3, image=0.3, audio=0.3),
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
    dict(type='PhotoMetricDistortion',
         brightness_delta=16,
         contrast_range=(0.75, 1.25),
         saturation_range=(0.75, 1.25),
         hue_delta=9),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Tokenize', vocab_root='dataset/vocab_small.txt',
         max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='FrameRandomErase',
         key_fields=['video'],
         aug_num_frame=9,
         aug_max_len=3,
         aug_num_block=3,
         aug_max_size=30),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'gt_labels'])
]

data = dict(
    workers_per_gpu=8,
    samples_per_gpu=16,
    train=dict(pipeline=train_pipeline))
