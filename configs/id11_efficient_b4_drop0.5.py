_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/single_branch.py', '_base_/datasets/base_dataset.py'
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Tokenize', vocab_root='dataset/vocab_small.txt',
         max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(380, 380)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'gt_labels'])
]

val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='Tokenize', vocab_root='dataset/vocab_small.txt',
         max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(380, 380)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio'])
]

data = dict(train=dict(pipeline=train_pipeline),
            val=dict(pipeline=val_pipeline),
            test=dict(pipeline=val_pipeline))

data = dict(train=dict(pipeline=train_pipeline))
model = dict(modal_used=['image'],
             pretrained=dict(_delete_=True),
             branch_config=dict(image=dict(
                 _delete_=True, type='EffecientNet', arch='efficientnet-b4')),
             head_config=dict(image=dict(dropout_p=0.5, in_dim=1792)))
