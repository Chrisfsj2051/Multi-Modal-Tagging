_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/single_branch.py', '_base_/datasets/base_dataset.py'
]

norm_cfg = dict(type='SyncBN')
# norm_cfg = dict(type='BN1d')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='CutOut',
         n_holes=3,
         cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.07, 0.07)]),
    dict(type='CutOut',
         n_holes=1,
         cutout_ratio=[(0.2, 0.2), (0.15, 0.15), (0.13, 0.13)]),
    dict(type='AutoAugment',
         policies=[[dict(type='Shear', prob=0.5, level=i)]
                   for i in range(1, 11)] +
         [[dict(type='Rotate', prob=0.5, level=i)] for i in range(1, 11)]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'image', 'text', 'audio', 'meta_info', 'gt_labels'])
]

data = dict(samples_per_gpu=8,
            workers_per_gpu=8,
            train=dict(pipeline=train_pipeline))
model = dict(
    modal_used=['image'],
    # pretrained=dict(_delete_=True),
    branch_config=dict(
        video=dict(norm_cfg=norm_cfg),
        audio=dict(norm_cfg=norm_cfg),
    ),
    head_config=dict(image=dict(dropout_p=0.8, norm_cfg=norm_cfg),
                     video=dict(norm_cfg=norm_cfg),
                     text=dict(norm_cfg=norm_cfg),
                     audio=dict(norm_cfg=norm_cfg),
                     fusion=dict(norm_cfg=norm_cfg)))

optimizer = dict(lr=0.1,
                 paramwise_cfg=dict(
                     custom_keys={
                         'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'text_branch': dict(lr_mult=0.001, decay_mult=1.0),
                         'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'fusion': dict(weight_decay_mult=1.0)
                     }))
