_base_ = 'id7.py'

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
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'gt_labels'])
]

data = dict(train=dict(pipeline=train_pipeline))

model = dict(modal_used=['image'],
             branch_config=dict(video=dict(norm_cfg=dict(type='SyncBN')),
                                audio=dict(norm_cfg=dict(type='SyncBN'))),
             head_config=dict(image=dict(dropout_p=0.8,
                                         norm_cfg=dict(type='SyncBN')),
                              video=dict(norm_cfg=dict(type='SyncBN')),
                              text=dict(norm_cfg=dict(type='SyncBN')),
                              audio=dict(norm_cfg=dict(type='SyncBN')),
                              fusion=dict(norm_cfg=dict(type='SyncBN'))))
