_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(mode=1, modal_used=['image'])

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='LoadAnnotations'),
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

optimizer = dict(_delete_=True, type='SGD', lr=0.02, weight_decay=0.0001)
