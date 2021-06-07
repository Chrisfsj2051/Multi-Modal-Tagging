_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'pretrained/text0.7178_audio0.6702_video0.7139_image0.7039.pth'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='Tokenize', vocab_root='dataset/vocab_small.txt',
         max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'gt_labels'])
]

data = dict(train=dict(pipeline=train_pipeline))


model = dict(mode=2,
             head_config=dict(
                 fusion=dict(
                     _delete_=True,
                     type='ClsHead',
                     out_dim=82,
                     in_dim=1024,
                     loss=dict(type='MultiLabelBCEWithLogitsLoss'))
             ))
optimizer = dict(_delete_=True,
                 type='SGD',
                 lr=0.05,
                 momentum=0.9,
                 weight_decay=0.0001)


