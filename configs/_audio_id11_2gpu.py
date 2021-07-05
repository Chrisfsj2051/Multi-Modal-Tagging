_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/audio.py', '_base_/datasets/audio.py'
]

optimizer = dict(
    lr=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01, decay_mult=1.0),
        }
    )
)

model = dict(
    head=dict(
        dropout_p=0.5,
        cls_head_config=dict(
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=4))
    )
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='FrameRandomErase',
         key_fields=['video'],
         aug_num_frame=0.1,
         aug_max_len=1,
         aug_num_block=2,
         aug_max_size=5),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['audio', 'meta_info', 'gt_labels'])
]

val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['audio', 'meta_info'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(type='TaggingDataset',
               ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
               label_id_file='dataset/tagging/label_super_id.txt',
               pipeline=train_pipeline),
    val=dict(type='TaggingDataset',
             ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
             label_id_file='dataset/tagging/label_super_id.txt',
             pipeline=val_pipeline),
    test=dict(type='TaggingDataset',
              ann_file='dataset/tagging/GroundTruth/datafile/test_2nd.txt',
              label_id_file='dataset/tagging/label_super_id.txt',
              pipeline=val_pipeline))
