_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/single_branch.py', '_base_/datasets/base_dataset.py'
]

norm_cfg = dict(type='SyncBN')
# norm_cfg = dict(type='BN1d')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'text', 'meta_info', 'gt_labels'])
]

val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'text', 'meta_info'])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='PretrainMatchDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/text.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=train_pipeline),
    val=dict(type='PretrainMatchDataset',
             ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
             label_id_file='dataset/tagging/label_super_id.txt',
             pipeline=val_pipeline),
    test=dict(type='PretrainMatchDataset',
              ann_file='dataset/tagging/GroundTruth/datafile/test.txt',
              label_id_file='dataset/tagging/label_super_id.txt',
              pipeline=val_pipeline))

model = dict(
    modal_used=['video', 'text'],
    type='PretrainMatchModel',
    branch_config=dict(
        video=dict(norm_cfg=norm_cfg),
        audio=dict(norm_cfg=norm_cfg),
    ),
    head_config=dict(
        image=dict(dropout_p=0.8, norm_cfg=norm_cfg),
        video=dict(norm_cfg=norm_cfg),
        text=dict(norm_cfg=norm_cfg),
        audio=dict(norm_cfg=norm_cfg),
        fusion=dict(
            norm_cfg=norm_cfg,
            type='SEHead',
            in_dim=17408,
            gating_reduction=8,
            dropout_p=0.2,
            out_dim=1024,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=1,
                loss=dict(type='BCEWithLogitsLoss')
            )
        )
    )
)

optimizer = dict(
    lr=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
            'text_branch': dict(lr_mult=0.01, decay_mult=1.0),
            'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
            'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
            'fusion': dict(weight_decay_mult=1.0)
        }))
