_base_ = '_video_id45_2gpu_2x.py'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='VideoResamplePad', seq_len=120),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info', 'gt_labels'])
]

data = dict(
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(type='TaggingDataset',
                 ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
                 label_id_file='dataset/tagging/label_super_id.txt',
                 pipeline=train_pipeline),
            dict(type='TaggingDataset',
                 ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
                 label_id_file='dataset/tagging/label_super_id.txt',
                 pipeline=train_pipeline),
        ]
    )
)

train_total_iters = 22000
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[train_total_iters // 3, 2 * train_total_iters // 3])
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)