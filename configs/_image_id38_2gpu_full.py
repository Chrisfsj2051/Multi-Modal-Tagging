_base_ = '_image_id38_2gpu.py'

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
    dict(type='Collect', keys=['image', 'meta_info', 'gt_labels'])
]


val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['image', 'meta_info'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
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
        ],
    ),
    val=dict(type='TaggingDataset',
             ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
             label_id_file='dataset/tagging/label_super_id.txt',
             pipeline=val_pipeline),
    test=dict(type='TaggingDataset',
              ann_file='dataset/tagging/GroundTruth/datafile/test_2nd.txt',
              label_id_file='dataset/tagging/label_super_id.txt',
              pipeline=val_pipeline))
