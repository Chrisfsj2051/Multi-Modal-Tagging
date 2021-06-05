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
         aug_num_frame=30,
         aug_max_len=10,
         aug_num_block=10,
         aug_max_size=100),
    dict(type='FrameRandomSwap',
         key_fields=['video'],
         aug_num_frame=30,
         aug_max_len=10,
         aug_num_block=10,
         aug_max_size=300),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'gt_labels'])
]

val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='Tokenize', vocab_root='dataset/vocab_small.txt',
         max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio'])
]

data = dict(samples_per_gpu=2,
            workers_per_gpu=2,
            train=dict(
                type='TaggingDataset',
                ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
                label_id_file='dataset/tagging/label_id.txt',
                pipeline=train_pipeline),
            val=dict(type='TaggingDataset',
                     ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
                     label_id_file='dataset/tagging/label_id.txt',
                     pipeline=val_pipeline),
            test=dict(type='TaggingDataset',
                      ann_file='dataset/tagging/GroundTruth/datafile/test.txt',
                      label_id_file='dataset/tagging/label_id.txt',
                      pipeline=val_pipeline))
