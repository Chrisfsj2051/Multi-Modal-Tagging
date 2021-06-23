_base_ = 'video_id3_4gpu.py'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(150, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info', 'gt_labels'])
]

val_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(150, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info'])
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

model = dict(type='SingleBranchModel',
             key='video',
             backbone=dict(type='NeXtVLAD',
                           feature_size=1024,
                           max_frames=100,
                           cluster_size=128))
