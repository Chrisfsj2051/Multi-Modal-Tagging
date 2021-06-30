_base_ = 'video_id4_2gpu.py'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

model = dict(
    type='SemiSingleBranchModel',
    gt_thr=0.9
)

train_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info', 'gt_labels'])
]

extra_train_pipeline_1 = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_test_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/test_5k'))),
    # dict(type='LoadAnnotations',
    #      replace_dict=dict(video=(
    #          'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
    #          'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info'])
]

extra_train_pipeline_2 = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_test_5k_2nd/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/test_5k_2nd'))),
    # dict(type='LoadAnnotations',
    #      replace_dict=dict(video=(
    #          'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
    #          'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        _delete_=True,
        type='TwoStreamDataset',
        main_dataset_config=dict(
            type='TaggingDataset',
            ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
            label_id_file='dataset/tagging/label_super_id.txt',
            pipeline=train_pipeline
        ),
        extra_dataset_config=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='TaggingDataset',
                    ann_file='dataset/tagging/GroundTruth/datafile/test.txt',  # change to test
                    # ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
                    label_id_file='dataset/tagging/label_super_id.txt',
                    pipeline=extra_train_pipeline_1,
                    test_mode=True
                ),
                dict(
                    type='TaggingDataset',
                    ann_file='dataset/tagging/GroundTruth/datafile/test_2nd.txt',  # change to test
                    # ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
                    label_id_file='dataset/tagging/label_super_id.txt',
                    pipeline=extra_train_pipeline_2,
                    test_mode=True
                )
            ]
        )
    ),
)
