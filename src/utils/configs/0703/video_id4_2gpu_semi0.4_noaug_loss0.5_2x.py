_base_ = 'video_id4_2gpu.py'

train_total_iters = 20000

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[train_total_iters // 3 , 2 * train_total_iters // 3]
)
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)


model = dict(
    type='SemiSingleBranchModel',
    gt_thr=0.4,
    unlabeled_loss_weight=0.5
)

custom_hooks = [
    dict(
        type='SemiEMAHook',
        burnin_iters=1000,
        ema_eval=False,
        momentum=0.001
    )
]


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    # dict(type='PhotoMetricDistortion',
    #      brightness_delta=32,
    #      contrast_range=(0.5, 1.5),
    #      saturation_range=(0.5, 1.5),
    #      hue_delta=18),
    # dict(type='CutOut',
    #      n_holes=3,
    #      cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.07, 0.07)]),
    # dict(type='CutOut',
    #      n_holes=1,
    #      cutout_ratio=[(0.2, 0.2), (0.15, 0.15), (0.13, 0.13)]),
    # dict(type='AutoAugment',
    #      policies=[[dict(type='Shear', prob=0.5, level=i)]
    #                for i in range(1, 11)] +
    #      [[dict(type='Rotate', prob=0.5, level=i)] for i in range(1, 11)]),
    # dict(type='FrameRandomErase',
    #      key_fields=['video'],
    #      aug_num_frame=9,
    #      aug_max_len=3,
    #      aug_num_block=3,
    #      aug_max_size=30),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'meta_info', 'gt_labels'])
]


weak_train_pipeline_1 = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_test_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/test_5k'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info'])
]

strong_train_pipeline_1 = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_test_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/test_5k'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    # dict(type='PhotoMetricDistortion',
    #      brightness_delta=32,
    #      contrast_range=(0.5, 1.5),
    #      saturation_range=(0.5, 1.5),
    #      hue_delta=18),
    # dict(type='CutOut',
    #      n_holes=3,
    #      cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.07, 0.07)]),
    # dict(type='CutOut',
    #      n_holes=1,
    #      cutout_ratio=[(0.2, 0.2), (0.15, 0.15), (0.13, 0.13)]),
    # dict(type='AutoAugment',
    #      policies=[[dict(type='Shear', prob=0.5, level=i)]
    #                for i in range(1, 11)] +
    #      [[dict(type='Rotate', prob=0.5, level=i)] for i in range(1, 11)]),
    # dict(type='FrameRandomErase',
    #      key_fields=['video'],
    #      aug_num_frame=30,
    #      aug_max_len=10,
    #      aug_num_block=10,
    #      aug_max_size=100),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'meta_info'])
]

weak_train_pipeline_2 = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_test_5k_2nd/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/test_5k_2nd'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info'])
]

strong_train_pipeline_2=[
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_test_5k_2nd/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/test_5k_2nd'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    # dict(type='PhotoMetricDistortion',
    #      brightness_delta=32,
    #      contrast_range=(0.5, 1.5),
    #      saturation_range=(0.5, 1.5),
    #      hue_delta=18),
    # dict(type='CutOut',
    #      n_holes=3,
    #      cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.07, 0.07)]),
    # dict(type='CutOut',
    #      n_holes=1,
    #      cutout_ratio=[(0.2, 0.2), (0.15, 0.15), (0.13, 0.13)]),
    # dict(type='AutoAugment',
    #      policies=[[dict(type='Shear', prob=0.5, level=i)]
    #                for i in range(1, 11)] +
    #      [[dict(type='Rotate', prob=0.5, level=i)] for i in range(1, 11)]),
    # dict(type='FrameRandomErase',
    #      key_fields=['video'],
    #      aug_num_frame=30,
    #      aug_max_len=10,
    #      aug_num_block=10,
    #      aug_max_size=100),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'meta_info'])
]

data = dict(
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
                    type='TaggingDatasetWithAugs',
                    ann_file='dataset/tagging/GroundTruth/datafile/test.txt',  # change to test
                    label_id_file='dataset/tagging/label_super_id.txt',
                    pipeline=weak_train_pipeline_1,
                    strong_pipeline = strong_train_pipeline_1,
                    test_mode=True
                ),
                dict(
                    type='TaggingDatasetWithAugs',
                    ann_file='dataset/tagging/GroundTruth/datafile/test_2nd.txt',  # change to test
                    label_id_file='dataset/tagging/label_super_id.txt',
                    pipeline=weak_train_pipeline_2,
                    strong_pipeline = strong_train_pipeline_2,
                    test_mode=True
                )
            ]
        )
    )
)
