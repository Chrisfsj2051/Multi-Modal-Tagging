_base_ = '_fusion_id51_2gpu.py'
load_from = 'pretrained/image40_text32_video46_audio13_full.pth'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=512, concat_ocr_asr=True, random_permute=True),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='VideoResamplePad', seq_len=120),
    dict(
        type='FrameRandomErase',
        key_fields=['video'],
        aug_num_frame=0.1,
        aug_max_len=1,
        aug_num_block=10,
        aug_max_size=60),
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
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'image', 'text', 'audio', 'meta_info', 'gt_labels'])
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
        ],
    )
)
