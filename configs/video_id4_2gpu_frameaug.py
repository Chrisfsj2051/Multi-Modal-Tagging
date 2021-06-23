_base_ = 'video_id4_2gpu.py'

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
    dict(type='FrameRandomErase',
         key_fields=['video'],
         aug_num_frame=30,
         aug_max_len=10,
         aug_num_block=10,
         aug_max_size=100),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info', 'gt_labels'])
]

data = dict(train=dict(pipeline=train_pipeline))
