_base_ = 'audio_id3_4gpu.py'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type='LoadAnnotations',
        replace_dict=dict(
            audio=('tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging',
                   'extracted_audio_feats/train_5k'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info', 'gt_labels'])
]

val_pipeline = [
    dict(
        type='LoadAnnotations',
        replace_dict=dict(
            audio=('tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging',
                   'extracted_audio_feats/train_5k'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info'])
]

test_pipeline = [
    dict(
        type='LoadAnnotations',
        replace_dict=dict(
            audio=('tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging',
                   'extracted_audio_feats/test_5k_2nd'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'meta_info'])
]

data = dict(samples_per_gpu=8,
            workers_per_gpu=8,
            train=dict(pipeline=train_pipeline),
            val=dict(pipeline=val_pipeline),
            test=dict(pipeline=test_pipeline))
