_base_ = 'audio_id3_4gpu.py'
# optimizer_config = dict(_delete_=True, grad_clip=None)
optimizer = dict(type='Adam',
                 amsgrad=True,
                 lr=0.01,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(
                     custom_keys={
                         'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'text_branch': dict(lr_mult=0.001, decay_mult=1.0),
                         'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'audio_branch': dict(lr_mult=0.1, decay_mult=1.0),
                         'fusion': dict(weight_decay_mult=1.0)
                     }))
model = dict(
    type='SingleBranchModel',
    key='audio',
    pretrained='pretrained/Cnn14_mAP=0.431.pth',
    backbone=dict(_delete_=True,
                  type='PANNS',
                  sample_rate=3400,
                  window_size=1024,
                  hop_size=320,
                  mel_bins=64,
                  fmin=50,
                  fmax=14000),
    head=dict(
        type='SingleSEHead',
        in_dim=2048,
        #             dropout_p=0.8
    ))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type='LoadAnnotationsWithWAV',
        replace_dict=dict(
            audio=('tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging',
                   'extracted_audio_frame/train_5k'))),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(34000, )),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['audio', 'meta_info', 'gt_labels'])
]

val_pipeline = [
    dict(
        type='LoadAnnotationsWithWAV',
        replace_dict=dict(
            audio=('tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging',
                   'extracted_audio_frame/train_5k'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(34000, )),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['audio', 'meta_info'])
]

test_pipeline = [
    dict(
        type='LoadAnnotations',
        replace_dict=dict(
            audio=('tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging',
                   'extracted_audio_frame/test_5k_2nd'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(34000, )),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['audio', 'meta_info'])
]

data = dict(samples_per_gpu=8,
            workers_per_gpu=8,
            train=dict(pipeline=train_pipeline),
            val=dict(pipeline=val_pipeline),
            test=dict(pipeline=test_pipeline))
