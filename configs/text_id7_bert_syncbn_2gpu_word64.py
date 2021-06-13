_base_ = 'text_id6_bert.py'

norm_cfg = dict(type='SyncBN')
# norm_cfg = dict(type='BN1d')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=64),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'image', 'text', 'audio', 'meta_info', 'gt_labels'])
]

data = dict(samples_per_gpu=8,
            workers_per_gpu=8,
            train=dict(pipeline=train_pipeline))

model = dict(modal_used=['text'],
             branch_config=dict(text=dict(_delete_=True, type='Bert')),
             head_config=dict(image=dict(dropout_p=0.8, norm_cfg=norm_cfg),
                              video=dict(norm_cfg=norm_cfg),
                              text=dict(norm_cfg=norm_cfg, in_dim=768),
                              audio=dict(norm_cfg=norm_cfg),
                              fusion=dict(norm_cfg=norm_cfg)))
