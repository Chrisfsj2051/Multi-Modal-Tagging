_base_ = 'id11.py'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='AutoAugment',
         policies=[[
             dict(type='Translate',
                  max_translate_offset=70.,
                  prob=0.5,
                  level=i,
                  direction='horizontal')
         ] for i in range(1, 11)] + [[
             dict(type='Translate',
                  max_translate_offset=70.,
                  prob=0.5,
                  level=i,
                  direction='vertical')
         ] for i in range(1, 11)] + [[dict(type='Shear', prob=0.5, level=i)]
                                     for i in range(1, 11)] +
         [[dict(type='Rotate', prob=0.5, level=i)] for i in range(1, 11)] + [[
             dict(type='CutOut',
                  n_holes=5,
                  cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.15, 0.15),
                                (0.2, 0.2), (0.25, 0.25), (0.3, 0.3)])
         ]]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'image', 'text', 'audio', 'meta_info', 'gt_labels'])
]

data = dict(train=dict(pipeline=train_pipeline))
