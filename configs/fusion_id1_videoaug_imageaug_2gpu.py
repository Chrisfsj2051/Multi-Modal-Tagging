_base_ = 'fusion_id1.py'
load_from = 'pretrained/text19_audio1_video1_image34.pth'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
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
    dict(type='FrameRandomErase',
         key_fields=['video'],
         aug_num_frame=9,
         aug_max_len=3,
         aug_num_block=3,
         aug_max_size=30),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'image', 'text', 'audio', 'meta_info', 'gt_labels'])
]
data = dict(samples_per_gpu=8, train=dict(pipeline=train_pipeline))
