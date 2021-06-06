checkpoint_config = dict(interval=1000)
evaluation = dict(interval=1000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = []
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained/text0.717_audio0.675_video0.707_image0.706.pth'
resume_from = None
workflow = [('train', 1)]
optimizer = dict(type='SGD', lr=0.05, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[8000, 9000])
runner = dict(type='IterBasedRunner', max_iters=10000)
modal_used = ['image', 'video', 'text', 'audio']
model = dict(
    type='MultiBranchesFusionModel',
    pretrained=dict(
        image='pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth'),
    modal_dropout_p=dict(text=0.5, video=0.5, image=0.5, audio=0.5),
    use_batch_norm=False,
    mode=2,
    modal_used=['image', 'video', 'text', 'audio'],
    branch_config=dict(video=dict(type='NeXtVLAD',
                                  feature_size=1024,
                                  max_frames=300,
                                  cluster_size=128),
                       audio=dict(type='NeXtVLAD',
                                  feature_size=128,
                                  max_frames=300,
                                  cluster_size=64),
                       image=dict(type='ResNet',
                                  depth=50,
                                  num_stages=4,
                                  out_indices=(3, ),
                                  frozen_stages=1,
                                  norm_cfg=dict(type='BN', requires_grad=True),
                                  norm_eval=True,
                                  style='pytorch'),
                       text=dict(type='TwoStreamTextCNN',
                                 vocab_size=9906,
                                 ebd_dim=300,
                                 channel_in=128,
                                 channel_out=1024,
                                 filter_size=(2, 3, 4),
                                 dropout_p=0.5)),
    ebd_config=dict(video=dict(type='FCHead', in_dim=16384, out_dim=1024),
                    image=dict(type='FCHead', in_dim=2048, out_dim=1024),
                    text=dict(type='FCHead', in_dim=1024, out_dim=1024),
                    audio=dict(type='FCHead', in_dim=1024, out_dim=1024)),
    attn_config=dict(
        type='SEHead',
        in_dim=20480,
        gating_reduction=8,
        out_dim=1024,
        # input_dropout_p=0.2
    ),
    head_config=dict(video=dict(type='ClsHead',
                                in_dim=1024,
                                out_dim=82,
                                loss=dict(type='MultiLabelBCEWithLogitsLoss')),
                     image=dict(type='ClsHead',
                                in_dim=1024,
                                out_dim=82,
                                loss=dict(type='MultiLabelBCEWithLogitsLoss')),
                     text=dict(type='ClsHead',
                               in_dim=1024,
                               out_dim=82,
                               loss=dict(type='MultiLabelBCEWithLogitsLoss')),
                     audio=dict(type='ClsHead',
                                in_dim=1024,
                                out_dim=82,
                                loss=dict(type='MultiLabelBCEWithLogitsLoss')),
                     fusion=dict(
                         type='ClsHead',
                         in_dim=1024,
                         out_dim=82,
                         loss=dict(type='MultiLabelBCEWithLogitsLoss'))))
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='Tokenize', vocab_root='dataset/vocab_small.txt',
         max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='Tokenize', vocab_root='dataset/vocab_small.txt',
         max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio'])
]
data = dict(samples_per_gpu=2,
            workers_per_gpu=2,
            train=dict(
                type='TaggingDataset',
                ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
                label_id_file='dataset/tagging/label_super_id.txt',
                pipeline=[
                    dict(type='LoadAnnotations'),
                    dict(type='Tokenize',
                         vocab_root='dataset/vocab_small.txt',
                         max_length=256),
                    dict(type='Pad',
                         video_pad_size=(300, 1024),
                         audio_pad_size=(300, 128)),
                    dict(type='Resize', size=(224, 224)),
                    dict(type='Normalize',
                         mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375]),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect',
                         keys=['video', 'image', 'text', 'audio', 'gt_labels'])
                ]),
            val=dict(type='TaggingDataset',
                     ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
                     label_id_file='dataset/tagging/label_super_id.txt',
                     pipeline=[
                         dict(type='LoadAnnotations'),
                         dict(type='Tokenize',
                              vocab_root='dataset/vocab_small.txt',
                              max_length=256),
                         dict(type='Pad',
                              video_pad_size=(300, 1024),
                              audio_pad_size=(300, 128)),
                         dict(type='Resize', size=(224, 224)),
                         dict(type='Normalize',
                              mean=[123.675, 116.28, 103.53],
                              std=[58.395, 57.12, 57.375]),
                         dict(type='DefaultFormatBundle'),
                         dict(type='Collect',
                              keys=['video', 'image', 'text', 'audio'])
                     ]),
            test=dict(type='TaggingDataset',
                      ann_file='dataset/tagging/GroundTruth/datafile/test.txt',
                      label_id_file='dataset/tagging/label_id.txt',
                      pipeline=[
                          dict(type='LoadAnnotations'),
                          dict(type='Tokenize',
                               vocab_root='dataset/vocab_small.txt',
                               max_length=256),
                          dict(type='Pad',
                               video_pad_size=(300, 1024),
                               audio_pad_size=(300, 128)),
                          dict(type='Resize', size=(224, 224)),
                          dict(type='Normalize',
                               mean=[123.675, 116.28, 103.53],
                               std=[58.395, 57.12, 57.375]),
                          dict(type='DefaultFormatBundle'),
                          dict(type='Collect',
                               keys=['video', 'image', 'text', 'audio'])
                      ]))
work_dir = './work_dirs/mode2_se_dropmodal_0.3'
gpu_ids = range(0, 2)
