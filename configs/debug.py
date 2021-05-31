_base_ = ['_base_/default_runtime.py', '_base_/schedules/schedule_1x.py']
norm_cfg = dict(type='BN', requires_grad=False)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])
checkpoint_config = dict(interval=1000)
model = dict(
    type='MultiBranchesFusionModel',
    pretrained=dict(
        image=
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
    ),
    video_branch=dict(type='NeXtVLAD',
                      feature_size=1024,
                      max_frames=300,
                      cluster_size=128),
    image_branch=dict(type='ResNet',
                      depth=50,
                      num_stages=4,
                      out_indices=(3, ),
                      frozen_stages=1,
                      norm_cfg=dict(type='BN', requires_grad=True),
                      norm_eval=True,
                      style='pytorch'),
    head=dict(type='ClsHead', loss=dict(type='MultiLabelBCEWithLogitsLoss')))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(type='TaggingDataset',
               ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
               label_id_file='dataset/tagging/label_id.txt',
               pipeline=[
                   dict(type='LoadAnnotations'),
                   dict(type='Pad', size=(300, 1024)),
                   dict(type='Resize', size=(224, 224)),
                   dict(type='Normalize', **img_norm_cfg),
                   dict(type='DefaultFormatBundle'),
                   dict(type='Collect', keys=['video', 'image', 'gt_labels'])
               ]),
    val=dict(type='TaggingDataset',
             ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
             label_id_file='dataset/tagging/label_id.txt',
             pipeline=[
                 dict(type='LoadAnnotations'),
                 dict(type='Pad', size=(300, 1024)),
                 dict(type='Resize', size=(224, 224)),
                 dict(type='Normalize', **img_norm_cfg),
                 dict(type='DefaultFormatBundle'),
                 dict(type='Collect', keys=['video', 'image'])
             ]),
    test=dict(type='TaggingDataset',
             ann_file='dataset/tagging/GroundTruth/datafile/test.txt',
             label_id_file='dataset/tagging/label_id.txt',
             pipeline=[
                 dict(type='LoadAnnotations'),
                 dict(type='Pad', size=(300, 1024)),
                 dict(type='Resize', size=(224, 224)),
                 dict(type='Normalize', **img_norm_cfg),
                 dict(type='DefaultFormatBundle'),
                 dict(type='Collect', keys=['video', 'image'])
             ]),
)
