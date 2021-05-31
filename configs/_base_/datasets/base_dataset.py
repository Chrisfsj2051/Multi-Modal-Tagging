
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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
              ])
)