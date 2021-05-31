
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
                      out_indices=(3,),
                      frozen_stages=1,
                      norm_cfg=dict(type='BN', requires_grad=True),
                      norm_eval=True,
                      style='pytorch'),
    video_edb=dict(type='FCHead', in_dim=16384, out_dim=1024),
    image_edb=dict(type='FCHead', in_dim=2048, out_dim=1024),
    video_head=dict(type='ClsHead', in_dim=1024, out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss')),
    image_head=dict(type='ClsHead', in_dim=1024, out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss')),
    fusion_head=dict(type='ClsHead', in_dim=1024 * 2, out_dim=82,
                     loss=dict(type='MultiLabelBCEWithLogitsLoss')))