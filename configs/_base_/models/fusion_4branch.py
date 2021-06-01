model = dict(
    type='MultiBranchesFusionModel',
    pretrained=dict(
        image='../pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
    ),
    # modal_used=['image', 'video', 'text'],
modal_used=['text'],
    branch_config=dict(
        video=dict(
            type='NeXtVLAD',
            feature_size=1024,
            max_frames=300,
            cluster_size=128
        ),
        image=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch'
        ),
        text=dict(
            type='TwoStreamTextCNN',
            vocab_size=9906,
            ebd_dim=300,
            channel_in=128,
            channel_out=1024,
            filter_size=(2, 3, 4)
        )
    ),
    ebd_config=dict(
        video=dict(type='FCHead', in_dim=16384, out_dim=1024),
        image=dict(type='FCHead', in_dim=2048, out_dim=1024),
        text=dict(type='FCHead', in_dim=1024, out_dim=1024)
    ),
    head_config=dict(
        video=dict(type='ClsHead', in_dim=1024, out_dim=82,
                   loss=dict(type='MultiLabelBCEWithLogitsLoss')),
        image=dict(type='ClsHead', in_dim=1024, out_dim=82,
                   loss=dict(type='MultiLabelBCEWithLogitsLoss')),
        text=dict(type='ClsHead', in_dim=1024, out_dim=82,
                   loss=dict(type='MultiLabelBCEWithLogitsLoss')),
        fusion=dict(type='ClsHead', in_dim=1024 * 1, out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss'))
    )
)
