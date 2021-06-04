modal_used = ['image', 'video', 'text', 'audio']

model = dict(
    type='MultiBranchesFusionModel',
    pretrained=dict(
        image='pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
    ),
    modal_dropout_p=dict(text=0.3, video=0.3, image=0.3, audio=0.3),
    use_batch_norm=False,
    mode=3,
    modal_used=modal_used,
    branch_config=dict(
        video=dict(
            type='NeXtVLAD',
            feature_size=1024,
            max_frames=300,
            cluster_size=128
        ),
        audio=dict(
            type='NeXtVLAD',
            feature_size=128,
            max_frames=300,
            cluster_size=64
        ),
        image=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            # plugins=[
            #     dict(
            #         cfg=dict(
            #             type='DropBlock',
            #             drop_prob=0.1,
            #             block_size=5,
            #             postfix='_1'),
            #         stages=(False, False, True, True),
            #         position='after_conv1'),
            #     dict(
            #         cfg=dict(
            #             type='DropBlock',
            #             drop_prob=0.1,
            #             block_size=5 ,
            #             postfix='_2'),
            #         stages=(False, False, True, True),
            #         position='after_conv2'),
            #     dict(
            #         cfg=dict(
            #             type='DropBlock',
            #             drop_prob=0.1,
            #             block_size=5,
            #             postfix='_3'),
            #         stages=(False, False, True, True),
            #         position='after_conv3')
            # ]
        ),
        text=dict(
            type='TwoStreamTextCNN',
            vocab_size=9906,
            ebd_dim=300,
            channel_in=128,
            channel_out=1024,
            filter_size=(2, 3, 4),
        )
    ),
    ebd_config=dict(
        video=dict(type='FCHead', in_dim=16384, out_dim=1024),
        image=dict(type='FCHead', in_dim=2048, out_dim=1024),
        text=dict(type='FCHead', in_dim=1024, out_dim=1024),
        audio=dict(type='FCHead', in_dim=1024, out_dim=1024)
    ),
    attn_config=dict(
        type='SEHead', in_dim=20480,
        gating_reduction=8, out_dim=1024,
        # input_dropout_p=0.2
    ),
    head_config=dict(
        video=dict(type='ClsHead', in_dim=1024, out_dim=82,
                   loss=dict(type='MultiLabelBCEWithLogitsLoss')),
        image=dict(type='ClsHead', in_dim=1024, out_dim=82,
                   loss=dict(type='MultiLabelBCEWithLogitsLoss')),
        text=dict(type='ClsHead', in_dim=1024, out_dim=82,
                  loss=dict(type='MultiLabelBCEWithLogitsLoss')),
        audio=dict(type='ClsHead', in_dim=1024, out_dim=82,
                   loss=dict(type='MultiLabelBCEWithLogitsLoss')),
        fusion=dict(type='ClsHead', in_dim=1024, out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss'))
    )
)
