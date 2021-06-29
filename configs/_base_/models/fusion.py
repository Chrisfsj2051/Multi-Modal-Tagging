modal_used = ['image', 'video', 'text', 'audio']

model = dict(
    type='MultiBranchFusionModel',
    modal_dropout_p=dict(text=0.0, video=0.0, image=0.0, audio=0.0),
    fusion_config=dict(
        type='FusionSEHead',
        in_dim=20480,
        gating_reduction=8,
        dropout_p=0.8,
        out_dim=1024,
        cls_head_config=dict(
            type='ClsHead',
            in_dim=1024,
            out_dim=82,
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=4)
        )
    ),
    branch_config=dict(
        video=dict(
            type='SingleBranchModel',
            key='video',
            backbone=dict(
                type='NeXtVLAD',
                feature_size=1024,
                max_frames=300,
                cluster_size=128
            ),
            head=dict(
                type='SingleSEHead',
                in_dim=16384,
                gating_reduction=8,
                out_dim=1024,
                dropout_p=0.8,
                cls_head_config=dict(
                    type='ClsHead',
                    in_dim=1024,
                    out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=4)
                )
            )
        ),
        image=dict(
            type='SingleBranchModel',
            pretrained='torchvision://resnet50',
            key='image',
            backbone=dict(type='ResNet',
                          depth=50,
                          num_stages=4,
                          out_indices=(3,),
                          frozen_stages=1,
                          norm_cfg=dict(type='BN', requires_grad=True),
                          norm_eval=True,
                          style='pytorch'),
            head=dict(
                type='SingleSEHead',
                in_dim=2048,
                gating_reduction=8,
                out_dim=1024,
                dropout_p=0.8,
                cls_head_config=dict(
                    type='ClsHead',
                    in_dim=1024,
                    out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=4))
            )
        ),
        audio=dict(
            type='SingleBranchModel',
            key='audio',
            backbone=dict(
                type='NeXtVLAD',
                feature_size=128,
                max_frames=300,
                cluster_size=64
            ),
            head=dict(
                type='SingleSEHead',
                in_dim=1024,
                gating_reduction=8,
                out_dim=1024,
                dropout_p=0.5,
                cls_head_config=dict(
                    type='ClsHead',
                    in_dim=1024,
                    out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=4))
            )
        ),
        text=dict(
            type='SingleBranchModel',
            key='text',
            backbone=dict(
                type='TwoStreamTextCNN',
                vocab_size=21129,
                ebd_dim=300,
                channel_in=256,
                channel_out=1024,
                filter_size=(2, 3, 4)
            ),
            head=dict(
                type='SingleSEHead',
                in_dim=1024,
                gating_reduction=8,
                out_dim=1024,
                dropout_p=0.5,
                cls_head_config=dict(
                    type='ClsHead',
                    in_dim=1024,
                    out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=4))
            )
        )
    )
)
