model = dict(
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
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8))
    ))
