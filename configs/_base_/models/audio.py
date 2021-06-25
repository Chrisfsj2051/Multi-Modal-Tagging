model = dict(
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
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8))
    ))
