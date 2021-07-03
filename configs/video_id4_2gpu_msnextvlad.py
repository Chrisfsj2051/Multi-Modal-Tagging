_base_ = 'video_id4_2gpu.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='MultiScaleNeXtVLAD',
        max_frames_list=[300, 100, 30, 10],
        feature_size=1024,
        cluster_size=128
    ),
    head=dict(
        type='SingleSEHead',
        in_dim=65536,
        gating_reduction=8,
        out_dim=1024,
        dropout_p=0.8,
        cls_head_config=dict(
            type='ClsHead',
            in_dim=1024,
            out_dim=82,
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8))
    )
)