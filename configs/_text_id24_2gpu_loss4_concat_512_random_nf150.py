_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/text.py', '_base_/datasets/text.py'
]

norm_cfg = dict(type='BN1d')

model = dict(
    type='SingleBranchModel',
    key='text',
    backbone=dict(
        _delete_=True,
        type='TextCNN',
        vocab_size=21129,
        ebd_dim=300,
        num_filters=150,
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
    ))


optimizer = dict(
    lr=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01, decay_mult=1.0),
        }
    )
)
