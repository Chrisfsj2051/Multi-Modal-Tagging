model = dict(
    type='SingleBranchModel',
    key='text',
    backbone=dict(
        type='TextCNN',
        vocab_size=21129,
        ebd_dim=300,
        num_filters=512,
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
