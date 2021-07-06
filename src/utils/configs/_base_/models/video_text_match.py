model = dict(
    type='ModalMatchModel',
    modal_keys=['video', 'text'],
    backbone_config=dict(
        video=dict(
            type='NeXtVLAD',
            feature_size=1024,
            max_frames=300,
            cluster_size=128
        ),
        text=dict(
            type='TwoStreamTextCNN',
            vocab_size=21129,
            ebd_dim=300,
            channel_in=256,
            channel_out=1024,
            filter_size=(2, 3, 4)
        )
    ),
    head_config=dict(
        type='ModalMatchHead',
        fc_dim1=16384,
        fc_dim2=1024,
        hidden_dim=2048,
        loss=dict(type='BCEWithLogitsLoss')
    ))
