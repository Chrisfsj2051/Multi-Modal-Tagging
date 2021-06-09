modal_used = ['image', 'video', 'text', 'audio']

model = dict(
    type='MultiBranchesFusionModel',
    pretrained=dict(
        image='pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth'),
    modal_dropout_p=dict(text=0.0, video=0.0, image=0.0, audio=0.0),
    use_batch_norm=False,
    mode=3,
    modal_used=modal_used,
    branch_config=dict(
        video=dict(type='NeXtVLAD',
                   feature_size=1024,
                   max_frames=300,
                   cluster_size=128),
        audio=dict(type='NeXtVLAD',
                   feature_size=128,
                   max_frames=300,
                   cluster_size=64),
        image=dict(type='ResNet',
                   depth=50,
                   num_stages=4,
                   out_indices=(3,),
                   frozen_stages=1,
                   norm_cfg=dict(type='BN', requires_grad=True),
                   norm_eval=True,
                   style='pytorch'),
        text=dict(type='TwoStreamTextCNN',
                  vocab_size=9906,
                  ebd_dim=300,
                  channel_in=256,
                  channel_out=1024,
                  filter_size=(2, 3, 4),
                  dropout_p=0.0)),
    # ebd_config=dict(video=dict(type='FCHead', in_dim=16384, out_dim=1024),
    #                 image=dict(type='FCHead', in_dim=2048, out_dim=1024),
    #                 text=dict(type='FCHead', in_dim=1024, out_dim=1024),
    #                 audio=dict(type='FCHead', in_dim=1024, out_dim=1024)),
    head_config=dict(
        video=dict(
            type='SEHead',
            in_dim=16384,
            gating_reduction=8,
            out_dim=1024,
            dropout_p=0.8,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss')
            ),
        ),
        image=dict(
            type='SEHead',
            in_dim=2048,
            gating_reduction=8,
            out_dim=1024,
            dropout_p=0.5,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss')
            ),
        ),
        text=dict(
            type='SEHead',
            in_dim=1024,
            gating_reduction=8,
            out_dim=1024,
            dropout_p=0.5,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss')
            ),
        ),
        audio=dict(
            type='SEHead',
            in_dim=1024,
            dropout_p=0.5,
            gating_reduction=8,
            out_dim=1024,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss')
            ),
        ),
        fusion=dict(
            type='SEHead',
            in_dim=20480,
            gating_reduction=8,
            dropout_p=0.8,
            out_dim=1024,
            cls_head_config=dict(
                type='ClsHead',
                in_dim=1024,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss')
                # type='HMCHead',
                # feat_dim=512,
                # out_dim=82,
                # in_dim=1024,
                # loss=dict(type='MultiLabelBCEWithLogitsLoss',
                #           apply_onehot=False,
                #           with_sigmoid=False),
                # label_id_file='dataset/tagging/label_super_id.txt'
            )
        )
    )
)
