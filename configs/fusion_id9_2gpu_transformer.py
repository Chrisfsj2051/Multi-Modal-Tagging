_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/fusion.py', '_base_/datasets/fusion.py'
]
load_from = 'pretrained/image37_text23_video4_audio3.pth'
data = dict(samples_per_gpu=8, workers_per_gpu=8)

model = dict(
    fusion_config=dict(
        _delete_=True,
        type='TransformerHead',
        in_dim=dict(
            video=16384,
            audio=1024,
            image=2048,
            text=1024
        ),
        num_layers=4,
        hidden_dim=512,
        dropout_p=0.8,
        num_head=4,
        cls_head_config=dict(
            type='ClsHead',
            in_dim=4 * 512,
            out_dim=82,
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8)
        )
    )
)

optimizer = dict(
    _delete_=True,
    type='Adam',
    amsgrad=True,
    lr=0.01,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'image_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'text_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'video_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'audio_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
        }))

optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
