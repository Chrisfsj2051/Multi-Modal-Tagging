_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/fusion.py', '_base_/datasets/fusion.py'
]
load_from = None
data = dict(samples_per_gpu=8, workers_per_gpu=8)
# data = dict(samples_per_gpu=2, workers_per_gpu=1)
# custom_hooks = [
#     dict(type='FreezeParamHook',
#          param_pattern=['video', 'audio', 'image', 'text'],
#          eval_pattern=['video', 'audio', 'image', 'text'],
#          freeze_iters=1000)
# ]
# find_unused_parameters=True

model = dict(
    modal_dropout_p=dict(text=0.1, video=0.1, image=0.1, audio=0.1),
    fusion_config=dict(
        _delete_=True,
        type='TransformerHead',
        in_dim=dict(
            video=16384,
            audio=1024,
            image=2048,
            text=1024
        ),
        num_layers=5,
        hidden_dim=512,
        dropout_p=0.9,
        num_head=8,
        cls_head_config=dict(
            dropout_p=0.8,
            type='ClsHead',
            in_dim=4 * 512,
            out_dim=82,
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8)
        )
    )
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'image_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'text_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'video_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'audio_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
        }))
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[7000])
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
