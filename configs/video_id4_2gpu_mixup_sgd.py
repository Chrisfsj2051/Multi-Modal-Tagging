_base_ = 'video_id4_2gpu.py'

# data = dict(samples_per_gpu=8, workers_per_gpu=1)
checkpoint_config = dict(interval=10000)
model = dict(
    head=dict(
        type='SingleMixupSEHead',
        cls_head_config=dict(
            type='MixupClsHead',
            loss=dict(type='MixupMultiLabelBCEWithLogitsLoss')
        )
    )
)

# train_total_iters = 10000
#
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[train_total_iters // 3, 2 * train_total_iters // 3]
# )
# runner = dict(type='IterBasedRunner', max_iters=train_total_iters)
optimizer = dict(
    _delete_=True,
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.01, decay_mult=1.0)))
)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[8000, 9000])
runner = dict(type='IterBasedRunner', max_iters=10000)
