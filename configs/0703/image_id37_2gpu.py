_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/image.py', '_base_/datasets/image.py'
]
# loss weight
optimizer = dict(
    lr=0.1,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01, decay_mult=1.0),
        }
    )
)

norm_cfg = dict(type='SyncBN')
model = dict(
    # backbone=dict(norm_cfg=norm_cfg),
    head=dict(norm_cfg=norm_cfg)
)

data = dict(samples_per_gpu=8)