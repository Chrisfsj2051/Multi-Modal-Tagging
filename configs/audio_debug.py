_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/audio.py', '_base_/datasets/audio.py'
]

norm_cfg = dict(type='BN1d')

model = dict(
    head=dict(norm_cfg=norm_cfg)
)

optimizer = dict(
    lr=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01, decay_mult=1.0),
        }
    )
)
