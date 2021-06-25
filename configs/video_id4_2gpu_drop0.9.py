_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/video.py', '_base_/datasets/video.py'
]

norm_cfg = dict(type='BN1d')

model = dict(head=dict(norm_cfg=norm_cfg, dropout_p=0.9))

optimizer = dict(
    lr=0.01,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.01, decay_mult=1.0),
    }))

data = dict(samples_per_gpu=8, workers_per_gpu=8)
