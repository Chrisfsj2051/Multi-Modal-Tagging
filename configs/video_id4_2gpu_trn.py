_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/video.py', '_base_/datasets/video.py'
]

norm_cfg = dict(type='BN1d')

model = dict(
    backbone=dict(
        _delete_=True,
        type='TRN',
        num_segment=300,
        input_dim=1024,
        output_dim=2048
    ),
    head=dict(
        norm_cfg=norm_cfg,
        in_dim=2048
    ),
)

optimizer = dict(
    lr=0.01,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.01, decay_mult=1.0),
    }))

# data = dict(samples_per_gpu=2, workers_per_gpu=1)
data = dict(samples_per_gpu=8, workers_per_gpu=8)
