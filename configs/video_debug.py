_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/single_video.py', '_base_/datasets/base_dataset.py'
]

norm_cfg = dict(type='BN1d')

model = dict(
    modal_used=['video'],
             backbone=dict(
                 video=dict(norm_cfg=norm_cfg),
                 audio=dict(norm_cfg=norm_cfg),
             ),
             head=dict(dropout_p=0.8, norm_cfg=norm_cfg)
             )

optimizer = dict(
    lr=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01, decay_mult=1.0),
        }
    )
)
