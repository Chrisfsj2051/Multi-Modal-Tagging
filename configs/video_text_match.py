_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/video_text_match.py', '_base_/datasets/video_text_match.py'
]

# evaluation = dict(interval=100)
# data = dict(samples_per_gpu=2, workers_per_gpu=0)
# checkpoint_config = dict(interval=10000)

optimizer = dict(
    _delete_=True,
    type='Adam',
    amsgrad=True,
    lr=0.00002,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'image_branch': dict(lr_mult=1, decay_mult=1.0),
                     'text_branch': dict(lr_mult=1, decay_mult=1.0),
                     'video_branch': dict(lr_mult=1, decay_mult=1.0),
                     'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                     'fusion': dict(weight_decay_mult=1.0)})
)