_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/fusion.py', '_base_/datasets/fusion.py'
]
load_from = None

model = dict(
    modal_dropout_p=dict(text=0.1, video=0.1, image=0.1, audio=0.1),
)

checkpoint_config = dict(interval=10)

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
data = dict(samples_per_gpu=2, workers_per_gpu=1)
