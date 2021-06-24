_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/fusion.py', '_base_/datasets/fusion.py'
]
load_from = 'pretrained/image37_text23_video4_audio3.pth'

custom_hooks = [
    dict(type='FreezeParamHook',
         param_pattern=['video', 'audio', 'image', 'text'],
         eval_pattern=['video', 'audio', 'image', 'text'],
         freeze_iters=1000)
]

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

data = dict(samples_per_gpu=8, workers_per_gpu=8)
