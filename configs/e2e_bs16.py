_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

# load_from = 'work_dirs/mode1_text_aug_0.1/iter_3000.pth'

train_total_iters = 10000

optimizer = dict(_delete_=True,
                 type='Adam',
                 amsgrad=True,
                 lr=0.01,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(
                     custom_keys={
                         'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'text_branch': dict(lr_mult=0.001, decay_mult=1.0),
                         'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'fusion': dict(weight_decay_mult=1.0)
                     }))

model = dict(
    mode=3,
    branch_config=dict(text=dict(_delete_=True, type='Bert')),
    head_config=dict(text=dict(type='SEHead', in_dim=768),
                     fusion=dict(type='SEHead', in_dim=20224)),
    modal_dropout_p=dict(text=0.3, video=0.3, image=0.3, audio=0.3),
)

optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[train_total_iters // 3, 2 * train_total_iters // 3])

runner = dict(type='IterBasedRunner', max_iters=train_total_iters)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

data = dict(
    workers_per_gpu=0,
    samples_per_gpu=2,
)
