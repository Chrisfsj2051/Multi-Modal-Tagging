train_total_iters = 10000
# optimizer
optimizer = dict(
    type='Adam',
    amsgrad=True,
    lr=0.01,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                     'text_branch': dict(lr_mult=0.001, decay_mult=1.0),
                     'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                     'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                     'fusion': dict(weight_decay_mult=1.0)})
)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[train_total_iters // 3, 2 * train_total_iters // 3]
)

runner = dict(type='IterBasedRunner', max_iters=train_total_iters)
