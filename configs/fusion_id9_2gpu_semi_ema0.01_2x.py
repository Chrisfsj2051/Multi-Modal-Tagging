_base_ = 'fusion_id9_2gpu_semi_2x.py'

custom_hooks = [
    dict(
        type='SemiEMAHook',
        burnin_iters=1000,
        ema_eval=False,
        momentum=0.01
    )
]