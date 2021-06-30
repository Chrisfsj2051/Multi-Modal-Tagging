_base_ = 'video_id4_2gpu_semi0.7.py'

custom_hooks = [
    dict(
        type='SemiEMAHook',
        burnin_iters=500,
        ema_eval=False
    )
]