_base_ = 'video_id4_2gpu_semi0.4.py'


train_total_iters = 20000

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[train_total_iters // 3 , 2 * train_total_iters // 3]
)