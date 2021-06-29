_base_ = 'video_id4_2gpu.py'

model = dict(
    head=dict(
        dropout_p=0.9
    )
)

train_total_iters = 20000
lr_config = dict(
    warmup_iters=500,
    step=[train_total_iters // 3, 2 * train_total_iters // 3]
)

runner = dict(type='IterBasedRunner', max_iters=train_total_iters)
