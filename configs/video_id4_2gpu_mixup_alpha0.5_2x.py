_base_ = 'video_id4_2gpu.py'

# data = dict(samples_per_gpu=8, workers_per_gpu=1)
model = dict(
    head=dict(
        type='SingleMixupSEHead',
        alpha=0.5,
        cls_head_config=dict(
            type='MixupClsHead',
            loss=dict(type='MixupMultiLabelBCEWithLogitsLoss')
        )
    )
)

train_total_iters = 20000

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[train_total_iters // 3, 2 * train_total_iters // 3]
)
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)
