_base_ = '_fusion_id51_2gpu.py'

load_from = None

train_total_iters = 20000
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[train_total_iters // 3, 2 * train_total_iters // 3])
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)