_base_ = 'fusion_id9_2gpu_transformer_semi.py'
# load_from = None
train_total_iters = 30000
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[train_total_iters // 3, 2 * train_total_iters // 3])
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)