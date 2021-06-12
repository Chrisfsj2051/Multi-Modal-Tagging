_base_ = 'id11.py'

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[16000, 18000])
runner = dict(type='IterBasedRunner', max_iters=20000)
