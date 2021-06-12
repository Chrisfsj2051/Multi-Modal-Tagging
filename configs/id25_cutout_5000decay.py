_base_ = 'id25_cutout.py'

resume_from = 'work_dirs/id25_cutout/iter_5000.pth'

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[5005, 6000])