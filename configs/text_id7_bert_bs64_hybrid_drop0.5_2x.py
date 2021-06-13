_base_ = 'text_id6_bert.py'

norm_cfg = dict(type='SyncBN')
# norm_cfg = dict(type='BN1d')

data = dict(samples_per_gpu=32, workers_per_gpu=8)

model = dict(modal_used=['text'],
             branch_config=dict(text=dict(_delete_=True, type='Bert')),
             head_config=dict(image=dict(dropout_p=0.8, norm_cfg=norm_cfg),
                              video=dict(norm_cfg=norm_cfg),
                              text=dict(dropout_p=0.5,
                                        norm_cfg=norm_cfg,
                                        in_dim=768),
                              audio=dict(norm_cfg=norm_cfg),
                              fusion=dict(norm_cfg=norm_cfg)))

train_total_iters = 20000
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[train_total_iters // 3, 2 * train_total_iters // 3])

runner = dict(type='IterBasedRunner', max_iters=train_total_iters)
