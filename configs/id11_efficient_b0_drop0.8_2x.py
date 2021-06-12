_base_ = 'id11.py'

model = dict(modal_used=['image'],
             pretrained=dict(_delete_=True),
             branch_config=dict(image=dict(
                 _delete_=True, type='EffecientNet', arch='efficientnet-b0')),
             head_config=dict(image=dict(dropout_p=0.8, in_dim=1280)))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[16000, 18000])
runner = dict(type='IterBasedRunner', max_iters=20000)
