_base_ = 'text_baseline_adam.py'

model = dict(modal_used=['text'],
             branch_config=dict(text=dict(_delete_=True, type='Bert')),
             head_config=dict(text=dict(
                 type='SEHead',
                 in_dim=768,
             )))

optimizer = dict(lr=0.1,
                 paramwise_cfg=dict(
                     custom_keys={
                         'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'text_branch': dict(lr_mult=0.0001, decay_mult=1.0),
                         'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'fusion': dict(weight_decay_mult=1.0)
                     }))
