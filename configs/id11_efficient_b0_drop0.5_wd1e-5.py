_base_ = 'id11.py'

model = dict(modal_used=['image'],
             pretrained=dict(_delete_=True),
             branch_config=dict(image=dict(
                 _delete_=True, type='EffecientNet', arch='efficientnet-b0')),
             head_config=dict(image=dict(dropout_p=0.5, in_dim=1280)))

optimizer = dict(lr=0.1,
                 weight_decay=1e-5,
                 paramwise_cfg=dict(
                     custom_keys={
                         'image_branch': dict(lr_mult=0.01),
                         'text_branch': dict(lr_mult=0.001, decay_mult=1.0),
                         'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'fusion': dict(weight_decay_mult=1.0)
                     }))
