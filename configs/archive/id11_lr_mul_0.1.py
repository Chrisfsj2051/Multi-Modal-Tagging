_base_ = 'id11.py'

optimizer = dict(paramwise_cfg=dict(
    custom_keys={
        'image_branch': dict(lr_mult=0.1, decay_mult=1.0),
        'text_branch': dict(lr_mult=0.001, decay_mult=1.0),
        'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
        'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
        'fusion': dict(weight_decay_mult=1.0)
    }))
