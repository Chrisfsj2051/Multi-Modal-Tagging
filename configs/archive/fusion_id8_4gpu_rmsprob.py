_base_ = 'fusion_id8_4gpu.py'

# load_from = 'pretrained/image37_text23_video3_audio3.pth'

optimizer = dict(
    _delete_=True,
    type='RMSProb',
    amsgrad=True,
    lr=0.01,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'image_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'text_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'video_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'audio_branch.backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'fusion_head': dict(lr_mult=0.00, decay_mult=1.0)
        }))

