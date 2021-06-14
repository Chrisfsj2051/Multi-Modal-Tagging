_base_ = 'fusion_id1_videoaug_imageaug.py'

load_from = 'pretrained/text11_audio1_video1_image34.pth'

optimizer = dict(_delete_=True,
                 type='Adam',
                 amsgrad=True,
                 lr=0.01,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(
                     custom_keys={
                         'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'text_branch': dict(lr_mult=0.001, decay_mult=1.0),
                         'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'fusion': dict(weight_decay_mult=1.0)
                     }))

model = dict(branch_config=dict(text=dict(_delete_=True, type='Bert')),
             head_config=dict(text=dict(in_dim=768),
                              fusion=dict(in_dim=20224)))
