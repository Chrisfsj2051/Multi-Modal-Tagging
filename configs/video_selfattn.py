_base_ = 'video_id1_4gpu.py'

model = dict(head_config=dict(
    video=dict(_delete_=True,
               type='SelfAttnFusionHead',
               dim_in=1024,
               num_head=4,
               dim_hidden=2048,
               num_layers=3,
               cls_head_config=dict(type='ClsHead',
                                    in_dim=1024,
                                    out_dim=82,
                                    loss=dict(
                                        type='MultiLabelBCEWithLogitsLoss')))))

optimizer = dict(lr=0.01,
                 paramwise_cfg=dict(
                     custom_keys={
                         'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'text_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'fusion': dict(weight_decay_mult=1.0)
                     }))
