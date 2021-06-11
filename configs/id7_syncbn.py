_base_ = 'id7.py'

model = dict(modal_used=['image'],
             branch_config=dict(video=dict(norm_cfg=dict(type='SyncBN')),
                                audio=dict(norm_cfg=dict(type='SyncBN'))),
             head_config=dict(image=dict(norm_cfg=dict(type='SyncBN')),
                              video=dict(norm_cfg=dict(type='SyncBN')),
                              text=dict(norm_cfg=dict(type='SyncBN')),
                              audio=dict(norm_cfg=dict(type='SyncBN')),
                              fusion=dict(norm_cfg=dict(type='SyncBN'))))
