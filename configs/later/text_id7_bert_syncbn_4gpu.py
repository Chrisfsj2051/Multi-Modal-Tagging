_base_ = 'text_id6_bert.py'

norm_cfg = dict(type='SyncBN')
# norm_cfg = dict(type='BN1d')

data = dict(samples_per_gpu=4, workers_per_gpu=8)

model = dict(modal_used=['text'],
             branch_config=dict(text=dict(_delete_=True, type='Bert')),
             head_config=dict(image=dict(dropout_p=0.8, norm_cfg=norm_cfg),
                              video=dict(norm_cfg=norm_cfg),
                              text=dict(norm_cfg=norm_cfg, in_dim=768),
                              audio=dict(norm_cfg=norm_cfg),
                              fusion=dict(norm_cfg=norm_cfg)))
