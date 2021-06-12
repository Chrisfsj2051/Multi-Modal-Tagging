_base_ = 'id11.py'

model = dict(modal_used=['image'],
             pretrained=dict(_delete_=True),
             branch_config=dict(image=dict(
                 _delete_=True, type='EffecientNet', arch='efficientnet-b0')),
             head_config=dict(image=dict(dropout_p=0.0, in_dim=1280)))
