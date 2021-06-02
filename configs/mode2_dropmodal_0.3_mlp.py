_base_ = 'mode2.py'

model = dict(
    head_config=dict(fusion=dict(type='MLPHead')),
    modal_dropout_p=dict(_delete_=True, text=0.3, video=0.3, image=0.3, audio=0.3),
)
