_base_ = 'mode2.py'

model = dict(
    modal_dropout_p=dict(_delete_=True, text=0.1, video=0.1, image=0.1, audio=0.1),
)
