_base_ = 'mode2.py'

model = dict(
    modal_dropout_p=dict(_delete_=True, text=0.5, video=0.5, image=0.5, audio=0.5),
)
