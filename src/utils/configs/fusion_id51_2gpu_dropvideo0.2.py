_base_ = '_fusion_id51_2gpu.py'

model = dict(
    modal_dropout_p=dict(text=0.0, video=0.2, image=0.0, audio=0.0)
)