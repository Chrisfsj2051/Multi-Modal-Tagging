_base_ = '_fusion_id51_2gpu.py'

model = dict(
    modal_dropout_p=dict(text=0.1, video=0.1, image=0.1, audio=0.1)
)