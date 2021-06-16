_base_ = 'fusion_id5_4gpu.py'

load_from = 'pretrained/image35_text21_video2_audio2.pth'
# load_from = None

model=dict(
    modal_dropout_p=dict(text=0.3, video=0.3, image=0.3, audio=0.3)
)
