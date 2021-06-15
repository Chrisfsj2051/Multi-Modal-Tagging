_base_ = 'fusion_id5_4gpu.py'

load_from = 'pretrained/image35_text22_video2_audio2.pth'

model = dict(branch_config=dict(text=dict(type='TextCNN')))
