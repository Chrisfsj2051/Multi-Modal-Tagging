_base_ = 'text_id3_4gpu_2gpu.py'

model = dict(branch_config=dict(text=dict(type='TextCNN')))
