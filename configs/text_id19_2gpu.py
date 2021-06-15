_base_ = 'text_id3_2gpu.py'

model = dict(branch_config=dict(text=dict(type='TextCNN')))
