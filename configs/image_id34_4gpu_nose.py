_base_ = 'image_id34_4gpu.py'

model = dict(head_config=dict(
    image=dict(_delete_=True,
               type='ClsHead',
               in_dim=2048,
               out_dim=82,
               loss=dict(type='MultiLabelBCEWithLogitsLoss'))))
