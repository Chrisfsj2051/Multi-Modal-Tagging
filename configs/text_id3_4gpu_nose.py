_base_ = 'text_id3_4gpu.py'

model = dict(head_config=dict(
    text=dict(_delete_=True,
              type='ClsHead',
              in_dim=1024,
              out_dim=82,
              loss=dict(type='MultiLabelBCEWithLogitsLoss'))))
