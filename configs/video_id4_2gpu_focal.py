_base_ = 'video_id4_2gpu.py'

model = dict(
    head=dict(
        cls_head_config=dict(
            loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5, loss_weight=8))
    ))
