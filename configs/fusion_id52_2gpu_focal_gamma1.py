_base_ = '_fusion_id52_2gpu.py'

model = dict(
    fusion_config=dict( cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1))),
    branch_config=dict(
        video=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1)))),
        image=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1)))),
        text=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1)))),
        audio=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1))))
    )
)
