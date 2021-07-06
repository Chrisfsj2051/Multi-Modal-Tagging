_base_ = '_fusion_id52_2gpu.py'

model = dict(
    fusion_config=dict( cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5, loss_weight=22))),
    branch_config=dict(
        video=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5, loss_weight=22)))),
        image=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5, loss_weight=22)))),
        text=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5, loss_weight=22)))),
        audio=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5, loss_weight=22))))
    )
)
