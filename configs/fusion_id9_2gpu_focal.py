_base_ = 'fusion_id9_2gpu.py'

load_from =None

data = dict(samples_per_gpu=2, workers_per_gpu=1)

model = dict(
    fusion_config=dict(
        cls_head_config=dict(
            loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.0, use_alpha=True)
        )
    ),
    branch_config=dict(
        video=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5)))),
        image=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5)))),
        audio=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5)))),
        text=dict(head=dict(cls_head_config=dict(loss=dict(type='MultiLabelBCEWithLogitsFocalLoss', gamma=1.5)))),
    )
)