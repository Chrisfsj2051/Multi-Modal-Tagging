_base_ = '_fusion_id51_2gpu.py'

load_from = None

model = dict(
    modal_dropout_p=dict(text=0.1, video=0.1, image=0.1, audio=0.1),
    branch_config=dict(
        video=dict(head=dict(cls_head_config=dict(loss=dict(loss_weight=4)))),
        image=dict(head=dict(cls_head_config=dict(loss=dict(loss_weight=4)))),
        text=dict(head=dict(cls_head_config=dict(loss=dict(loss_weight=4)))),
        audio=dict(head=dict(cls_head_config=dict(loss=dict(loss_weight=4)))),
    )
)
