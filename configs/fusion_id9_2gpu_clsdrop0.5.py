_base_ = 'fusion_id9_2gpu.py'

model = dict(
    fusion_config=dict(
        cls_head_config=dict(
            dropout_p=0.5
        )
    )
)