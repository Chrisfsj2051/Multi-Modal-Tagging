_base_ = 'fusion_2gpu_final.py'

model = dict(
    fusion_config=dict(
        cls_head_config=dict(
            dropout_p=0.5
        )
    )
)