_base_ = 'video_id4_2gpu_mixup_2x.py'

model = dict(
    head=dict(
        type='SingleMixupSEHead',
        alpha_ub=0.2,
        cls_head_config=dict(
            type='MixupClsHead',
            loss=dict(type='MixupMultiLabelBCEWithLogitsLoss')
        )
    )
)
