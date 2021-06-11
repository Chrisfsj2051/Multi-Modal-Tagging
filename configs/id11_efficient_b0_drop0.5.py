_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/single_branch.py', '_base_/datasets/base_dataset.py'
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

model = dict(modal_used=['image'],
             pretrained=dict(_delete_=True),
             branch_config=dict(image=dict(
                 _delete_=True, type='EffecientNet', arch='efficientnet-b0')),
             head_config=dict(image=dict(dropout_p=0.5, in_dim=1280)))
