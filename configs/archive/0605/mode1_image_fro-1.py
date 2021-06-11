_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(
    mode=1,
    modal_used=['image'],
    branch_config=dict(image=dict(type='ResNet',
                                  depth=50,
                                  num_stages=4,
                                  out_indices=(3, ),
                                  frozen_stages=-1,
                                  norm_cfg=dict(type='BN', requires_grad=True),
                                  norm_eval=False)))

optimizer = dict(_delete_=True, type='SGD', lr=0.02, weight_decay=0.0001)
