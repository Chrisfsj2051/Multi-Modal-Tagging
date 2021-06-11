_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/single_branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(modal_used=['image'],
             branch_config=dict(
                 video=dict(norm_cfg=dict(type='SyncBN')),
                 audio=dict(norm_cfg=dict(type='SyncBN'))
             ),
             head_config=dict(
                 image=dict(norm_cfg=dict(type='SyncBN')),
                 video=dict(norm_cfg=dict(type='SyncBN')),
                 text=dict(norm_cfg=dict(type='SyncBN')),
                 audio=dict(norm_cfg=dict(type='SyncBN')),
                 fusion=dict(norm_cfg=dict(type='SyncBN'))
             ))
data = dict(samples_per_gpu=2, workers_per_gpu=8)
