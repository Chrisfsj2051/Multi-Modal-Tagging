_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

checkpoint_config = dict(interval=5000)
evaluation = dict(interval=2000)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
