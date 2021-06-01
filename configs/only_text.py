_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

model = dict(
    type='MultiBranchesFusionModel',
    pretrained=dict(
        _delete_=True
    ))
checkpoint_config = dict(interval=2000)
evaluation = dict(interval=2000)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
