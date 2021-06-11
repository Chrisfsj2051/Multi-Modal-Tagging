_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/single_branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(modal_used=['image'])
data = dict(samples_per_gpu=8, workers_per_gpu=8)
