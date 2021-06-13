_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/single_branch.py', '_base_/datasets/base_dataset.py'
]

data = dict(samples_per_gpu=2, workers_per_gpu=4)

model = dict(modal_used=['image'], head_config=dict(image=dict(dropout_p=0.0)))
