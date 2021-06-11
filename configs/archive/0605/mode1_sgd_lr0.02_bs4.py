_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(mode=1, )

data = dict(samples_per_gpu=4, workers_per_gpu=2)

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
