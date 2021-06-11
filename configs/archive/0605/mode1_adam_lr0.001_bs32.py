_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(
    mode=1,
    pretrained=dict(_delete_=True),
)

data = dict(samples_per_gpu=16, workers_per_gpu=8)

optimizer = dict(_delete_=True, type='Adam', lr=0.001, weight_decay=0.0001)
