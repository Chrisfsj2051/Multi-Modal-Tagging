_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

model = dict(
    mode=1,
)

data=dict(samples_per_gpu=16,
          workers_per_gpu=8)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)