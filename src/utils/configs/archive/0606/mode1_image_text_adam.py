_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(mode=1, modal_used=['text', 'image'])

optimizer = dict(_delete_=True, type='Adam', lr=0.001, weight_decay=0.0001)

data = dict(samples_per_gpu=4)

# optimizer = dict(_delete_=True,
#                  type='SGD',
#                  momentum=0.9,
#                  lr=0.02,
#                  weight_decay=0.0001)
