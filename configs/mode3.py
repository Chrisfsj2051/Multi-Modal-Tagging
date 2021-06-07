_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'work_dirs/mode2/iter_10000.pth'

model = dict(mode=3)
optimizer = dict(_delete_=True, type='Adam', lr=0.0001, weight_decay=0.0001)
