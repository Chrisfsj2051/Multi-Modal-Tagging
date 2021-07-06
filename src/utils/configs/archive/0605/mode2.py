_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'temp_files/iter_10000.pth'
model = dict(mode=2, pretrained=dict(_delete_=True))
data = dict(workers_per_gpu=0)
