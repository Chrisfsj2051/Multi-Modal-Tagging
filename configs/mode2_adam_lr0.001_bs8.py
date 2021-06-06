_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'pretrained/text0.717_audio0.675_video0.707_image0.706.pth'

model = dict(mode=2)
optimizer = dict(_delete_=True, type='Adam', lr=0.001, weight_decay=0.0001)
data = dict(samples_per_gpu=8)
