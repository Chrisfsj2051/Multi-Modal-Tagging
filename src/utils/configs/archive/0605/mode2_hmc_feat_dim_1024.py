_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'pretrained/text0.717_audio0.675_video0.707_image0.706.pth'

model = dict(mode=2, head_config=dict(fusion=dict(feat_dim=1024)))
optimizer = dict(_delete_=True,
                 type='SGD',
                 lr=0.05,
                 momentum=0.9,
                 weight_decay=0.0001)
