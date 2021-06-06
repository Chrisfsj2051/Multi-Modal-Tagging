_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'pretrained/text0.717_audio0.675_video0.707_image0.706.pth'

model = dict(
    mode=2,
    head_config=dict(
        fusion=dict(type='HMCHead',
                    feat_dim=512,
                    loss=dict(apply_onehot=False, with_sigmoid=False),
                    label_id_file='dataset/tagging/label_super_id.txt')))
optimizer = dict(_delete_=True, type='Adam', lr=0.001, weight_decay=0.0001)
