_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(mode=1,
             modal_used=['image'],
             ebd_config=dict(image=dict(type='FCHead', dropout_p=0.0)))
optimizer = dict(_delete_=True, type='SGD', lr=0.02, weight_decay=0.0001)
data = dict(train=dict(type='TaggingDataset',
                       label_id_file='dataset/tagging/label_super_id.txt'))
