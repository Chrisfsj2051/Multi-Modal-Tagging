_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'pretrained/text0.717_audio0.675_video0.707_image0.706.pth'

model = dict(
    mode=2,
    branch_config=dict(text=dict(
        type='TwoStreamTextCNN',
        vocab_size=9906,
        ebd_dim=300,
        channel_in=128,
    )),
    head_config=dict(fusion=dict(
        dict(type='HMCHead',
             label_id_file='dataset/tagging/label_super_id_0312.txt'))))
optimizer = dict(_delete_=True,
                 type='SGD',
                 lr=0.05,
                 momentum=0.9,
                 weight_decay=0.0001)

data = dict(
    train=dict(label_id_file='dataset/tagging/label_super_id_0312.txt'),
    val=dict(label_id_file='dataset/tagging/label_super_id_0312.txt'))
