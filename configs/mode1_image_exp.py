_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(mode=1,
             modal_used=['image'],
             ebd_config=dict(image=dict(type='FCHead')),
             head_config=dict(
                 image=dict(type='HMCHead',
                            in_dim=1024,
                            out_dim=82,
                            feat_dim=512,
                            label_id_file='dataset/tagging/label_super_id.txt',
                            loss=dict(type='MultiLabelBCEWithLogitsLoss'))))
# optimizer = dict(_delete_=True, type='SGD', lr=0.02, weight_decay=0.0001)
data = dict(train=dict(type='TaggingDataset',
                       label_id_file='dataset/tagging/label_super_id.txt'))
