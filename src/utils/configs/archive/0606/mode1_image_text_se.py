_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(mode=1,
             modal_used=['text', 'image'],
             ebd_config=dict(
                 video=dict(
                     type='SEHead',
                     in_dim=16384,
                     gating_reduction=8,
                     out_dim=1024
                 ),
                 image=dict(
                     type='SEHead',
                     in_dim=2048,
                     gating_reduction=8,
                     out_dim=1024
                 ),
                 text=dict(
                     type='SEHead',
                     in_dim=1024,
                     gating_reduction=8,
                     out_dim=1024
                 ),
                 audio=dict(
                     type='SEHead',
                     in_dim=1024,
                     gating_reduction=8,
                     out_dim=1024
                 ),
             ),
             )
optimizer = dict(_delete_=True,
                 type='SGD',
                 momentum=0.9,
                 lr=0.02,
                 weight_decay=0.0001)
