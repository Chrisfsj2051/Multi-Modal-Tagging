_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(
    mode=1,
    modal_used=['image'],
    branch_config=dict(image_branch=dict(plugins=[
        dict(cfg=dict(
            type='DropBlock', drop_prob=0.15, block_size=7, postfix='_1'),
             stages=(False, False, True, True),
             position='after_conv1'),
        dict(cfg=dict(
            type='DropBlock', drop_prob=0.15, block_size=7, postfix='_2'),
             stages=(False, False, True, True),
             position='after_conv2'),
        dict(cfg=dict(
            type='DropBlock', drop_prob=0.15, block_size=7, postfix='_3'),
             stages=(False, False, True, True),
             position='after_conv3')
    ])))

optimizer = dict(_delete_=True, type='SGD', lr=0.02, weight_decay=0.0001)
