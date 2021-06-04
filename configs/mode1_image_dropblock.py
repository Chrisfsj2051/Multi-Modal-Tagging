_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

model = dict(
    mode=1,
    # use_batch_norm=True,
    modal_used=['image'],
    branch_config=dict(
        image=dict(
            plugins=[
                dict(
                    cfg=dict(
                        type='DropBlock',
                        drop_prob=0.1,
                        block_size=5,
                        postfix='_1'),
                    stages=(False, False, True, True),
                    position='after_conv1'),
                dict(
                    cfg=dict(
                        type='DropBlock',
                        drop_prob=0.1,
                        block_size=5,
                        postfix='_2'),
                    stages=(False, False, True, True),
                    position='after_conv2'),
                dict(
                    cfg=dict(
                        type='DropBlock',
                        drop_prob=0.1,
                        block_size=5,
                        postfix='_3'),
                    stages=(False, False, True, True),
                    position='after_conv3')
            ]
        )
    )
)
optimizer = dict(_delete_=True, type='SGD', lr=0.02, weight_decay=0.0001)
# data = dict(samples_per_gpu=32)
# evaluation = dict(interval=250)
