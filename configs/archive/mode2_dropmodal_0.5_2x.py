_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

load_from = 'pretrained/text0.717_audio0.675_video0.707_image0.706.pth'

model = dict(
    mode=2,
    modal_dropout_p=dict(text=0.5, video=0.5, image=0.5, audio=0.5),
)
optimizer = dict(_delete_=True,
                 type='SGD',
                 lr=0.05,
                 momentum=0.9,
                 weight_decay=0.0001)
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001,
                 step=[16000, 18000])
runner = dict(type='IterBasedRunner', max_iters=20000)
