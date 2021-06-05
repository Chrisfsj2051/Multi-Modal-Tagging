_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(mode=1,
             modal_used=['video', 'text'],
             head_config=dict(fusion=dict(in_dim=17408)))

optimizer = dict(_delete_=True, type='Adam', lr=0.001, weight_decay=0.0001)

data = dict(samples_per_gpu=4)
