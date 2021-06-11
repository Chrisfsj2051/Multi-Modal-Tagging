_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_sgd.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

model = dict(mode=1,
             use_layer_norm=True,
             modal_used=['text', 'image'],
             head_config=dict(fusion=dict(in_dim=3072)))
optimizer = dict(_delete_=True, type='SGD', lr=0.02, weight_decay=0.0001)
