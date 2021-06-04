_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

model = dict(
    mode=1,
    modal_used=['image'],
)
optimizer = dict(_delete_=True, type='SGD', lr=0.02, weight_decay=0.0001)
