_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

model = dict(
    mode=1,
    use_batch_norm=True,
    modal_used=['image'],
    ebd_config=dict(
        image=dict(type='FCHead', dropout_p=0.5)
    )
)
optimizer = dict(_delete_=True, type='SGD', lr=0.08, weight_decay=0.0001)
data = dict(samples_per_gpu=16)
evaluation = dict(interval=250)
