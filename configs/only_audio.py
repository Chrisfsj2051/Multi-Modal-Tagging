_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

model = dict(
    type='MultiBranchesFusionModel',
    modal_used=['audio'],
    pretrained=dict(
        _delete_=True
    ),
    head_config=dict(
        fusion=dict(type='ClsHead', in_dim=1024 * 1, out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss'))
    )
)
checkpoint_config = dict(interval=2000)
evaluation = dict(interval=2000)
data = dict(samples_per_gpu=2, workers_per_gpu=4)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
