_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

load_from = 'work_dirs/mode1/iter_10000.pth'
model = dict(
    mode=2,
    pretrained=dict(
        _delete_=True),
    head_config=dict(
        fusion=dict(type='SEHead', in_dim=20480, out_dim=82,
                    gating_reduction=8, hidden_size=1024,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss'))
    )
)
data = dict(workers_per_gpu=0)
