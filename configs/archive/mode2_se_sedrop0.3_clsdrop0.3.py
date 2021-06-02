_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

load_from = 'pretrained/text0.717_audio0.675_video0.707_image0.706.pth'

model = dict(
    mode=2,
    head_config=dict(
        fusion=dict(type='SEHead', in_dim=20480, out_dim=82,
                    gating_reduction=8, hidden_size=1024,
                    input_dropout_p=0.3, dropout_p=0.3,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss'))
    )
)
data = dict(workers_per_gpu=2)
optimizer = dict(_delete_=True, type='SGD', lr=0.05, weight_decay=0.0001)