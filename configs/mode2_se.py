_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

model = dict(
    mode=2,
    pretrained=dict(
        image='pretrained/image_0.7061.pth',
        text='pretrained/text_0.7176.pth',
        audio='pretrained/audio_0.6759.pth',
        video='pretrained/video_0.7072.pth',
    ),
    head_config=dict(
        fusion=dict(type='SEHead', in_dim=20480, out_dim=82,
                    gating_reduction=8, hidden_size=1024,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss'))
    )
)
data = dict(workers_per_gpu=2)
