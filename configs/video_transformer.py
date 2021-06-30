_base_ = 'video_id4_2gpu.py'

data = dict(samples_per_gpu=2, workers_per_gpu=1)

model = dict(
    backbone=dict(
        _delete_=True,
        type='TransformerEncoder',
        dim_in=1024,
        num_head=4,
        dim_hidden=2048,
        dim_out=2048,
        seq_len=300,
        num_layers=3),
    head=dict(
        in_dim=2048
    )
)
