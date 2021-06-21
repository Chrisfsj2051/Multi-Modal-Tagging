_base_ = 'video_id3_4gpu.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='TransformerEncoder',
        dim_in=1024,
        num_head=4,
        dim_hidden=2048,
        dim_out=1024,
        seq_len=300,
        num_layers=3),
    head=dict(
        in_dim=1024
    )
)
