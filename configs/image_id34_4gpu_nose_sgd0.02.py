_base_ = 'image_id34_4gpu.py'

optimizer = dict(_delete_=True,
                 type='SGD',
                 lr=0.02,
                 momentum=0.9,
                 weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# learning policy
lr_config = dict(_delete_=True,
                 policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[8000, 9000])

model = dict(head_config=dict(
    fusion=dict(_delete_=True,
                type='ClsHead',
                in_dim=2048,
                out_dim=82,
                loss=dict(type='MultiLabelBCEWithLogitsLoss'))))
