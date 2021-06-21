modal_used = ['image', 'video', 'text', 'audio']

model = dict(type='SingleBranchModel',
             pretrained='torchvision://resnet50',
             backbone=dict(type='ResNet',
                           depth=50,
                           num_stages=4,
                           out_indices=(3, ),
                           frozen_stages=1,
                           norm_cfg=dict(type='BN', requires_grad=True),
                           norm_eval=True,
                           style='pytorch'),
             head=dict(
                 type='SingleSEHead',
                 in_dim=2048,
                 gating_reduction=8,
                 out_dim=1024,
                 dropout_p=0.5,
                 cls_head_config=dict(
                     type='ClsHead',
                     in_dim=1024,
                     out_dim=82,
                     loss=dict(type='MultiLabelBCEWithLogitsLoss')),
             ))
