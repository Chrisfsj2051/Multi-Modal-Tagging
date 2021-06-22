# model = dict(
#     type='ModalMatchModel',
#     modal_keys=['video', 'text'],
#     backbone_config=dict(
#         video=dict(
#             type='NeXtVLAD',
#             feature_size=1024,
#             max_frames=300,
#             cluster_size=128
#         ),
#         text=dict(
#             type='TwoStreamTextCNN',
#             vocab_size=21129,
#             ebd_dim=300,
#             channel_in=256,
#             channel_out=1024,
#             filter_size=(2, 3, 4)
#         )
#     )
#
#     head=dict(
#         type='SingleSEHead',
#         in_dim=16384,
#         gating_reduction=8,
#         out_dim=1024,
#         dropout_p=0.8,
#         cls_head_config=dict(
#             type='ClsHead',
#             in_dim=1024,
#             out_dim=82,
#             loss=dict(type='MultiLabelBCEWithLogitsLoss'))
#     ))
