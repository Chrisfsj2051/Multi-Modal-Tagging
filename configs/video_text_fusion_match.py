_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/datasets/fusion.py'
]

load_from = 'pretrained/image37_text23_video4_audio3.pth'

model = dict(
    type='VideoTextWithExtraTaskModel',
    branch_config=dict(
        video=dict(
            type='SingleBranchModel',
            key='video',
            backbone=dict(
                type='NeXtVLAD',
                feature_size=1024,
                max_frames=300,
                cluster_size=128
            ),
            head=dict(
                type='SingleSEHead',
                in_dim=16384,
                gating_reduction=8,
                out_dim=1024,
                dropout_p=0.8,
                cls_head_config=dict(
                    type='ClsHead',
                    in_dim=1024,
                    out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8)
                )
            )
        ),
        text=dict(
            type='SingleBranchModel',
            key='text',
            backbone=dict(
                type='TwoStreamTextCNN',
                vocab_size=21129,
                ebd_dim=300,
                channel_in=256,
                channel_out=1024,
                filter_size=(2, 3, 4)
            ),
            head=dict(
                type='SingleSEHead',
                in_dim=1024,
                gating_reduction=8,
                out_dim=1024,
                dropout_p=0.5,
                cls_head_config=dict(
                    type='ClsHead',
                    in_dim=1024,
                    out_dim=82,
                    loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8))
            )
        )
    ),
    modal_match_config=dict(
        type='ModalMatchHead',
        fc_dim1=16384,
        fc_dim2=1024,
        hidden_dim=2048,
        loss=dict(type='BCEWithLogitsLoss', loss_weight=8)
    ),
    fusion_config=dict(
        type='FusionSEHead',
        in_dim=17408,
        gating_reduction=8,
        dropout_p=0.8,
        out_dim=1024,
        cls_head_config=dict(
            type='ClsHead',
            in_dim=1024,
            out_dim=82,
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8)
        )
    )
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['video', 'text', 'meta_info', 'gt_labels'])
]

val_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/train_5k'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'meta_info'])
]

test_pipeline = [
    dict(type='LoadAnnotations',
         replace_dict=dict(video=(
             'tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging',
             'extracted_video_feats/L16_LN/test_5k_2nd'))),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['video', 'image', 'text', 'audio', 'meta_info'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(type='TaggingDataset',
               ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
               label_id_file='dataset/tagging/label_super_id.txt',
               pipeline=train_pipeline),
    val=dict(type='TaggingDataset',
             ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
             label_id_file='dataset/tagging/label_super_id.txt',
             pipeline=val_pipeline),
    test=dict(type='TaggingDataset',
              ann_file='dataset/tagging/GroundTruth/datafile/test_2nd.txt',
              label_id_file='dataset/tagging/label_super_id.txt',
              pipeline=test_pipeline))
