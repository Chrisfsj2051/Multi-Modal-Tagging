_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x.py',
    '_base_/models/fusion_4branch.py', '_base_/datasets/base_dataset.py'
]

checkpoint_config = dict(interval=1000)
evaluation = dict(interval=100)
model = dict(
    type='MultiBranchesFusionModel',
    head_config=dict(
        fusion=dict(type='HMCHead',
                    feat_dim=512,
                    loss=dict(apply_onehot=False, with_sigmoid=False),
                    label_id_file='dataset/tagging/label_super_id.txt')),
    modal_dropout_p=dict(_delete_=True,
                         text=0.5,
                         video=0.5,
                         image=0.5,
                         audio=0.5),
)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=10, norm_type=2))

data = dict(workers_per_gpu=0,
            test=dict(type='TaggingDataset',
                      label_id_file='dataset/tagging/label_super_id.txt',
                      ann_file='dataset/tagging/GroundTruth/datafile/val.txt'))
