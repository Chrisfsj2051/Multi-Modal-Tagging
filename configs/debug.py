_base_ = ['_base_/default_runtime.py',
          '_base_/schedules/schedule_1x.py',
          '_base_/models/fusion_4branch.py',
          '_base_/datasets/base_dataset.py']

checkpoint_config = dict(interval=1000)
evaluation = dict(interval=100)
model = dict(
    type='MultiBranchesFusionModel',
    pretrained=dict(_delete_=True),
    use_layer_norm=True,
    modal_dropout_p=dict(_delete_=True, text=0.5, video=0.5, image=0.5, audio=0.5),
)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

data = dict(workers_per_gpu=0,
            test=dict(type='TaggingDataset',
                      ann_file='dataset/tagging/GroundTruth/datafile/train.txt'))
