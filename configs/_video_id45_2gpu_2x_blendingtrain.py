_base_ = '_video_id45_2gpu_2x.py'

data = dict(
    train=dict(ann_file='dataset/tagging/GroundTruth/datafile/blending_train_dev.txt'),
    val=dict(ann_file='dataset/tagging/GroundTruth/datafile/blending_val_dev.txt')
)

train_total_iters = int(20000 * 0.9)
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[train_total_iters // 3, 2 * train_total_iters // 3])
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)