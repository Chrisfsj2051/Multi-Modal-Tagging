_base_ = 'video_id4_2gpu_semi0.4.py'

model = dict(
    type='SemiSingleBranchModel',
    gt_thr=0.4,
    unlabeled_loss_weight=0.5
)


train_total_iters = 20000

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[train_total_iters // 3 , 2 * train_total_iters // 3]
)
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)