_base_ = 'fusion_id51_2gpu_semi_2x.py'

model = dict(
    type='SemiMultiBranchFusionModel',
    gt_thr=0.5,
    ignore_thr=0.3,
    unlabeled_loss_weight=0.5
)

