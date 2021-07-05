_base_ = 'fusion_id51_2gpu_semi_2x.py'

model = dict(
    type='SemiMultiBranchFusionModel',
    gt_thr=0.9,
    ignore_thr=0.1,
    unlabeled_loss_weight=0.5
)

