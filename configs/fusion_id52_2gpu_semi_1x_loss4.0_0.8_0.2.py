_base_ = 'fusion_id52_2gpu_semi_1x_loss4.0.py'

model = dict(
    type='SemiMultiBranchFusionModel',
    gt_thr=0.8,
    ignore_thr=0.2,
    # unlabeled_loss_weight=1.0
)