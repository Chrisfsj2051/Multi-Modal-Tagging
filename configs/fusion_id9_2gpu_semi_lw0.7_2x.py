_base_ = 'fusion_id9_2gpu_semi_2x.py'

model = dict(type='SemiMultiBranchFusionModel',
             gt_thr=0.7,
             unlabeled_loss_weight=1.0)