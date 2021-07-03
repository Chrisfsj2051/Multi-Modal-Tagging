_base_ = 'fusion_id9_2gpu_semi_2x.py'

# custom_hooks = [
#     dict(
#         type='SemiEMAHook',
#         burnin_iters=1000,
#         ema_eval=False,
#         momentum=0.1
#     )
# ]

model = dict(type='SemiMultiBranchFusionModel',
             gt_thr=0.5,
             unlabeled_loss_weight=0.7,
             only_fusion=True
             )

train_total_iters = 30000
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[train_total_iters // 3, 2 * train_total_iters // 3])
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)