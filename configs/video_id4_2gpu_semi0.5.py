_base_ = 'video_id4_2gpu_semi0.4.py'
model = dict(
    type='SemiSingleBranchModel',
    gt_thr=0.5
)
