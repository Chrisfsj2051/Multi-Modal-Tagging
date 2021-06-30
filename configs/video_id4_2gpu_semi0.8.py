_base_ = 'video_id4_2gpu_semi0.7.py'
model = dict(
    type='SemiSingleBranchModel',
    gt_thr=0.8
)
