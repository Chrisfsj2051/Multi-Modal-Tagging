_base_ = 'fusion_id9_2gpu.py'

# load_from = None
# data = dict(samples_per_gpu=2, workers_per_gpu=1)

model = dict(
    type='MultiBranchFusionModel',
    modal_dropout_p=dict(text=0.0, video=0.0, image=1.0, audio=0.0)
)