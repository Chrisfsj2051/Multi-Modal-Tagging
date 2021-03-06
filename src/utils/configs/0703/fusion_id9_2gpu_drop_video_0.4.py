_base_ = '_fusion_id51_2gpu.py'

# load_from = None
# data = dict(samples_per_gpu=2, workers_per_gpu=1)

model = dict(
    type='MultiBranchFusionModel',
    modal_dropout_p=dict(text=0.0, video=0.4, image=0.0, audio=0.0)
)