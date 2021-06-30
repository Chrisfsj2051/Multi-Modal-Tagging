_base_ = 'fusion_id9_2gpu.py'

# load_from = None
# data = dict(samples_per_gpu=2, workers_per_gpu=1)

model = dict(
    type='MultiBranchFusionModel',
    fusion_config=dict(dropout_p=0.9)
)