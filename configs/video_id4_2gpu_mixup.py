_base_ = 'video_id4_2gpu.py'

data = dict(samples_per_gpu=2, workers_per_gpu=1)

model=dict(
    head = dict(
        type='SingleMixupSEHead',
        cls_head_config=dict(type= 'MixupClsHead')
    )
)