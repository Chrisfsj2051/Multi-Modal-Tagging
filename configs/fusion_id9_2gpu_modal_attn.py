_base_ = 'fusion_id9_2gpu.py'


# checkpoint_config = dict(interval=10000)
# evaluation = dict(interval=100)
#
#
# load_from = None
# data = dict(samples_per_gpu=2, workers_per_gpu=1)

model = dict(
    fusion_config=dict(
        type='FusionSEHeadWithModalAttn',
        modal_in_dim=dict(
            video = 16384,
            audio=1024,
            text=1024,
            image=2048
        )
    )
)


