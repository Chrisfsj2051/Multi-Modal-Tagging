_base_ = [
    '_base_/default_runtime.py', '_base_/schedules/schedule_1x_adam.py',
    '_base_/models/video_text_match.py', '_base_/datasets/video_text_match.py'
]
# evaluation = dict(interval=100)

# data = dict(samples_per_gpu=2, workers_per_gpu=2)
