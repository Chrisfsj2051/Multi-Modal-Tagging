_base_ = 'text_id7_bert_syncbn_4gpu.py'

data = dict(samples_per_gpu=8, workers_per_gpu=8)

optimizer = dict(lr=0.1,
                 paramwise_cfg=dict(
                     custom_keys={
                         'image_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'text_branch': dict(lr_mult=0.0005, decay_mult=1.0),
                         'video_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'audio_branch': dict(lr_mult=0.01, decay_mult=1.0),
                         'fusion': dict(weight_decay_mult=1.0)
                     }))
