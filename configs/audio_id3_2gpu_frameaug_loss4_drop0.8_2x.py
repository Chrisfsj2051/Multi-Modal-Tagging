_base_ = 'audio_id3_2gpu_frameaug_loss4.py'

model = dict(
    head=dict(
        dropout_p=0.8,
    ))

train_total_iters = 20000
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[train_total_iters // 3, 2 * train_total_iters // 3])
runner = dict(type='IterBasedRunner', max_iters=train_total_iters)