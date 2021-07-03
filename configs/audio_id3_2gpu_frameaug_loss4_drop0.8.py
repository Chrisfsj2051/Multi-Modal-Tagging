_base_ = 'audio_id3_2gpu_frameaug_loss4.py'

model = dict(
    head=dict(
        dropout_p=0.8,
    ))

