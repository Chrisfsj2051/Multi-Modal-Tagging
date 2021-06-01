_base_ = 'baseline.py'

model  = dict(
    ebd_config=dict(
        video=dict(dropout_p=0.3),
        image=dict(dropout_p=0.8),
        text=dict(dropout_p=0.8),
        audio=dict(dropout_p=0.3)
    )
)

