_base_ = 'mode2.py'

model = dict(modal_dropout_p=dict(_delete_=True,
                                  text=0.3,
                                  video=0.3,
                                  image=0.3,
                                  audio=0.3), )
data = dict(samples_per_gpu=32)
