_base_ = 'text_id7_bert_syncbn_4gpu.py'

data = dict(samples_per_gpu=16, workers_per_gpu=8)
