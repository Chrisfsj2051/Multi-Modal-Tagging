_base_ = 'text_id7_bert_syncbn_4gpu.py'

data = dict(samples_per_gpu=8, workers_per_gpu=8)

model = dict(branch_config=dict(
    text=dict(_delete_=True, type='Bert', fixed=True)))
