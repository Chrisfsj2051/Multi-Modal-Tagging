import os
import random

configs = os.listdir('configs/')
cnt = 0
for i, cfg in enumerate(configs):
    if 'bert' not in cfg:
        continue
    PORT = random.randint(10000, 30000)
    if cnt % 2 == 0:
        cmd = f'PORT={PORT} nohup bash tools/dist_train.sh {"configs/"+cfg} 2 &'  # noqa
    else:
        cmd = f'PORT={PORT} bash tools/dist_train.sh {"configs/" + cfg} 2'
    cnt += 1
    print(cmd)
    print('sleep 3')
