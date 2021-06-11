import os
import random

configs = os.listdir('configs/')

for cfg in configs:
    if 'eff' not in cfg:
        continue
    PORT = random.randint(10000, 30000)
    # print(f'PORT={PORT} nohup bash tools/dist_train.sh {"configs/"+cfg} 2 &\nsleep 1\n') # noqa
    print(f'PORT={PORT} bash tools/dist_train.sh {"configs/"+cfg} 2')
