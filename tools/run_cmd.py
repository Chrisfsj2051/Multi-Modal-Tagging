import os
import random

configs = os.listdir('configs/')

for cfg in configs:
    if 'mode2' not in cfg or 'hmc' in cfg:
        continue
    PORT = random.randint(10000, 30000)
    print(f'PORT={PORT} nohup bash tools/dist_train.sh {"configs/"+cfg} 2 &')
