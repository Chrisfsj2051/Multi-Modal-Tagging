import json
import os
import random
from multiprocessing import Pool

import mmcv
import nlpcda

# nlpcda.Homophone(create_num=3, change_rate=0.3),

all_aug_list = [
    nlpcda.Randomword(create_num=3, change_rate=0.1),
    nlpcda.Similarword(create_num=3, change_rate=0.08),
    nlpcda.RandomDeleteChar(create_num=3, change_rate=0.05),
    nlpcda.CharPositionExchange(create_num=3, change_rate=0.05, char_gram=3),
    nlpcda.EquivalentChar(create_num=3, change_rate=0.08)
]

single_aug_list = [
    nlpcda.Randomword(create_num=3, change_rate=0.2),
    nlpcda.Similarword(create_num=3, change_rate=0.2),
    nlpcda.RandomDeleteChar(create_num=3, change_rate=0.2),
    nlpcda.CharPositionExchange(create_num=3, change_rate=0.2, char_gram=3),
    nlpcda.EquivalentChar(create_num=3, change_rate=0.2)
]

save_path = 'dataset/text_aug/'
data_root = 'dataset/tagging/tagging_dataset_train_5k/text_txt/tagging/'
mmcv.mkdir_or_exist(save_path)


def go(data):
    id_name = data.split('.')[0]
    cur_save_path = save_path + id_name + '/'
    if os.path.isdir(data_root + data):
        print(f'Skip {data_root + data}')
        return

    with open(data_root + data, 'r', encoding='utf-8') as f:
        results = json.load(f)
    for key in ['video_ocr', 'video_asr']:
        # if os.path.exists(cur_save_path + key):
        #     continue
        mmcv.mkdir_or_exist(cur_save_path + key)
        text = results[key].replace('|', '，')
        aug_results = [text]
        for aug in single_aug_list:
            new_text = aug.replace(text)
            for tex in new_text:
                aug_results.append(tex)
        random.shuffle(all_aug_list)

        for cnt in range(6):
            text = aug_results[0]
            for aug in all_aug_list:
                text = random.choice(aug.replace(text))
            aug_results.append(text)

        for cnt in range(len(aug_results)):
            with open(cur_save_path + key + f'/{cnt}.txt',
                      'w',
                      encoding='utf-8') as f:
                # print(cur_save_path + key + f'/{cnt}.txt')
                f.write(aug_results[cnt])


pool = Pool(16)
pool.map(go, os.listdir(data_root))
