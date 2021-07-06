import os
import random

TRAIN_RATIO = 0.9

if __name__ == '__main__':
    data_base = 'dataset/tagging/GroundTruth/datafile/'
    with open(data_base + 'train.txt', 'r', encoding='utf-8') as f:
        contents = f.readlines() + ['\n']
    groups = [contents[x:x + 6] for x in range(0, len(contents), 6)]
    random.seed(1)
    random.shuffle(groups)
    train_dev_path = data_base + 'blending_train_dev.txt'
    val_dev_path = data_base + 'blending_val_dev.txt'
    txt_list = [train_dev_path, val_dev_path]
    TRAIN_NUM = int(len(groups) * 0.9)
    data_list = [groups[:TRAIN_NUM], groups[TRAIN_NUM:]]
    for txt, data in zip(txt_list, data_list):
        with open(txt, 'w', encoding='utf-8') as f:
            for d in data:
                f.writelines(d)
