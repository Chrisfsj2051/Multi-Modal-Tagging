import json
import os

from tqdm import tqdm

from mmt.utils.tokenization import FullTokenizer

with open('dataset/vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.readlines()

vocab = [x.strip() for x in vocab]
flag = [0] * len(vocab)
vocab_to_index = dict(zip(vocab, list(range(len(vocab)))))
tokenizer = FullTokenizer('dataset/vocab.txt')

for im_root in ('dataset/tagging/tagging_dataset_train_5k/text_txt/tagging/',
                'dataset/tagging/tagging_dataset_test_5k/text_txt/tagging/'):
    for file in tqdm(os.listdir(im_root)):
        with open(im_root + file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        for line in content.values():
            tokens = tokenizer.tokenize(line)
            for t in tokens:
                flag[vocab_to_index[t]] = 1

for i in range(200):
    flag[i] = 1

vocab_small = [vocab[i] + '\n' for i in range(len(flag)) if flag[i]]
with open('vocab_small.txt', 'w', encoding='utf-8') as f:
    f.writelines(vocab_small)
