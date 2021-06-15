import argparse
import random


def get_option():
    parse = argparse.ArgumentParser()
    parse.add_argument('--anns')
    parse.add_argument('--out_anns')
    return parse.parse_args()


if __name__ == '__main__':
    args = get_option()
    random.seed(1)
    with open(args.anns, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    modal_path = [[] for _ in range(6)]
    for i in range(0, len(contents), 6):
        for j in range(5):
            modal_path[j].append(contents[i+j].strip())

    print('in')
