import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch_id', help='checkpoint file', choices=['a', 'b'])
    parser.add_argument('--out', help='checkpoint file')
    parser.add_argument('--no_keep_head', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.text)
    if 'state_dict' in ckpt.keys():
        ckpt = ckpt['state_dict']

    ret = {}
    for key, val in ckpt.items():
        if f'backbone_{args.branch_id}' in key:
            new_key = key.replace(f'backbone_{args.branch_id}', 'backbone')
            print(new_key)
            # ret[new_key] = val

    # torch.save(ret, args.out)
    # print(f'Saved as {args.out}')


if __name__ == '__main__':
    main()
