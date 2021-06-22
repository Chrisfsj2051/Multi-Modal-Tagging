import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori', help='checkpoint file')
    parser.add_argument('--new', help='checkpoint file')
    parser.add_argument('--modal', help='checkpoint file')
    parser.add_argument('--out', help='checkpoint file')
    return parser.parse_args()


def main():
    args = parse_args()

    old_ckpt = torch.load(args.ori)
    new_ckpt = torch.load(args.new)
    if 'state_dict' in old_ckpt.keys():
        old_ckpt = old_ckpt['state_dict']
    if 'state_dict' in new_ckpt.keys():
        new_ckpt = new_ckpt['state_dict']

    for key, val in new_ckpt.items():
        branch_key = args.modal + '_branch.' + key
        assert key in old_ckpt.keys()
        old_ckpt[branch_key] = val
    torch.save(old_ckpt, args.out)
    print(f'Saved as {args.out}')


if __name__ == '__main__':
    main()
