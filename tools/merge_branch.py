import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='checkpoint file')
    parser.add_argument('--image', help='checkpoint file')
    parser.add_argument('--video', help='checkpoint file')
    parser.add_argument('--audio', help='checkpoint file')
    parser.add_argument('--out', help='checkpoint file')
    return parser.parse_args()


def main():
    args = parse_args()
    text_ckpt = torch.load(args.text)['state_dict']
    image_ckpt = torch.load(args.image)['state_dict']
    video_ckpt = torch.load(args.video)['state_dict']
    audio_ckpt = torch.load(args.audio)['state_dict']
    ret = {}
    for typ, dic in zip(['text', 'image', 'video', 'audio'],
                        [text_ckpt, image_ckpt, video_ckpt, audio_ckpt]):
        for key, val in dic.items():
            if key.startswith(typ):
                ret[key] = val
    torch.save(ret, args.out)
    print(f'Saved as {args.out}')


if __name__ == '__main__':
    main()
