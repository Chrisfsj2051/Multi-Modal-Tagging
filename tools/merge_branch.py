import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='checkpoint file')
    parser.add_argument('--image', help='checkpoint file')
    parser.add_argument('--video', help='checkpoint file')
    parser.add_argument('--audio', help='checkpoint file')
    parser.add_argument('--out', help='checkpoint file')
    parser.add_argument('--no_keep_head', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    text_ckpt = torch.load(args.text)
    image_ckpt = torch.load(args.image)
    video_ckpt = torch.load(args.video)
    audio_ckpt = torch.load(args.audio)
    if 'state_dict' in text_ckpt.keys():
        text_ckpt = text_ckpt['state_dict']
    if 'state_dict' in image_ckpt.keys():
        image_ckpt = image_ckpt['state_dict']
    if 'state_dict' in video_ckpt.keys():
        video_ckpt = video_ckpt['state_dict']
    if 'state_dict' in audio_ckpt.keys():
        audio_ckpt = audio_ckpt['state_dict']
    ret = {}
    for typ, dic in zip(['text', 'image', 'video', 'audio'],
                        [text_ckpt, image_ckpt, video_ckpt, audio_ckpt]):
        for key, val in dic.items():
            ret[typ + '_branch.'+key] = val
        #     if key.startswith(typ) and (not args.no_keep_head
        #                                 or 'head' not in key):
        #         ret[key] = val
    torch.save(ret, args.out)
    print(f'Saved as {args.out}')


if __name__ == '__main__':
    main()
