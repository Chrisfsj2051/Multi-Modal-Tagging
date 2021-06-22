import argparse
import os
import subprocess

import mmcv
import numpy as np
from tqdm import tqdm


def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    return parser.parse_args()


def extract_frame(video_filename, video_path, output_path):
    max_frame=300
    output_path = os.path.join(output_path, video_filename.split('.')[0])
    video_filename = os.path.join(video_path, video_filename)
    # assert not os.path.exists(output_path)
    mmcv.mkdir_or_exist(output_path)
    command = "ffmpeg -v error -t %i -i %s -f image2 -vf fps=fps=1 %s/vid%%03d.jpg" % (
        max_frame, video_filename, output_path)
    status, output = subprocess.getstatusoutput(command)
    rgb_path_list = os.listdir(output_path)
    return rgb_path_list

def main():
    args = option()
    assert os.path.exists(args.video_path)
    mmcv.mkdir_or_exist(args.save_path)
    video_list = os.listdir(args.video_path)
    # video_list = [os.path.join(args.video_path, x) for x in video_list]
    video_lens = []
    for video_name in tqdm(video_list):
        rgb_list = extract_frame(video_name, args.video_path, args.save_path)
        video_lens.append(len(rgb_list))
    video_lens = np.array(video_lens)
    print(f'Max_len={video_lens.max()}, Min_len={video_lens.min()}, Median_len={video_lens[len(video_lens)//2]}')

if __name__=='__main__':
    main()