import argparse
import os
import subprocess
from multiprocessing import Pool

import mmcv
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm


def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    return parser.parse_args()


def extract_frame(args):
    video_filename, video_path, output_path = args
    seg_save_path = os.path.join(output_path,
                                 video_filename.replace('.mp4', ''))
    wav_save_path = '/tmp/' + video_filename.replace('.mp4', '.wav')
    output_path = os.path.join(output_path, video_filename.split('.')[0])
    video_filename = os.path.join(video_path, video_filename)
    # assert not os.path.exists(output_path)
    mmcv.mkdir_or_exist(output_path)
    command = f'ffmpeg -i {video_filename} -ac 1 -ar 16000 -y {wav_save_path}'
    status, output = subprocess.getstatusoutput(command)
    assert status == 0
    # 不支持滑窗分割
    # command = f'ffmpeg -i {wav_save_path} -f segment ' \
    #           f'-segment_time 10 -c copy {seg_save_path}/out%03d.wav'
    # status, output = subprocess.getstatusoutput(command)
    # assert status == 0
    wav = AudioSegment.from_wav(wav_save_path)
    for i in range(0, len(wav) // 1000, 3):
        if i * 1000 > len(wav):
            break
        seg = wav[i * 1000:i * 1000 + 10000]
        if len(seg) < 10000:
            seg = seg + AudioSegment.silent(10000 - len(seg))
        seg.export(f'{seg_save_path}/{i//5:03d}.wav', format='wav')
    return len(os.listdir(seg_save_path))


def main():
    args = option()
    assert os.path.exists(args.video_path)
    mmcv.mkdir_or_exist(args.video_path)
    video_list = os.listdir(args.video_path)
    # video_list = [os.path.join(args.video_path, x) for x in video_list]
    # video_lens = []

    pool = Pool(16)
    audio_lens = list(
        tqdm(pool.imap(extract_frame,
                       [(filename, args.video_path, args.save_path)
                        for filename in video_list]),
             total=len(video_list)))
    # for video_name in tqdm(video_list):
    #     rgb_list = extract_frame(video_name, args.video_path, args.save_path)
    #     video_lens.append(len(rgb_list))
    pool.close()
    pool.join()
    audio_lens = np.array(audio_lens)
    print(f'Max_len={audio_lens.max()}, Min_len={audio_lens.min()}, '
          f'Median_len={audio_lens[len(audio_lens) // 2]}')


if __name__ == '__main__':
    main()
