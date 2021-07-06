import argparse
import multiprocessing
import os
from multiprocessing import Pool

import torch
from tqdm import tqdm

from src.utils.mmt.utils import Cnn14


def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    return parser.parse_args()


def extract_audio_feats(args):
    filename, audio_path, save_path, model = args
    file_root = os.path.join(audio_path, filename)
    feats_list = []
    # for file_path in os.listdir(file_root):
    #     file_path = os.path.join(file_root, file_path)
    #     (waveform, _) = librosa.core.load(file_path, sr=3000000, mono=True)
    #     waveform = waveform[None, :]  # (1, audio_length)
    #     waveform = move_data_to_device(waveform, torch.device('cuda'))
    #     with torch.no_grad():
    #         feats = model(waveform, None)
    #         feats_list.append(feats.tolist())
    # mmcv.mkdir_or_exist(save_path)
    # feats = np.array(feats_list).reshape(len(feats_list), -1)
    # np.save(os.path.join(save_path, filename + '.npy'), feats)


def main():
    args = option()
    sample_rate = 3400
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin, fmax = 50, 14000
    ckpt_path = 'pretrained/Cnn14_mAP=0.431.pth'
    cmd = f'wget -O {ckpt_path} https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth'  # noqa
    print(cmd)
    model = Cnn14(sample_rate=sample_rate,
                  window_size=window_size,
                  hop_size=hop_size,
                  mel_bins=mel_bins,
                  fmin=fmin,
                  fmax=fmax,
                  classes_num=527)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    id_list = os.listdir(args.audio_path)

    # For windows:
    # for ids in tqdm(id_list):
    #     extract_audio_feats([ids, args.audio_path, args.save_path, model])

    multiprocessing.set_start_method('spawn')
    pool = Pool(8)
    list(
        tqdm(pool.imap(extract_audio_feats,
                       [(id_name, args.audio_path, args.save_path, model)
                        for id_name in id_list]),
             total=len(id_list)))
    pool.close()
    pool.join()


if __name__ == '__main__':
    # t  = np.load('dataset/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/6f718a69d07253a1aa5f4ea6aa5934d1.npy')
    # print(t.shape)
    main()
