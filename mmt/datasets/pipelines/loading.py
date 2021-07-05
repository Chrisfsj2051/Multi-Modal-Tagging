import json
import os
import random
from copy import deepcopy

import librosa
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotations(object):
    def __init__(self, replace_dict=dict()):
        self.replace_dict = replace_dict

    def load_video_anns(self, results):
        if 'video' in self.replace_dict.keys():
            assert self.replace_dict['video'][0] in results['video_anns'], \
                f'{results["video_anns"]} doesn"t contain {self.replace_dict["video"][0]}'
            results['video_anns'] = results['video_anns'].replace(
                self.replace_dict['video'][0], self.replace_dict['video'][1])
        results['video'] = np.load(results.pop('video_anns')).astype(np.float32)
        results['meta_info']['video_len'] = results['video'].shape[0]

    def load_audio_anns(self, results):
        if 'audio' in self.replace_dict.keys():
            assert self.replace_dict['audio'][0] in results['audio_anns']
            results['audio_anns'] = results['audio_anns'].replace(
                self.replace_dict['audio'][0], self.replace_dict['audio'][1])
        results['audio'] = np.load(results.pop('audio_anns')).astype(np.float32)

    def load_text_anns(self, results):
        with open(results['text_anns'], 'r', encoding='utf-8') as f:
            results['text'] = json.load(f)
        results['text']['video_ocr'] = results['text']['video_ocr'].replace(
            '|', ',')
        results['text']['video_asr'] = results['text']['video_asr'].replace(
            '|', ',')
        assert (len(results['text']['video_ocr'])
                or len(results['text']['video_asr']))
        if len(results['text']['video_asr']) == 0:
            results['text']['video_asr'] = deepcopy(
                results['text']['video_ocr'])
        if len(results['text']['video_ocr']) == 0:
            results['text']['video_ocr'] = deepcopy(
                results['text']['video_asr'])

    def load_image_anns(self, results):
        results['image'] = mmcv.imread(results['image_anns'])
        assert results['image'] is not None

    def __call__(self, results):
        if 'meta_info' not in results.keys():
            results['meta_info'] = {}
        results['meta_info']['id_name'] = results['audio_anns'].split('/')[-1].split('.')[0]
        try:
            self.load_video_anns(results)
            self.load_audio_anns(results)
            self.load_text_anns(results)
            self.load_image_anns(results)
            return results
        except Exception as e:
            print(e, f' while loading {results["image_anns"]}')
            return None


@PIPELINES.register_module()
class LoadAnnotationsWithWAV(LoadAnnotations):
    def load_audio_anns(self, results):
        if 'audio' in self.replace_dict.keys():
            assert self.replace_dict['audio'][0] in results['audio_anns']
            results['audio_anns'] = results['audio_anns'].replace(
                self.replace_dict['audio'][0], self.replace_dict['audio'][1])
        audio_path = results.pop('audio_anns').replace('.npy', '')
        file_list = os.listdir(audio_path)
        file_list = file_list[:(len(file_list) + 1) // 2]
        audio_path = os.path.join(audio_path, random.choice(file_list))
        results['audio'], _ = librosa.core.load(audio_path, sr=3400, mono=True)
