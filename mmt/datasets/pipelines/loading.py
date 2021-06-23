import json
from copy import deepcopy

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotations(object):
    def __init__(self, replace_dict=dict()):
        self.replace_dict = replace_dict

    def replace_anns(self, results):
        if 'video' in self.replace_dict.keys():
            assert self.replace_dict['video'][0] in results['video_anns']
            results['video_anns'] = results['video_anns'].replace(
                self.replace_dict['video'][0], self.replace_dict['video'][1])
        return results

    def __call__(self, results):
        results = self.replace_anns(results)
        try:
            results['id_name'] = results['audio_anns'].split('/')[-1].split(
                '.')[0]
            results['audio'] = np.load(results['audio_anns']).astype(
                np.float32)
            results['video'] = np.load(results['video_anns']).astype(
                np.float32)
            with open(results['text_anns'], 'r', encoding='utf-8') as f:
                results['text'] = json.load(f)
            results['text']['video_ocr'] = results['text'][
                'video_ocr'].replace('|', ',')
            assert (len(results['text']['video_ocr'])
                    or len(results['text']['video_asr']))
            if len(results['text']['video_asr']) == 0:
                results['text']['video_asr'] = deepcopy(
                    results['text']['video_ocr'])
            if len(results['text']['video_ocr']) == 0:
                results['text']['video_ocr'] = deepcopy(
                    results['text']['video_asr'])
            results['image'] = mmcv.imread(results['image_anns'])
            assert results['image'] is not None
            return results
        except Exception as e:
            print(e, f' while loading {results["image_anns"]}')
            return None
