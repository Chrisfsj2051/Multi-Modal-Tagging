import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotations(object):
    def __call__(self, results):
        try:
            results['audio'] = np.load(results['audio_anns'])
            results['video'] = np.load(results['video_anns'])
            with open(results['text_anns'], 'r', encoding='utf-8') as f:
                results['text'] = f.readline().strip()
            results['image'] = mmcv.imread(results['image_anns'])
            return results
        except Exception as e:
            print(e)
            return  None
