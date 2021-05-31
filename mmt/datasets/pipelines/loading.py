import mmcv
import numpy as np
import pdb
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
            print(e, f' while loading {results["image_anns"]}')
            pdb.set_trace()
            return None
