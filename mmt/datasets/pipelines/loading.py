import json

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
                results['text'] = json.load(f)
            results['image'] = mmcv.imread(results['image_anns'])
            assert results['image'] is not None
            return results
        except Exception as e:
            print(e, f' while loading {results["image_anns"]}')
            return None
