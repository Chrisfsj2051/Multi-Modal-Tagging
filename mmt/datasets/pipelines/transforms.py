import random

import mmcv
import numpy as np

from ..builder import PIPELINES
from ...utils.tokenization import FullTokenizer


@PIPELINES.register_module()
class Pad(object):
    def __init__(self, video_pad_size, audio_pad_size, pad_val=0):
        self.video_pad_size = video_pad_size
        self.audio_pad_size = audio_pad_size
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        # ((d1_padL, d1_padR), (d2_padL, d2_padR))
        video_pad_shape = [[0, x - y]
                     for x, y in zip(self.video_pad_size, results['video'].shape)]
        audio_pad_shape = [[0, x - y]
                     for x, y in zip(self.audio_pad_size, results['audio'].shape)]
        results['video'] = np.pad(results['video'],
                                  video_pad_shape,
                                  constant_values=self.pad_val)
        results['audio'] = np.pad(results['audio'],
                                  audio_pad_shape,
                                  constant_values=self.pad_val)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, results):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(results['image'],
                                              size=self.size,
                                              return_scale=True)
        results['image'] = img
        return results

@PIPELINES.register_module()
class Tokenize(object):
    def __init__(self, vocab_root, max_length):
        self.tokenizer = FullTokenizer(vocab_root)
        self.max_length = max_length

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pad = self.tokenizer.convert_tokens_to_ids(['[PAD]'])
        ids = ids + (self.max_length - len(ids)) * pad
        return ids[:self.max_length]

    def __call__(self, results):
        text_dict = results.pop('text')
        results['ocr_text'] = self.tokenize(text_dict['video_ocr'])
        results['asr_text'] = self.tokenize(text_dict['video_asr'])
        return results


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['image'] = mmcv.imnormalize(results['image'], self.mean,
                                            self.std)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str



@PIPELINES.register_module()
class FrameRandomErase(object):
    def __init__(self, key_fields, erase_num, erase_max_len):
        self.key_fields = key_fields
        self.erase_num = erase_num
        self.erase_max_len = erase_max_len

    def __call__(self, results):
        for key in self.key_fields:
            assert key  in results.keys()
            item = results[key]
            assert item.ndim == 2
            for cnt in range(self.erase_num):
                st = random.randint(0, item.shape[0]-1)
                ed = random.randint(st+1, min(item.shape[0], st+self.erase_max_len))
                results[key][st:ed] *= 0
        return results