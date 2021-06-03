import abc
import os
import random
from copy import deepcopy

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


class FrameAugBox(metaclass=abc.ABCMeta):

    def __init__(self, key_fields, aug_num_frame, aug_num_block, aug_max_len, aug_max_size):
        self.key_fields = key_fields
        self.aug_num_frame = aug_num_frame
        self.aug_num_block = aug_num_block
        self.aug_max_len = aug_max_len
        self.aug_max_size = aug_max_size

    def frame_aug(self, results):
        for key in self.key_fields:
            assert key in results.keys()
            assert results[key].ndim == 2
            results[key] = self.apply_frame_aug(results[key])
        return results

    def block_aug(self, results):
        for key in self.key_fields:
            assert key in results.keys()
            assert results[key].ndim == 2
            results[key] = self.apply_block_aug(results[key])

    def __call__(self, results):
        self.frame_aug(results)
        self.block_aug(results)
        return results


@PIPELINES.register_module()
class FrameRandomErase(FrameAugBox):

    def apply_frame_aug(self, x):
        for cnt in range(self.aug_num_frame):
            st = random.randint(0, x.shape[0] - 1)
            ed = random.randint(st + 1, min(x.shape[0], st + self.aug_max_len))
            x[st:ed] *= 0
        return x

    def apply_block_aug(self, x):
        for cnt in range(self.aug_num_block):
            st = random.randint(0, x.shape[1] - 1)
            ed = random.randint(st + 1, min(x.shape[1], st + self.aug_max_size))
            x[:, st:ed] *= 0
        return x


@PIPELINES.register_module()
class FrameRandomReverse(FrameAugBox):

    def apply_frame_aug(self, x):
        for cnt in range(self.aug_num_frame):
            st = random.randint(0, x.shape[0] - 1)
            ed = random.randint(st + 1, min(x.shape[0], st + self.aug_max_len))
            x[st:ed] = x[st:ed][::-1]

        return x

    def apply_block_aug(self, x):
        for cnt in range(self.aug_num_block):
            st = random.randint(0, x.shape[1] - 1)
            ed = random.randint(st + 1, min(x.shape[1], st + self.aug_max_size))
            x[st:ed] = x[st:ed][::-1]
        return x


@PIPELINES.register_module()
class FrameRandomSwap(FrameAugBox):

    def apply_frame_aug(self, x):
        for cnt in range(self.aug_num_frame):
            st1 = random.randint(0, x.shape[0] - 1)
            ed1 = random.randint(st1 + 1, min(x.shape[0], st1 + self.aug_max_len))
            st2 = random.randint(0, x.shape[0] - ed1 + st1)
            ed2 = st2 + ed1 - st1
            temp = deepcopy(x[st1: ed1])
            x[st1:ed1] = x[st2:ed2]
            x[st2:ed2] = temp

        return x

    def apply_block_aug(self, x):
        for cnt in range(self.aug_num_block):
            st1 = random.randint(0, x.shape[1] - 1)
            ed1 = random.randint(st1 + 1, min(x.shape[1], st1 + self.aug_max_size))
            st2 = random.randint(0, x.shape[1] - ed1 + st1)
            ed2 = st2 + ed1 - st1
            temp = deepcopy(x[:, st1: ed1])
            x[:, st1:ed1] = x[:, st2:ed2]
            x[:, st2:ed2] = temp
        return x


@PIPELINES.register_module()
class TextOfflineAug(object):

    def __init__(self, aug_prob, aug_root):
        self.aug_prob = aug_prob
        self.aug_root = aug_root

    def __call__(self, results):
        if random.uniform(0, 1) < self.aug_prob:
            return results
        assert 'id_name' in results.keys()
        for key in ['video_asr', 'video_orc']:
            data_path = self.aug_root + '/' + results['id_name'] + '/' + key + '/'
            file = random.choice(os.listdir(data_path))
            with open(data_path + file, 'r', encoding='utf-8') as f:
                results['text'][key] = f.read().strip()

        return results


@PIPELINES.register_module()
class RandomFlip(object):
    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def __call__(self, results):
        use_flip, flip_direction = None, None
        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            use_flip = cur_dir is not None

        flip_direction = cur_dir
        if use_flip:
            results['image'] = mmcv.imflip(
                results['image'], direction=flip_direction)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'





@PIPELINES.register_module()
class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        ori_dtype = results['image'].dtype
        img = results['image'].astype(np.float32)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['image'] = img.to(ori_dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str