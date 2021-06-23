import abc
import os
import random
from copy import deepcopy

import mmcv
import numpy as np

from mmt.utils.third_party.bert_pytorch.pytorch_pretrained import BertTokenizer
from mmt.utils.tokenization import FullTokenizer

from ..builder import PIPELINES


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
        if results['video'].shape[0] > self.video_pad_size[0]:
            results['video'] = results['video'][:self.video_pad_size[0]]
        if results['audio'].shape[0] > self.audio_pad_size[0]:
            results['audio'] = results['audio'][:self.audio_pad_size[0]]
        video_pad_shape = [[
            0, x - y
        ] for x, y in zip(self.video_pad_size, results['video'].shape)]
        audio_pad_shape = [[
            0, x - y
        ] for x, y in zip(self.audio_pad_size, results['audio'].shape)]
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
class BertTokenize(object):
    def __init__(self, bert_path, max_length, hybrid=False):
        assert max_length <= 512, 'Re-train bert if max_length>512'
        self.hybrid = hybrid
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_length = max_length

    def tokenize(self, text):
        PAD, CLS = '[PAD]', '[CLS]'  # noqa
        token = self.tokenizer.tokenize(text)
        assert sum([x == '[UNK]' for x in token]) / len(token) < 0.5, \
            'Please check if the vocab file is correctly loaded'
        token = [CLS] + token
        seq_len = len(token)
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        pad_size = self.max_length
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
        assert sum(mask) == seq_len
        return token_ids, mask, seq_len

    def __call__(self, results):
        text = results.pop('text')
        if 'meta_info' not in results.keys():
            results['meta_info'] = {}
        if self.hybrid:
            assert self.max_length % 4 == 0
            block_len = self.max_length // 4
            if len(text['video_ocr']) > block_len * 3:
                text['video_ocr'] = text['video_ocr'][:block_len * 3]
                text['video_ocr'] = text['video_ocr'][:-1] + '#'
                text['video_ocr'] += text['video_asr'][-block_len:]
            else:
                text['video_ocr'] += '#'
                remain_len = self.max_length - len(text['video_ocr'])
                text['video_ocr'] += text['video_asr'][-remain_len:]
            ocr_token, ocr_mask, ocr_seq_len = self.tokenize(text['video_ocr'])
            results['text'] = ocr_token
            results['meta_info']['text_mask'] = ocr_mask
            results['meta_info']['text_seq_len'] = ocr_seq_len
        else:
            ocr_token, ocr_mask, ocr_seq_len = self.tokenize(text['video_ocr'])
            asr_token, asr_mask, asr_seq_len = self.tokenize(text['video_asr'])
            results['ocr_text'], results['asr_text'] = ocr_token, asr_token
            results['meta_info']['ocr_mask'] = ocr_mask
            results['meta_info']['asr_mask'] = asr_mask
            results['meta_info']['ocr_seq_len'] = ocr_seq_len
            results['meta_info']['asr_seq_len'] = asr_seq_len
        return results


@PIPELINES.register_module()
class Tokenize(object):
    def __init__(self, vocab_root, max_length):
        raise NotImplementedError('Please use BertTokenize')
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
    def __init__(self, key_fields, aug_num_frame, aug_num_block, aug_max_len,
                 aug_max_size):
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
            ed = random.randint(st + 1, min(x.shape[1],
                                            st + self.aug_max_size))
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
            ed = random.randint(st + 1, min(x.shape[1],
                                            st + self.aug_max_size))
            x[st:ed] = x[st:ed][::-1]
        return x


@PIPELINES.register_module()
class FrameRandomSwap(FrameAugBox):
    def apply_frame_aug(self, x):
        for cnt in range(self.aug_num_frame):
            st1 = random.randint(0, x.shape[0] - 1)
            ed1 = random.randint(st1 + 1,
                                 min(x.shape[0], st1 + self.aug_max_len))
            st2 = random.randint(0, x.shape[0] - ed1 + st1)
            ed2 = st2 + ed1 - st1
            temp = deepcopy(x[st1:ed1])
            x[st1:ed1] = x[st2:ed2]
            x[st2:ed2] = temp

        return x

    def apply_block_aug(self, x):
        for cnt in range(self.aug_num_block):
            st1 = random.randint(0, x.shape[1] - 1)
            ed1 = random.randint(st1 + 1,
                                 min(x.shape[1], st1 + self.aug_max_size))
            st2 = random.randint(0, x.shape[1] - ed1 + st1)
            ed2 = st2 + ed1 - st1
            temp = deepcopy(x[:, st1:ed1])
            x[:, st1:ed1] = x[:, st2:ed2]
            x[:, st2:ed2] = temp
        return x


@PIPELINES.register_module()
class TextOfflineAug(object):
    def __init__(self, aug_prob, aug_root, valid_index=None):
        self.aug_prob = aug_prob
        self.aug_root = aug_root
        self.valid_index = valid_index

    def __call__(self, results):
        if random.uniform(0, 1) > self.aug_prob:
            return results
        assert 'id_name' in results.keys()
        for key in ['video_asr', 'video_ocr']:
            data_path = os.path.join(self.aug_root, results['id_name'], key)
            file_list = sorted(list(os.listdir(data_path)))
            # if self.valid_index is not None:
            #     print('in')
            file = random.choice(file_list)
            with open(os.path.join(data_path, file), 'r',
                      encoding='utf-8') as f:
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
            results['image'] = mmcv.imflip(results['image'],
                                           direction=flip_direction)

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
        ori_dtype = results['image'].dtype
        img = results['image'].astype(np.float32)
        # random brightness
        if np.random.randint(0, 2):
            delta = np.random.uniform(-self.brightness_delta,
                                      self.brightness_delta)
            img += delta

        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower,
                                          self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(self.saturation_lower,
                                             self.saturation_upper)

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower,
                                          self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]

        results['image'] = img.astype(ori_dtype)
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


@PIPELINES.register_module()
class CutOut:
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.
    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """
    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0)):

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def __call__(self, results):
        """Call function to drop some regions of image."""
        h, w, c = results['image'].shape
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            results['image'][y1:y2, x1:x2, :] = self.fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in})'
        return repr_str
