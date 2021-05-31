import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class Pad(object):
    def __init__(self, size=None, pad_val=0):
        self.size = size
        self.pad_val = pad_val

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key],
                                      shape=results['pad_shape'][:2])

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        # ((d1_padL, d1_padR), (d2_padL, d2_padR))
        pad_shape = [[0, x - y]
                     for x, y in zip(self.size, results['video'].shape)]
        results['video'] = np.pad(results['video'],
                                  pad_shape,
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
