from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class Transpose(object):
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """
    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to transpose the channel order of data in results.

        Args:
            results (dict): Result dict contains the data to transpose.

        Returns:
            dict: The result dict contains the data transposed to \
                ``self.order``.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@PIPELINES.register_module()
class ToDataContainer(object):
    """Convert results to :obj:`mmcv.DataContainer` by given fields.

    Args:
        fields (Sequence[dict]): Each field is a dict like
            ``dict(key='xxx', **kwargs)``. The ``key`` in result will
            be converted to :obj:`mmcv.DataContainer` with ``**kwargs``.
            Default: ``(dict(key='img', stack=True), dict(key='gt_bboxes'),
            dict(key='gt_labels'))``.
    """
    def __init__(self,
                 fields=(dict(key='img', stack=True), dict(key='gt_bboxes'),
                         dict(key='gt_labels'))):
        self.fields = fields

    def __call__(self, results):
        """Call function to convert data in results to
        :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted to \
                :obj:`mmcv.DataContainer`.
        """

        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """
        # img = results['image']
        # img = np.ascontiguousarray(img.transpose(2, 0, 1))
        # results['image'] = DC(to_tensor(img), stack=True)
        results['video'] = DC(to_tensor(results['video']),
                              stack=True,
                              pad_dims=1)
        results['image'] = DC(to_tensor(results['image'].transpose(2, 0, 1)),
                              stack=True,
                              pad_dims=1)
        if 'gt_labels' in results.keys():
            results['gt_labels'] = DC(to_tensor(results['gt_labels']))
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect(object):
    def __init__(self, keys, meta_keys=[]):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        # data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'
