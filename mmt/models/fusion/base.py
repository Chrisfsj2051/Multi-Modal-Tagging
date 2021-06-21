from abc import ABCMeta
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16  # noqa


class BaseFusionModel(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""
    def __init__(self):
        super(BaseFusionModel, self).__init__()
        self.fp16_enabled = False

    def train_step(self, data, optimizer):

        # for name, param in self.named_parameters():
        #     print(name)

        # for param in optimizer.param_groups:
        #     print(param['params'][0].shape, param['lr'])

        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss,
                       log_vars=log_vars,
                       num_samples=len(data['gt_labels']))

        return outputs

    # @auto_fp16(apply_to=('img', ))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)
