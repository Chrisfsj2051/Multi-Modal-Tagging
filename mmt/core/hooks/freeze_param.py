from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook

from mmt.utils import get_root_logger


@HOOKS.register_module()
class FreezeParamHook(Hook):
    def __init__(self, param_pattern: list, eval_pattern: list, freeze_iters):
        self.param_pattern = param_pattern
        self.freeze_iters = freeze_iters
        assert freeze_iters > 0
        self.enabled = False
        self.eval_print_flag = True

    def before_iter(self, runner):
        logger = get_root_logger()
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if runner.iter < self.freeze_iters:
            for key, module in model.named_modules():
                for pattern in self.param_pattern:
                    if pattern in key:
                        module.eval()
                        if self.eval_print_flag:
                            logger.info(f'Eval {key}')

            self.eval_print_flag = False

        if runner.iter < self.freeze_iters and not self.enabled:
            for key, param in model.named_parameters():
                for pattern in self.param_pattern:
                    if pattern in key:
                        param.requires_grad = False
                        logger.info(f'Freeze {key}')
            self.enabled = True
        elif runner.iter >= self.freeze_iters and self.enabled:
            for key, param in model.named_parameters():
                for pattern in self.param_pattern:
                    if pattern in key:
                        param.requires_grad = True
                        logger.info(f'Activate {key}')
            self.enabled = False
