from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, EMAHook, Hook


@HOOKS.register_module()
class SemiStatusHook(Hook):

    def __init__(self, burnin_iters):
        super(SemiStatusHook, self).__init__()
        self.burnin_iters = burnin_iters

    def get_model(self, runner):
        """get model"""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        model = self.get_model(runner)
        model.burnin = self.burnin_iters >= runner.iter
