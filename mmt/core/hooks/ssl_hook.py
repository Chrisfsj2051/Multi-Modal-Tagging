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


@HOOKS.register_module()
class SemiEMAHook(EMAHook):
    """Semi EMA Hook"""

    def __init__(self, ema_eval, burnin_iters, **kwargs):
        super(SemiEMAHook, self).__init__(**kwargs)
        self.ema_eval = ema_eval
        self.burnin_iters = burnin_iters

    def get_model(self, runner):
        """get model"""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model

    def before_run(self, runner):
        """before run"""
        self.get_model(runner).ema_hook = self
        self.initialized = False

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        if not self.get_model(runner).burnin and self.ema_eval:
            self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        if not self.get_model(runner).burnin and self.ema_eval:
            self._swap_ema_parameters()

    def before_train_iter(self, runner):
        """before train iter"""
        model = self.get_model(runner)
        model.burnin = self.burnin_iters >= runner.iter
        if not model.burnin and not self.initialized:
            self.param_ema_buffer = {}
            self.model_parameters = dict(model.named_parameters(recurse=True))
            for name, value in self.model_parameters.items():
                # "." is not allowed in module's buffer name
                buffer_name = f"ema_{name.replace('.', '_')}"
                self.param_ema_buffer[name] = buffer_name
                model.register_buffer(buffer_name, value.data.clone())
            self.model_buffers = dict(model.named_buffers(recurse=True))
            self.initialized = True

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        model = self.get_model(runner)
        if model.burnin:
            return
        curr_step = runner.iter - self.burnin_iters
        # We warm up the momentum considering the instability at beginning
        momentum = min(self.momentum,
                       (1 + curr_step) / (self.warm_up + curr_step))
        if curr_step % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            buffer_name = self.param_ema_buffer[name]
            buffer_parameter = self.model_buffers[buffer_name]
            buffer_parameter.mul_(1 - momentum).add_(momentum, parameter.data)
