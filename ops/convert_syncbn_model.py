from ops.SyncBatchNorm import SyncBatchNorm
import torch

def convert_syncbn_model(module, process_group=None, channel_last=False):
    '''
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`apex.parallel.SyncBatchNorm`.

    All ``torch.nn.BatchNorm*N*d`` wrap around
    ``torch.nn.modules.batchnorm._BatchNorm``, so this function lets you easily switch
    to use sync BN.

    Args:
        module (torch.nn.Module): input module

    Example::

      #  >>> # model is an instance of torch.nn.Module
      #  >>> import apex
      #  >>> sync_bn_model = apex.parallel.convert_syncbn_model(model)
    '''
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, process_group, channel_last=channel_last)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_syncbn_model(child,
                                                  process_group=process_group,
                                                  channel_last=channel_last))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod
from model.MomentumBN import MomentumBatchNorm1d,MomentumBatchNorm2d,MomentumSN
from model.Small_BatchNorm import Small_BatchNormSN
def convert_momentumbn_model(module,momentum):
    '''
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`apex.parallel.SyncBatchNorm`.

    All ``torch.nn.BatchNorm*N*d`` wrap around
    ``torch.nn.modules.batchnorm._BatchNorm``, so this function lets you easily switch
    to use sync BN.

    Args:
        module (torch.nn.Module): input module

    Example::

      #  >>> # model is an instance of torch.nn.Module
      #  >>> import apex
      #  >>> sync_bn_model = apex.parallel.convert_syncbn_model(model)
    '''
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
            mod = MomentumBatchNorm1d(module.num_features, module.eps, momentum, module.affine,
                                      module.track_running_stats)
        else:
            mod = MomentumBatchNorm2d(module.num_features, module.eps, momentum, module.affine,
                                      module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    #add special support for group batch norm
    if isinstance(module,Small_BatchNormSN):
        print("convert sn to momentum sn!")
        mod = MomentumSN(module.num_group,module.num_features, module.eps, momentum, module.affine,
                                      module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_momentumbn_model(child,momentum))
    del module
    return mod

from Small_BatchNorm import Small_BatchNormSN
def convert_groupbn_model(module,num_group):
    '''
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`apex.parallel.SyncBatchNorm`.

    All ``torch.nn.BatchNorm*N*d`` wrap around
    ``torch.nn.modules.batchnorm._BatchNorm``, so this function lets you easily switch
    to use group BN.

    Args:
        module (torch.nn.Module): input module

    '''
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = Small_BatchNormSN(
                 num_group,
                 module.num_features,
                 eps=module.eps,
                 momentum=module.momentum,
                 affine=module.affine,
                 track_running_stats=module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_groupbn_model(child,num_group))
    del module
    return mod
