import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from torch.autograd.function import Function

class SyncBatchnormFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_variance, eps, process_group, world_size):
        torch.cuda.nvtx.range_push("sync_BN_fw")
        # transpose it to channel last to support broadcasting for input with different rank
        c_last_input = input.transpose(1, -1).contiguous().clone()

        ctx.save_for_backward(c_last_input, weight, bias,
                              running_mean, running_variance)
        ctx.eps = eps
        ctx.process_group = process_group
        ctx.world_size = world_size

        c_last_input = (c_last_input - running_mean) / \
            torch.sqrt(running_variance + eps)

        if weight is not None:
            c_last_input = c_last_input * weight
        if bias is not None:
            c_last_input = c_last_input + bias

        torch.cuda.nvtx.range_pop()
        return c_last_input.transpose(1, -1).contiguous().clone()

    @staticmethod
    def backward(ctx, grad_output):
        torch.cuda.nvtx.range_push("sync_BN_bw")
        # mini batch mean & var are calculated by forward path.
        # mu = 1./N*np.sum(h, axis = 0)
        # var = 1./N*np.sum((h-mu)**2, axis = 0)
        c_last_input, weight, bias, running_mean, running_variance = ctx.saved_tensors

        eps = ctx.eps
        process_group = ctx.process_group
        world_size = ctx.world_size
        grad_input = grad_weight = grad_bias = None
        num_features = running_mean.size()[0]

        # transpose it to channel last to support broadcasting for input with different rank
        torch.cuda.nvtx.range_push("carilli field")
        c_last_grad = grad_output.transpose(1, -1).contiguous()
        # squash non-channel dimension so we can easily calculate mean
        c_grad = c_last_grad.view(-1, num_features).contiguous()
        torch.cuda.nvtx.range_pop()

        # calculate grad_input
        if ctx.needs_input_grad[0]:
            # dh = gamma * (var + eps)**(-1. / 2.) * (dy - np.mean(dy, axis=0)
            #     - (h - mu) * (var + eps)**(-1.0) * np.mean(dy * (h - mu), axis=0))
            mean_dy = c_grad.mean(0)
            mean_dy_xmu = (c_last_grad * (c_last_input -
                                          running_mean)).view(-1, num_features).mean(0)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    mean_dy, torch.distributed.ReduceOp.SUM, process_group)
                mean_dy = mean_dy / world_size
                torch.distributed.all_reduce(
                    mean_dy_xmu, torch.distributed.ReduceOp.SUM, process_group)
                mean_dy_xmu = mean_dy_xmu / world_size
            c_last_grad_input = (c_last_grad - mean_dy - (c_last_input - running_mean) / (
                running_variance + eps) * mean_dy_xmu) / torch.sqrt(running_variance + eps)
            if weight is not None:
                c_last_grad_input.mul_(weight)
            grad_input = c_last_grad_input.transpose(1, -1).contiguous()

        # calculate grad_weight
        grad_weight = None
        if weight is not None and ctx.needs_input_grad[1]:
            # dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
            grad_weight = ((c_last_input - running_mean) / torch.sqrt(
                running_variance + eps) * c_last_grad).view(-1, num_features).sum(0)

        # calculate grad_bias
        grad_bias = None
        if bias is not None and ctx.needs_input_grad[2]:
            # dbeta = np.sum(dy, axis=0)
            grad_bias = c_grad.sum(0)

        torch.cuda.nvtx.range_pop()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class SyncBatchNorm(_BatchNorm):
    """
    synchronized batch normalization module extented from ``torch.nn.BatchNormNd``
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.SyncBatchNorm` is designed to work with
    ``DistributedDataParallel``.
    When running in training mode, the layer reduces stats across all processes
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model. The model uses collective
    communication package from ``torch.distributed``.
    When running in evaluation mode, the layer falls back to
    ``torch.nn.functional.batch_norm``.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
    """

    warned = False

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False):
        if channel_last == True:
            raise AttributeError("channel_last is not supported by primitive SyncBatchNorm implementation. Try install apex with `--cuda_ext` if channel_last is desired.")

        if not SyncBatchNorm.warned:
            if hasattr(self, "syncbn_import_error"):
                print("Warning:  using Python fallback for SyncBatchNorm, possibly because apex was installed without --cuda_ext.  The exception raised when attempting to import the cuda backend was: ", self.syncbn_import_error)
            else:
                print("Warning:  using Python fallback for SyncBatchNorm")
            SyncBatchNorm.warned = True

        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.process_group = process_group

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def forward(self, input):
        torch.cuda.nvtx.range_push("sync_bn_fw_with_mean_var")
        mean = None
        var = None
        cast = None
        out = None

        # casting to handle mismatch input type to layer type
        if self.running_mean is not None:
            if self.running_mean.dtype != input.dtype:
                input = input.to(self.running_mean.dtype)
                cast = input.dtype
        elif self.weight is not None:
            if self.weight.dtype != input.dtype:
                input = input.to(self.weight.dtype)
                cast = input.dtype

        if not self.training and self.track_running_stats:
            # fall back to pytorch implementation for inference
            torch.cuda.nvtx.range_pop()
            out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            process_group = self.process_group
            world_size = 1
            if not self.process_group:
                process_group = torch.distributed.group.WORLD
            self.num_batches_tracked += 1
            with torch.no_grad():
                channel_first_input = input.transpose(0, 1).contiguous()
                squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)
                # total number of data points for each variance entry. Used to calculate unbiased variance estimate
                m = None
                local_m = float(squashed_input_tensor_view.size()[1])
                local_mean = torch.mean(squashed_input_tensor_view, 1)
                local_sqr_mean = torch.pow(
                    squashed_input_tensor_view, 2).mean(1)
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group)
                    torch.distributed.all_reduce(
                        local_mean, torch.distributed.ReduceOp.SUM, process_group)
                    mean = local_mean / world_size
                    torch.distributed.all_reduce(
                        local_sqr_mean, torch.distributed.ReduceOp.SUM, process_group)
                    sqr_mean = local_sqr_mean / world_size
                    m = local_m * world_size
                else:
                    m = local_m
                    mean = local_mean
                    sqr_mean = local_sqr_mean
                # var(x) = E (( x - mean_x ) ** 2)
                #        = 1 / N * sum ( x - mean_x ) ** 2
                #        = 1 / N * sum (x**2) - mean_x**2
                var = sqr_mean - mean.pow(2)

                if self.running_mean is not None:
                    self.running_mean = self.momentum * mean + \
                        (1 - self.momentum) * self.running_mean
                if self.running_var is not None:
                    # as noted by the paper, we used unbiased variance estimate of the mini-batch
                    # Var[x] = m / (m-1) * Eb (sample_variance)
                    self.running_var = m / \
                        (m-1) * self.momentum * var + \
                        (1 - self.momentum) * self.running_var
            torch.cuda.nvtx.range_pop()
            out = SyncBatchnormFunction.apply(input, self.weight, self.bias, mean, var, self.eps, process_group, world_size)
        return out.to(cast)



class SyncBatchnormFunction_optimize(Function):
   # import syncbn
    @staticmethod
    def forward(ctx, input, z, weight, bias, running_mean, running_variance, eps, track_running_stats = True, momentum = 1.0, process_group = None, channel_last = False, fuse_relu = False):
        input = input.contiguous()
        world_size = 0

        mean = None
        var_biased = None
        inv_std = None
        var = None
        out = None
        count = None
        if track_running_stats:
            if channel_last:
                count = int(input.numel()/input.size(-1))
                mean, var_biased = syncbn.welford_mean_var_c_last(input)
                num_channels = input.size(-1)
            else:
                count = int(input.numel()/input.size(1))
                mean, var_biased = syncbn.welford_mean_var(input)
                num_channels = input.size(1)

            if torch.distributed.is_initialized():
                if not process_group:
                    process_group = torch.distributed.group.WORLD
                device = mean.device
                world_size = torch.distributed.get_world_size(process_group)

                count_t = torch.empty(1, dtype=mean.dtype, device=mean.device).fill_(count)
                combined = torch.cat([mean.view(-1), var_biased.view(-1), count_t], dim=0)
                combined_list = [torch.empty_like(combined) for k in range(world_size)]
                torch.distributed.all_gather(combined_list, combined, process_group)
                combined = torch.stack(combined_list, dim=0)
                mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
                count_all = count_all.view(-1)
                mean, var, inv_std = syncbn.welford_parallel(mean_all, invstd_all, count_all.to(torch.int32), eps)
            else:
                device = mean.device
                count_all = torch.cuda.IntTensor([count], device=device)
                inv_std = 1.0 / torch.sqrt(var_biased + eps)
                var = var_biased * (count) / (count-1)

            if count == 1 and world_size < 2:
                raise ValueError('Expected more than 1 value per channel when training, got input size{}'.format(input.size()))

            r_m_inc = mean if running_mean.dtype != torch.float16 else mean.half()
            r_v_inc = var if running_variance.dtype != torch.float16 else var.half()
            running_mean.data = running_mean.data * (1-momentum) + momentum*r_m_inc
            running_variance.data = running_variance.data * (1-momentum) + momentum*r_v_inc
        else:
            mean = running_mean.data
            inv_std = 1.0 / torch.sqrt(running_variance.data + eps)

        ctx.save_for_backward(input, weight, mean, inv_std, z, bias, count_all.to(torch.int32))
        ctx.process_group = process_group
        ctx.channel_last = channel_last
        ctx.world_size = world_size
        ctx.fuse_relu = fuse_relu

        if channel_last:
            out = syncbn.batchnorm_forward_c_last(input, z, mean, inv_std, weight, bias, fuse_relu)
        else:
            out = syncbn.batchnorm_forward(input, mean, inv_std, weight, bias)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        # mini batch mean & var are calculated by forward path.
        # mu = 1./N*np.sum(h, axis = 0)
        # var = 1./N*np.sum((h-mu)**2, axis = 0)
        saved_input, weight, mean, inv_std, z, bias, count = ctx.saved_tensors
        process_group = ctx.process_group
        channel_last = ctx.channel_last
        world_size = ctx.world_size
        fuse_relu = ctx.fuse_relu
        grad_input = grad_z = grad_weight = grad_bias = None

        if fuse_relu:
            grad_output = syncbn.relu_bw_c_last(grad_output, saved_input, z, mean, inv_std, weight, bias)
        if isinstance(z, torch.Tensor) and ctx.needs_input_grad[1]:
            grad_z = grad_output.clone()

        # TODO: update kernel to not pre_divide by item_num
        if channel_last:
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn_c_last(grad_output, saved_input, mean, inv_std, weight)
        else:
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn(grad_output, saved_input, mean, inv_std, weight)

        # calculate grad_input
        if ctx.needs_input_grad[0]:

            if torch.distributed.is_initialized():
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(
                    combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

            if channel_last:
                grad_input = syncbn.batchnorm_backward_c_last(grad_output, saved_input, mean, inv_std, weight, sum_dy, sum_dy_xmu, count)
            else:
                grad_input = syncbn.batchnorm_backward(grad_output, saved_input, mean, inv_std, weight, sum_dy, sum_dy_xmu, count)

        if weight is None or not ctx.needs_input_grad[2]:
            grad_weight = None

        if weight is None or not ctx.needs_input_grad[3]:
            grad_bias = None

        return grad_input, grad_z, grad_weight, grad_bias, None, None, None, None, None, None, None, None


class SyncBatchNorm_optimize(_BatchNorm):
    """
    synchronized batch normalization module extented from `torch.nn.BatchNormNd`
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.SyncBatchNorm` is designed to work with
    `DistributedDataParallel`.

    When running in training mode, the layer reduces stats across all processes
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model. The model uses collective
    communication package from `torch.distributed`.

    When running in evaluation mode, the layer falls back to
    `torch.nn.functional.batch_norm`

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
        process_group: pass in a process group within which the stats of the
            mini-batch is being synchronized. ``None`` for using default process
            group
        channel_last: a boolean value that when set to ``True``, this module
            take the last dimension of the input tensor to be the channel
            dimension. Default: False

    Examples::
      #  >>> # channel first tensor
      #  >>> sbn = apex.parallel.SyncBatchNorm(100).cuda()
      #  >>> inp = torch.randn(10, 100, 14, 14).cuda()
      #  >>> out = sbn(inp)
      #  >>> inp = torch.randn(3, 100, 20).cuda()
      #  >>> out = sbn(inp)
      #  >>> # channel last tensor
      #  >>> sbn = apex.parallel.SyncBatchNorm(100, channel_last=True).cuda()
      #  >>> inp = torch.randn(10, 14, 14, 100).cuda()
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False, fuse_relu=False):
        super(SyncBatchNorm_optimize, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.process_group = process_group
        self.channel_last = channel_last
        self.fuse_relu = fuse_relu

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def _specify_channel_last(self, channel_last):
        self.channel_last = channel_last

    def forward(self, input, z = None):
        # if input.dim() == 2, we switch to channel_last for efficient memory accessing
        channel_last = self.channel_last if input.dim() != 2 else True

        if not self.training and self.track_running_stats and not self.channel_last and not self.fuse_relu and z == None:
            # fall back to pytorch implementation for inference
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            exponential_average_factor = 0.0
            if self.training and self.track_running_stats:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
            return SyncBatchnormFunction_optimize.apply(input, z, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.training or not self.track_running_stats, exponential_average_factor, self.process_group, self.channel_last, self.fuse_relu)

if __name__ == '__main__':
    sbn = SyncBatchNorm(100).cuda()
    inp = torch.randn(10, 100, 14, 14).cuda()
    out = sbn(inp)
    print(inp)
    print(out)
    #can not work below
    exit()
    sbn2 = SyncBatchNorm_optimize(100).cuda()
    out2 = sbn2(inp)
    print(out2)
