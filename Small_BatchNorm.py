import torch
import torch.nn as nn

import torch.distributed as dist




class Small_BatchNorm(nn.Module):
    def __init__(self,
                 num_group,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(Small_BatchNorm, self).__init__()

        self.num_group = num_group
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        #G*C normalization weigh and error
        if self.affine:
            self.weight = nn.Parameter(torch.ones([num_features])) #only C dimension keep
            self.bias = nn.Parameter(torch.zeros([num_features]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros([1,self.num_group,num_features]))
            self.register_buffer('running_var', torch.ones([1,self.num_group,num_features]))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

        self.reset_parameters()

    def extra_repr(self):
        return 'num_groups={}, num_features={}, eps={}, momentum={}'.\
            format(self.num_group, self.num_features, self.eps, self.momentum)

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def normalize(self, input):
        """
        input shape
        N*C
        Args:
            input:

        Returns:

        """
        cur_batch_size, num_features = input.shape
        input = torch.reshape(input, (-1, self.num_group, self.num_features)) # (N, G, C)
        var, mean = torch.var_mean(input, dim=0, keepdim=True, unbiased=False)
        input = (input - mean) / torch.sqrt(var + self.eps)

        if self.track_running_stats:

            self.running_mean = self.running_mean * (1. - self.momentum) + mean.detach() * self.momentum
            self.running_var = self.running_var * (1. - self.momentum) + var.detach() * self.momentum

        if self.affine:
            input = input * self.weight + self.bias
        input = input.view(cur_batch_size,num_features)
        return input
    def normalize_bn(self,input):
        var, mean = torch.var_mean(input, dim=0, keepdim=True, unbiased=False)
        input = (input - mean) / torch.sqrt(var + self.eps)
        if self.track_running_stats:

            self.running_mean = self.running_mean * (1. - self.momentum) + mean[None,:,:].detach() * self.momentum
            self.running_var = self.running_var * (1. - self.momentum) + var[None,:,:].detach() * self.momentum
        if self.affine:
            input = input * self.weight + self.bias
        return input

    def forward(self, input,group=True):
        if group:
            return self.normalize(input)
        else:
            return self.normalize_bn(input)

class Small_BatchNormSN(nn.Module):
    def __init__(self,
                 num_group,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(Small_BatchNormSN, self).__init__()

        self.num_group = num_group
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        #G*C normalization weigh and error
        if self.affine:
            self.weight = nn.Parameter(torch.ones([num_features])) #only C dimension keep
            self.bias = nn.Parameter(torch.zeros([num_features]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros([self.num_group,1,num_features]))
            self.register_buffer('running_var', torch.ones([self.num_group,1,num_features]))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

        self.reset_parameters()

    def extra_repr(self):
        return 'num_groups={}, num_features={}, eps={}, momentum={}'.\
            format(self.num_group, self.num_features, self.eps, self.momentum)

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def normalize(self, input):
        """
        input shape
        N*C
        Args:
            input:

        Returns:

        """
        if len(input.shape)==2:
            mode=0
            cur_batch_size, num_features = input.shape
            input = torch.reshape(input,
                                  (self.num_group, cur_batch_size // self.num_group, self.num_features))  # (G,N, C)
        elif len(input.shape)==3:
            mode=1
            cur_batch_size, num_features,height = input.shape
            input = torch.reshape(input,
                                  (self.num_group, cur_batch_size // self.num_group, self.num_features,height))  # (G,N, C,H)
        elif len(input.shape)==4:
            mode=2
            cur_batch_size, num_features, height,width = input.shape
            input = torch.reshape(input,
                                  (self.num_group, cur_batch_size // self.num_group, self.num_features,
                                   height,width))  # (G,N, C,H,W)
        else:
            print("input shape is not suppored: ",input.shape)
            print("only support 2D, 3D, 4D shape tensor for normalization")
            exit()
        if mode==0:
            var, mean = torch.var_mean(input, dim=1, keepdim=True, unbiased=False)
        elif mode==1:
            var, mean = torch.var_mean(input, dim=[1,3], keepdim=True, unbiased=False)
        else:
            var, mean = torch.var_mean(input, dim=[1,3,4], keepdim=True, unbiased=False)
        input = (input - mean) / torch.sqrt(var + self.eps)

        if self.track_running_stats:
            if mode==1:
                mean = mean[:,:,:,0]
                var = var[:,:,:,0]
            elif mode==2:
                mean = mean[:, :, :, 0,0]
                var = var[:, :, :, 0,0]
            self.running_mean = self.running_mean * (1. - self.momentum) + mean.detach() * self.momentum
            self.running_var = self.running_var * (1. - self.momentum) + var.detach() * self.momentum
        if mode==0:
            input = input.view(cur_batch_size,num_features)
        elif mode==1:
            input = input.view(cur_batch_size, num_features,height)
        elif mode==2:
            input = input.view(cur_batch_size, num_features, height,width)

        if self.affine:
            if mode==0:
                input = input * self.weight + self.bias
            elif mode==1:
                input = input * self.weight[None, :, None] + self.bias[None, :, None]
            elif mode==2:
                input = input * self.weight[None, :, None,None] + self.bias[None, :, None,None]

        return input

    def forward(self, input):
        return self.normalize(input)


