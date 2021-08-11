import parser
import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def argparser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default="data", type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--log_path', type=str, default="train_log", help="log path for saving models")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        # choices=model_names,
                        type=str,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_final', default=0.0006, type=float,
                        help='final learning rate')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to reloading encoder checkpoint (default: none)')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training,args.nodes_num*args.ngpu,here we specify with the number of nodes')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training,rank of total threads, 0 to args.world_size-1')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', type=int, default=1,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument("--mlp_dim",default=2048, type=int,help="hidden fc dim in projector/predictor")
    parser.add_argument("--pred_dim",default=512, type=int, help="predictor dimension of architecture")
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.12, type=float,
                        help='softmax temperature (default: 0.12)')
    parser.add_argument("--local_t",default = 0.2, type=float,help="softmax temperature for the local regions")
    parser.add_argument('--moco_k', default=65536, type=int,
                        help='Queue size (default: 65536)')

    # options for moco v2
    parser.add_argument('--mlp', type=int, default=1,
                        help='use mlp head')
    parser.add_argument('--cos', type=int, default=1,
                        help='use cosine lr schedule')
    parser.add_argument('--dataset', type=str, default="ImageNet", help="Specify dataset: ImageNet or cifar10")
    parser.add_argument('--choose', type=str, default=None, help="choose gpu for training")
    parser.add_argument('--save_path', default=None, type=str, help="model and record save path")
    # idea from swav#adds crops for it
    parser.add_argument("--nmb_crops", type=int, default=[1, 1, 1, 1, 1], nargs="+",
                        help="list of number of crops (example: [2, 6])")
    parser.add_argument("--size_crops", type=int, default=[224, 192, 160, 128, 96], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.2, 0.172, 0.143, 0.114, 0.086], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1.0, 0.86, 0.715, 0.571, 0.429], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")
    parser.add_argument('--cluster', type=int, default=65536, help="number of learnable comparison features")
    parser.add_argument('--memory_lr', type=float, default=3, help="learning rate for adversial memory bank")
    parser.add_argument("--ad_init", type=int, default=1, help="use feature encoding to init or not")
    parser.add_argument("--nodes_num", type=int, default=1, help="number of nodes to use")
    parser.add_argument("--ngpu", type=int, default=8, help="number of gpus per node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="addr for master node")
    parser.add_argument("--master_port", type=str, default="1234", help="port for master node")
    parser.add_argument('--node_rank', type=int, default=0, help='rank of machine, 0 to nodes_num-1')

    parser.add_argument('--mem_t', default=0.02, type=float,
                        help='temperature for memory bank(default: 0.07)')
    parser.add_argument('--mem_wd', default=1e-4, type=float,
                        help='weight decay of memory bank (default: 0)')
    parser.add_argument("--sym", type=int, default=0, help="train with symmetric loss or not")
    parser.add_argument("--multi_crop", type=int, default=0, help="train with multi crop")
    parser.add_argument("--mode",type=int,default=0,help="control mode for training")
    parser.add_argument("--img_size",type=int,default=224,help="img size for resize")
    parser.add_argument("--mask_size", type=int, default=32, help="mask size for images")
    parser.add_argument("--knn_freq",type=int,default=10, help="report current accuracy under specific iterations")
    parser.add_argument("--knn_batch_size",type=int, default=128, help="default batch size for knn eval")
    parser.add_argument("--knn_neighbor",type=int,default=200,help="nearest neighbor used to decide the labels")
    parser.add_argument("--tensorboard",type=int,default=0,help="use tensorboard or not")
    parser.add_argument("--type",type=int,default=0,help="running type control")
    parser.add_argument('--shift_ratio',type=float,default=0.5,help="shift ratio for the one and another one ")
    parser.add_argument("--num_roi",type=int,default=20,help="number of rois applied for one image")
    parser.add_argument("--slurm",type=int,default=0, help = "specify use slurm submitting job or not")
    parser.add_argument("--warmup_epochs",type=int,default=10, help="number of epochs for warm up")
    parser.add_argument("--warmup_lr",type=float, default=0, help="warmup init lr")
    #swav parameter
    parser.add_argument("--queue_length", type=int, default=0,
                        help="length of the queue (0 for no queue)")
    parser.add_argument("--epoch_queue_starts", type=int, default=15,
                        help="from this epoch, we start using a queue")
    parser.add_argument("--dump_path", type=str, default="swav_dump_path",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                        help="list of crops id used for computing assignments")
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                        help="freeze the prototypes during this many iterations from the start")

    parser.add_argument("--alpha",default=1, type=float,help="balance coefficient with the global and local learning")
    parser.add_argument("--sample_ratio",default=32, type=int, help="sampling ratio for the roi align to make sure we match well.")
    parser.add_argument("--align",default=1, type=int,help = "align or not in the crops")
    parser.add_argument("--shuffle_mode",default=0, type=int,help = "shuffle mode for shuffle bn")
    parser.add_argument("--mlp_bn_stat",default=1,type=int,help="for mlp bn, this is module tracks the running mean and variance, and when set to False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None")
    parser.add_argument("--lars",default=0,type=int, help="use lars optimizer or not")
    parser.add_argument("--group_norm_size",default=8, type=int,help="group norm size to normalize")
    parser.add_argument("--use_fp16", type=int, default=0,
                        help="whether to train with mixed precision or not")
    parser.add_argument("--momentum_stat",type=float,default=0.999,help="momentum hyper param for BN stat")
    parser.add_argument("--loco_conv_size",type=int,default=1,help="the size of conv filter in loco module")
    parser.add_argument("--loco_conv_stride",type=int,default=1,help="the stride of conv filter in loco module")
    parser.add_argument('--stop_grad_conv1', type=int,default=1,
                        help='stop-grad after first conv, or patch embedding')
    parser.add_argument("--crop_min",type=float,default=0.2,help="crop min scale in randomresized crop")
    parser.add_argument("--key_group",type=int,default=1,help="key grouping into different stats")
    parser.add_argument('--learning_rate_biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--learning_rate_weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    return parser