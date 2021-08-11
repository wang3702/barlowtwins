import os
from ops.os_operation import mkdir
def init_log_path(args):
    """
    :param args:
    :return:
    save model+log path
    """
    if args.save_path is None:
        save_path = os.path.join(os.getcwd(), "MaskCo")
        mkdir(save_path)
    else:
        save_path = args.save_path
        mkdir(args.save_path)
    save_path = os.path.join(save_path, args.dataset)
    mkdir(save_path)
    save_path = os.path.join(save_path, "mode_" + str(args.mode))
    mkdir(save_path)
    save_path = os.path.join(save_path, "Type_" + str(args.type))
    mkdir(save_path)
    save_path = os.path.join(save_path, "lr_" + str(args.lr))
    mkdir(save_path)
    save_path = os.path.join(save_path, "cos_" + str(args.cos))
    mkdir(save_path)
    save_path = os.path.join(save_path, "t_" + str(args.moco_t) + "_localt_" + str(args.local_t))
    mkdir(save_path)
    save_path = os.path.join(save_path, "nmb_crops_" + str(args.nmb_crops))
    mkdir(save_path)
    save_path = os.path.join(save_path, "size_crops_" + str(args.size_crops))
    mkdir(save_path)
    save_path = os.path.join(save_path, "min_scale_crops_" + str(args.min_scale_crops))
    mkdir(save_path)
    save_path = os.path.join(save_path, "max_scale_crops_" + str(args.max_scale_crops))
    mkdir(save_path)
    save_path = os.path.join(save_path, "masksize_" + str(args.mask_size))
    mkdir(save_path)
    save_path = os.path.join(save_path, "shiftratio_" + str(args.shift_ratio))
    mkdir(save_path)
    save_path = os.path.join(save_path, "num_roi_" + str(args.num_roi))
    mkdir(save_path)
    save_path = os.path.join(save_path, "alpha_" + str(args.alpha))
    mkdir(save_path)
    if args.img_size!=224:
        save_path = os.path.join(save_path, "Img_" + str(args.img_size))
        mkdir(save_path)
    #if args.sample_ratio !=32:
    #    save_path = os.path.join(save_path, "sampleratio_" + str(args.sample_ratio))
    #    mkdir(save_path)
    if args.moco_dim !=128:
        save_path = os.path.join(save_path, "mlpdim_" + str(args.moco_dim))
        mkdir(save_path)
    if args.align != 0:
        save_path = os.path.join(save_path,"align"+str(args.align))
        mkdir(save_path)
    if args.epochs!=100:
        save_path = os.path.join(save_path,"epoch_"+str(args.epochs))
        mkdir(save_path)
    if args.pred_dim !=512:
        save_path = os.path.join(save_path, "preddim_" + str(args.pred_dim))
        mkdir(save_path)
    if args.shuffle_mode !=0:
        save_path = os.path.join(save_path, "shuffle_" + str(args.shuffle_mode))
        mkdir(save_path)
    if args.mlp_bn_stat==0:
        save_path = os.path.join(save_path, "mlpbnstat_" + str(args.mlp_bn_stat))
        mkdir(save_path)
    if args.lars !=0:
        save_path = os.path.join(save_path, "lars_" + str(args.lars))
        mkdir(save_path)
    if args.group_norm_size!=8:
        save_path = os.path.join(save_path, "groupnormsize_" + str(args.group_norm_size))
        mkdir(save_path)
    if args.use_fp16:
        save_path = os.path.join(save_path, "fp16_b" + str(args.batch_size))
        mkdir(save_path)
    if args.momentum_stat!=0.999:
        save_path = os.path.join(save_path, "momentumstat_" + str(args.momentum_stat))
        mkdir(save_path)
    if args.loco_conv_size!=1 or args.loco_conv_stride!=1:
        save_path = os.path.join(save_path, "convloco_" + str(args.loco_conv_size)+"_stride_"+str(args.loco_conv_stride))
        mkdir(save_path)
    if args.crop_min!=0.2:
        save_path = os.path.join(save_path,"cropmin_" + str(args.crop_min) )
        mkdir(save_path)
    return save_path