
def convert_model_to_group(world_size ,group_norm_size ,model):
    total_world_size = world_size
    print("total world size %d, group num %d" % (total_world_size, group_norm_size))
    if total_world_size >= group_norm_size:
        cur_divide_group = 1
        gpu_per_group = total_world_size // group_norm_size
    else:
        gpu_per_group = 1
        cur_divide_group = group_norm_size // total_world_size

    print("groupBN %d gpu per group" % gpu_per_group)
    print("per gpu divided into %d groups" % cur_divide_group)
    import apex
    if cur_divide_group > 1:
        from ops.convert_syncbn_model import convert_groupbn_model
        model = convert_groupbn_model(model, cur_divide_group)
    else:
        process_group = apex.parallel.create_syncbn_process_group(gpu_per_group)
        print("current process group:", process_group)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    return model