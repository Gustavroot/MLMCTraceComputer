# Some extra utils functions



def flopsV(nr_levels, levels_info, level_id):
    if nr_levels == 1:
        return levels_info[0].A.nnz
    elif level_id == nr_levels-2:
        return 2 * levels_info[level_id].A.nnz + levels_info[level_id+1].A.nnz
    else:
        return 2*levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)
