# Some extra utils functions


import multigrid as mg

def flopsV(nr_levels, levels_info, level_id):
    #if nr_levels == 1:
    #    return levels_info[0].A.nnz
    #elif level_id == nr_levels-2:
    #    return 2 * levels_info[level_id].A.nnz + levels_info[level_id+1].A.nnz
    #else:
    #    return 2*levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)

    # adds the Krylov that wraps the AMG solver
    if nr_levels == 1:
        return levels_info[0].A.nnz
    elif level_id == nr_levels-2:
        return 2 * levels_info[level_id].A.nnz + levels_info[level_id+1].A.nnz
    else:
        if level_id==0:
            return (2+1) * levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)
        else:
            return 2 * levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)

# adds the Krylov that wraps the AMG solver
def flopsV_manual(nr_levels, levels_info, level_id):
    if nr_levels == 1:
        return mg.coarsest_iters_avg * levels_info[0].A.nnz
    elif level_id == nr_levels-2:
        return 2 * mg.smoother_iters * levels_info[level_id].A.nnz + levels_info[level_id+1].A.nnz
    else:
        if level_id==0:
            return (2+1) * mg.smoother_iters * levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)
        else:
            return 2 * mg.smoother_iters * levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)
