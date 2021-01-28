# Stochastic methods for the computation of the trace of the inverse

import numpy as np
import scipy as sp
from solver import solver_sparse
from math import sqrt
import pyamg
from utils import flopsV
from pyamg.aggregation.adaptive import adaptive_sa_solver
from aggregation import manual_aggregation




# compute tr(A^{-1}) via Hutchinson
def hutchinson(A, solver, params):

    # FIXME
    max_nr_levels = 9

    # TODO : check input params !

    # solver params
    solver_name = params['solver_params']['name']
    solver_tol = params['solver_params']['tol']
    #if solver_name=='mg':
    #    ml = params['solver_params']['ml']

    #use_mg = params['solver_params']['use_mg']

    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']

    #np.random.randint(2, size=10)

    # size of the problem
    N = A.shape[0]

    # pre-compute a rough estimate of the trace, to set then a tolerance
    nr_rough_iters = 5
    ests = np.zeros(nr_rough_iters)
    # main Hutchinson loop
    for i in range(nr_rough_iters):
        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1
        if not x.dtype==A.dtype:
            x = x.astype(A.dtype)
        z,num_iters = solver_sparse(A,x,solver_tol,solver,solver_name)
        #print(num_iters)
        e = np.vdot(x,z)
        ests[i] = e.real
    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)

    # then, set a rough tolerance
    rough_trace_tol = trace_tol*rough_trace
    #rough_solver_tol = rough_trace_tol*lambda_min/N
    rough_solver_tol = rough_trace_tol/N

    #print(rough_trace_tol/N)
    #exit(0)

    solver_iters = 0
    ests = np.zeros(trace_max_nr_ests)
    # main Hutchinson loop
    for i in range(trace_max_nr_ests):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1
        if not x.dtype==A.dtype:
            x = x.astype(A.dtype)

        z,num_iters = solver_sparse(A,x,rough_solver_tol,solver,solver_name)
        solver_iters += num_iters
        e = np.vdot(x,z)

        #print(num_iters)

        ests[i] = e.real

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(np.sum(np.square(ests[0:(i+1)]-ests_avg))/(i+1))
        error_est = ests_dev/sqrt(i+1)

        # break condition
        if i>5 and error_est<rough_trace_tol:
            break

    result = dict()
    result['trace'] = ests_avg
    result['std_dev'] = ests_dev
    result['nr_ests'] = i
    result['solver_iters'] = solver_iters
    if solver_name=='mg':
        #result['total_complexity'] = solver.cycle_complexity()*solver_iters
        result['total_complexity'] = flopsV(len(solver.levels), solver.levels, 0)*solver_iters

    return result


# compute tr(A^{-1}) via MLMC
def mlmc(A, solver, params):

    # FIXME
    max_nr_levels = 9

    # TODO : check input params !

    # solver params
    solver_name = params['solver_params']['name']
    solver_tol = params['solver_params']['tol']
    lambda_min = params['solver_params']['lambda_min']

    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']
    trace_ml_constr = params['multilevel_construction']

    # size of the problem
    N = A.shape[0]

    max_nr_levels = params['max_nr_levels']

    print("\nRunning MG setup from finest level ...")
    if trace_ml_constr=='pyamg':
        #ml = pyamg.smoothed_aggregation_solver(A)
        #[ml, work] = adaptive_sa_solver(A, num_candidates=5, improvement_iters=5)
        [ml, work] = adaptive_sa_solver(A, num_candidates=2, candidate_iters=2, improvement_iters=3,
                                        strength='symmetric', aggregate='standard', max_levels=max_nr_levels)
    elif trace_ml_constr=='manual_aggregation':

        # TODO : get <aggr_size> from input params
        aggr_size = 2
        aggrs = [aggr_size for i in range(max_nr_levels-1)]

        dof = [2]
        # TODO : get <dof_size> from input params
        dof_size = 2
        [dof.append(dof_size) for i in range(max_nr_levels-1)]

        ml = manual_aggregation(A, dof=dof, aggrs=aggrs, max_levels=max_nr_levels, dim=2)

    else:
        raise Exception("The specified <trace_multilevel_constructor> does not exist.")
    print("... done")

    exit(0)

    print("\nMultilevel information:")
    print(ml)

    # the actual number of levels
    nr_levels = len(ml.levels)

    #print("\nRunning MG setup for coarser levels ...")
    print("\nRunning MG setup for each level ...")
    ml_solvers = list()
    #ml_solvers.append(ml)
    #if nr_levels>2:
    for i in range(nr_levels-1):
        #mlx = pyamg.smoothed_aggregation_solver(ml.levels[i].A)
        #[mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=5, improvement_iters=5)
        [mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=2, candidate_iters=2, improvement_iters=3,
                                         strength='symmetric', aggregate='standard', max_levels=max_nr_levels-i)
        ml_solvers.append(mlx)
    print("... done")
    print("")

    print("\nComputing rough estimation of the trace ...")
    # first of all, get a rough estimate of the trace via Hutchinson
    trace_params = dict()
    solver_params = dict()
    solver_params['name'] = "mg"
    solver_params['tol'] = solver_tol
    trace_params['solver_params'] = solver_params
    trace_params['tol'] = trace_tol
    trace_params['max_nr_ests'] = 5
    result = hutchinson(A, ml_solvers[0], trace_params)
    trace = result['trace']
    std_dev = result['std_dev']
    nr_ests = result['nr_ests']
    solver_iters = result['solver_iters']
    print("... done \n")

    rough_trace = trace
    print("** rough estimation of the trace : "+str(rough_trace))

    #for i in range(nr_levels):
    #    mat_diff = ml.levels[i].A-ml.levels[i].A.H
    #    mat_diff = mat_diff.todense()
    #    print(np.linalg.norm(mat_diff))

    output_params = dict()
    output_params['nr_levels'] = nr_levels
    output_params['trace'] = 0.0
    output_params['total_complexity'] = 0.0
    output_params['std_dev'] = 0.0
    output_params['results'] = list()
    # setting to zero the counters and results to be returned
    for i in range(nr_levels):
        output_params['results'].append(dict())
        output_params['results'][i]['solver_iters'] = 0
        output_params['results'][i]['nr_ests'] = 0
        output_params['results'][i]['ests_avg'] = 0.0
        output_params['results'][i]['ests_dev'] = 0.0
        output_params['results'][i]['level_complexity'] = 0.0

    # compute level differences
    level_trace_tol  = trace_tol*rough_trace/sqrt(nr_levels)
    level_solver_tol = level_trace_tol/N

    #print(level_solver_tol)

    #print(trace_tol)
    #print(level_trace_tol)

    cummP = sp.sparse.identity(N)
    cummR = sp.sparse.identity(N)
    for i in range(nr_levels-1):

        # fine and coarse matrices
        Af = ml.levels[i].A
        Ac = ml.levels[i+1].A
        # P and R
        R = ml.levels[i].R
        P = ml.levels[i].P

        print("Computing for level "+str(i)+"...")

        ests = np.zeros(trace_max_nr_ests)
        for j in range(trace_max_nr_ests):

            # generate a Rademacher vector
            x0 = np.random.randint(2, size=N)
            x0 *= 2
            x0 -= 1
            if not x0.dtype==Af.dtype:
                x0 = x0.astype(Af.dtype)

            x = cummR*x0

            #z,num_iters = solver_sparse(Af,x,level_solver_tol,solver,"cg")
            z,num_iters = solver_sparse(Af,x,level_solver_tol,ml_solvers[i],"mg")
            output_params['results'][i]['solver_iters'] += num_iters

            #print(num_iters)

            xc = R*x

            # for the last level, there is no MG
            #y,num_iters = solver_sparse(Ac,xc,level_solver_tol,solver,"cg")
            if (i+1)==(nr_levels-1):
                y,num_iters = solver_sparse(Ac,xc,level_solver_tol,solver,"cg")
            else:
                y,num_iters = solver_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],"mg")
            output_params['results'][i+1]['solver_iters'] += num_iters

            e1 = np.vdot(x0,cummP*z)
            e2 = np.vdot(x0,cummP*P*y)

            ests[j] = e1.real-e2.real

            # average of estimates
            ests_avg = np.sum(ests[0:(j+1)])/(j+1)
            # and standard deviation
            ests_dev = sqrt(np.sum(np.square(ests[0:(j+1)]-ests_avg))/(j+1))
            error_est = ests_dev/sqrt(j+1)

            # break condition
            if j>5 and error_est<level_trace_tol:
                break

        # cummulative R and P
        cummP1 = cummP*P
        cummP = sp.sparse.csr_matrix.copy(cummP1)
        cummR1 = R*cummR
        cummR = sp.sparse.csr_matrix.copy(cummR1)

        output_params['results'][i]['nr_ests'] += j

        # set trace and standard deviation
        output_params['results'][i]['ests_avg'] = ests_avg
        output_params['results'][i]['ests_dev'] = ests_dev

        print("... done")

    # compute now at the coarsest level

    Ac = ml.levels[nr_levels-1].A
    Nc = Ac.shape[0]
    ests = np.zeros(trace_max_nr_ests)
    for i in range(trace_max_nr_ests):

        # generate a Rademacher vector
        xc = np.random.randint(2, size=Nc)
        xc *= 2
        xc -= 1
        if not xc.dtype==Ac.dtype:
            xc = xc.astype(Ac.dtype)

        x1 = cummP*xc
        x2 = cummR*x1

        y,num_iters = solver_sparse(Ac,x2,level_solver_tol,solver,"cg")
        output_params['results'][nr_levels-1]['solver_iters'] += num_iters

        ests[i] = np.vdot(xc,y).real

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(np.sum(np.square(ests[0:(i+1)]-ests_avg))/(i+1))
        error_est = ests_dev/sqrt(i+1)

        # break condition
        if i>5 and error_est<level_trace_tol:
            break

    output_params['results'][nr_levels-1]['nr_ests'] += i

    # set trace and standard deviation
    output_params['results'][nr_levels-1]['ests_avg'] = ests_avg
    output_params['results'][nr_levels-1]['ests_dev'] = ests_dev

    for i in range(nr_levels-1):
        #output_params['results'][i]['level_complexity'] += output_params['results'][i]['solver_iters']*ml_solvers[i].cycle_complexity()
        slvr = ml_solvers[i]
        output_params['results'][i]['level_complexity'] += output_params['results'][i]['solver_iters']*flopsV(len(slvr.levels), slvr.levels, 0)

    for i in range(nr_levels-1):
        output_params['total_complexity'] += output_params['results'][i]['level_complexity']

    #print(np.trace(np.linalg.inv(Ac.todense())))

    # total trace
    for i in range(nr_levels):
        output_params['trace'] += output_params['results'][i]['ests_avg']

    print("")

    #return (ests_avg,ests_dev,i,solver_iters)
    return output_params
