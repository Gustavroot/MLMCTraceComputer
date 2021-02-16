# Stochastic methods for the computation of the trace of the inverse

import numpy as np
import scipy as sp
from solver import solver_sparse
from math import sqrt
import pyamg
from utils import flopsV
from pyamg.aggregation.adaptive import adaptive_sa_solver
from aggregation import manual_aggregation
from scipy.sparse import csr_matrix
from numpy.linalg import norm as npnorm
from scipy.sparse.linalg import norm
import png
from numpy.linalg import eigh




class LevelML:
    R = 0
    P = 0
    A = 0
    Q = 0

class SimpleML:
    levels = []

    def __str__(self):
        return "For manual aggregation, printing <ml> is under construction"



def gamma3_application(v):
    v_size = int(v.shape[0]/2)
    v[v_size:] = -v[v_size:]
    return v


def gamma5_application(v,l):

    dof = [12,48]

    sz = v.shape[0]

    for i in range(int(sz/dof[l])):
        # negate first half
        for j in range(int(dof[l]/2)):
            v[i*dof[l]+j] = -v[i*dof[l]+j]

    return v


# https://stackoverflow.com/questions/33713221/create-png-image-from-sparse-data
def write_png(A, filename):
    m, n = A.shape

    w = png.Writer(n, m, greyscale=True, bitdepth=1)

    class RowIterator:
        def __init__(self, A):
            self.A = A.tocsr()
            self.current = 0
            return

        def __iter__(self):
            return self

        def __next__(self):
            if self.current+1 > A.shape[0]:
                raise StopIteration
            out = np.ones(A.shape[1], dtype=bool)
            out[self.A[self.current].indices] = False
            self.current += 1
            return out

    with open(filename, 'wb') as f:
        w.write(f, RowIterator(A))

    return



# compute tr(A^{-1}) via Hutchinson
def hutchinson(A, solver, params):

    # TODO : check input params !

    max_nr_levels = params['max_nr_levels']

    # solver params
    solver_name = params['solver_params']['name']
    solver_tol = params['solver_params']['tol']
    #if solver_name=='mg':
    #    ml = params['solver_params']['ml']

    #use_mg = params['solver_params']['use_mg']

    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']

    use_Q = params['use_Q']

    # size of the problem
    N = A.shape[0]

    solver_tol = 1e-9

    np.random.seed(123456)

    # pre-compute a rough estimate of the trace, to set then a tolerance
    nr_rough_iters = 10
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)
    # main Hutchinson loop
    for i in range(nr_rough_iters):

        # generate a Rademacher vector
        x = np.random.randint(4, size=N)
        x *= 2
        x -= 1
        x = np.where(x==3, -1j, x)
        x = np.where(x==5, 1j, x)
        x = x.astype(A.dtype)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0)

        z,num_iters = solver_sparse(A,x,solver_tol,solver,solver_name)
        #print(num_iters)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0)

        e = np.vdot(x,z)
        ests[i] = e

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)

    print("** rough estimation of the trace : "+str(rough_trace))
    print("")

    # then, set a rough tolerance
    rough_trace_tol = abs(trace_tol*rough_trace)
    #rough_solver_tol = rough_trace_tol*lambda_min/N
    rough_solver_tol = abs(rough_trace_tol/N)

    rough_solver_tol = 1e-9

    solver_iters = 0
    ests = np.zeros(trace_max_nr_ests, dtype=A.dtype)
    # main Hutchinson loop
    for i in range(trace_max_nr_ests):

        # generate a Rademacher vector
        x = np.random.randint(4, size=N)
        x *= 2
        x -= 1
        x = np.where(x==3, -1j, x)
        x = np.where(x==5, 1j, x)
        x = x.astype(A.dtype)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0)

        z,num_iters = solver_sparse(A,x,rough_solver_tol,solver,solver_name)
        #print(num_iters)

        solver_iters += num_iters

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0)

        e = np.vdot(x,z)

        ests[i] = e

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(i+1)]-ests_avg)))/(i+1))
        error_est = ests_dev/sqrt(i+1)

        #print(str(i)+" .. "+str(ests_avg)+" .. "+str(rough_trace)+" .. "+str(error_est)+" .. "+str(rough_trace_tol)+" .. "+str(num_iters))

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

    # TODO : check input params !

    # solver params
    solver_name = params['solver_params']['name']
    solver_tol = params['solver_params']['tol']
    lambda_min = params['solver_params']['lambda_min']

    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']
    trace_ml_constr = params['multilevel_construction']

    use_Q = params['use_Q']

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
        #aggr_size = 4
        #aggrs = [aggr_size for i in range(max_nr_levels-1)]

        aggrs = [4,4,4,2]
        #aggrs = [4,4]

        #dof = [2]
        # TODO : get <dof_size> from input params
        #dof_size = 8
        #[dof.append(dof_size) for i in range(max_nr_levels-1)]

        dof = [2,4,4,16,32]
        #dof = [2,2,2]

        # (128x128)x2 ---> (32x32)x(4x2) ---> (8x8)x(4x2) ---> (2x2)x(16x2) ---> (1x1)x(32x2)

        ml = manual_aggregation(A, dof=dof, aggrs=aggrs, max_levels=max_nr_levels, dim=2)
    elif trace_ml_constr=='from_files':

        import scipy.io as sio

        ml = SimpleML()

        # load A at each level
        for i in range(max_nr_levels):
            ml.levels.append(LevelML())
            mat_contents = sio.loadmat('LQCD_A'+str(i+1)+'.mat')
            Axx = mat_contents['A'+str(i+1)]
            ml.levels[i].A = Axx.copy()

        # load Q at each level
        #for i in range(max_nr_levels):
        #    mat_contents = sio.loadmat('LQCD_Q'+str(i+1)+'.mat')
        #    Qxx = mat_contents['Q'+str(i+1)]
        #    ml.levels[i].Q = Qxx.copy()

        # load P at each level
        for i in range(max_nr_levels-1):
            mat_contents = sio.loadmat('LQCD_P'+str(i+1)+'.mat')
            Pxx = mat_contents['P'+str(i+1)]
            ml.levels[i].P = Pxx.copy()
            # construct R from P
            Rxx = Pxx.copy()
            Rxx = Rxx.conjugate()
            Rxx = Rxx.transpose()
            ml.levels[i].R = Rxx.copy()

    else:
        raise Exception("The specified <trace_multilevel_constructor> does not exist.")
    print("... done")

    print("\nMultilevel information:")
    print(ml)

    # the actual number of levels
    nr_levels = len(ml.levels)

    for i in range(nr_levels):
        print("size(A"+str(i)+") = "+str(ml.levels[i].A.shape[0])+"x"+str(ml.levels[i].A.shape[1]))

    print("\nRunning MG setup for each level ...")
    ml_solvers = list()
    #for i in range(nr_levels-1):
    for i in range(nr_levels-1):
        mlx = pyamg.smoothed_aggregation_solver(ml.levels[i].A)
        #[mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=5, improvement_iters=5)
        #[mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=2, candidate_iters=2, improvement_iters=3,
        #                                 strength='symmetric', aggregate='standard', max_levels=max_nr_levels-i)
        #[mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=2, candidate_iters=2, improvement_iters=3,
        #                                 strength='symmetric', aggregate='standard', max_levels=9)
        ml_solvers.append(mlx)
    print("... done")

    solver_tol = 1e-9

    print("\nComputing rough estimation of the trace ...")
    np.random.seed(123456)
    # pre-compute a rough estimate of the trace, to set then a tolerance
    nr_rough_iters = 10
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)
    # main Hutchinson loop
    for i in range(nr_rough_iters):
        # generate a Rademacher vector
        x = np.random.randint(4, size=N)
        x *= 2
        x -= 1
        x = np.where(x==3, -1j, x)
        x = np.where(x==5, 1j, x)
        x = x.astype(A.dtype)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0)

        z,num_iters = solver_sparse(A,x,solver_tol,ml_solvers[0],"mg")

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0)

        e = np.vdot(x,z)
        ests[i] = e

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)
    print("... done \n")

    print("** rough estimation of the trace : "+str(rough_trace))

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
    level_trace_tol  = abs(trace_tol*rough_trace/sqrt(nr_levels))
    level_solver_tol = level_trace_tol/N

    #level_solver_tol = 1e-9/sqrt(nr_levels)
    level_solver_tol = 1e-9

    print("")

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

        ests = np.zeros(trace_max_nr_ests, dtype=Af.dtype)
        for j in range(trace_max_nr_ests):

            # generate a Rademacher vector
            x0 = np.random.randint(4, size=N)
            x0 *= 2
            x0 -= 1
            x0 = np.where(x0==3, -1j, x0)
            x0 = np.where(x0==5, 1j, x0)
            x0 = x0.astype(A.dtype)

            x = cummR*x0

            if use_Q:
                if params['problem_name']=='schwinger':
                    x = gamma3_application(x)
                elif params['problem_name']=='LQCD':
                    x = gamma5_application(x,i)

            z,num_iters = solver_sparse(Af,x,level_solver_tol,ml_solvers[i],"mg")
            #print(num_iters)

            output_params['results'][i]['solver_iters'] += num_iters

            if use_Q:
                if params['problem_name']=='schwinger':
                    x = gamma3_application(x)
                elif params['problem_name']=='LQCD':
                    x = gamma5_application(x,i)

            xc = R*x

            if use_Q:
                if params['problem_name']=='schwinger':
                    xc = gamma3_application(xc)
                elif params['problem_name']=='LQCD':
                    xc = gamma5_application(xc,i+1)

            # for the last level, there is no MG
            #y,num_iters = solver_sparse(Ac,xc,level_solver_tol,solver,"cg")

            if (i+1)==(nr_levels-1):
                #y,num_iters = solver_sparse(Ac,xc,level_solver_tol,solver,"cg")
                # solve directly
                np_Ac = Ac.todense()

                #w,v = eigh(np_Ac)
                #print(w)
                #exit(0)

                y = np.dot(np.linalg.inv(np_Ac),xc)
                y = np.asarray(y).reshape(-1)
                num_iters = 1
            else:
                y,num_iters = solver_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],"mg")
            #y,num_iters = solver_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],"mg")

            #print(num_iters)

            output_params['results'][i+1]['solver_iters'] += num_iters

            e1 = np.vdot(x0,cummP*z)
            e2 = np.vdot(x0,cummP*P*y)

            #print(e1)
            #print(e2)

            ests[j] = e1-e2

            # average of estimates
            ests_avg = np.sum(ests[0:(j+1)])/(j+1)
            # and standard deviation
            ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(j+1)]-ests_avg)))/(j+1))
            error_est = ests_dev/sqrt(j+1)

            #print(str(j)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol))

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
    ests = np.zeros(trace_max_nr_ests, dtype=Ac.dtype)

    np_Ac = Ac.todense()
    np_Ac_inv = np.linalg.inv(np_Ac)

    for i in range(trace_max_nr_ests):

        # generate a Rademacher vector
        xc = np.random.randint(4, size=Nc)
        xc *= 2
        xc -= 1
        xc = np.where(xc==3, -1j, xc)
        xc = np.where(xc==5, 1j, xc)
        xc = xc.astype(A.dtype)

        x1 = cummP*xc
        x2 = cummR*x1

        if use_Q:
            if params['problem_name']=='schwinger':
                x2 = gamma3_application(x2)
            elif params['problem_name']=='LQCD':
                x2 = gamma5_application(x2,nr_levels-1)

        #y,num_iters = solver_sparse(Ac,x2,level_solver_tol,ml_solvers[nr_levels-1],"mg")
        y = np.dot(np_Ac_inv,x2)
        y = np.asarray(y).reshape(-1)
        num_iters = 1

        #print(num_iters)

        output_params['results'][nr_levels-1]['solver_iters'] += num_iters

        ests[i] = np.vdot(xc,y)

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(i+1)]-ests_avg)))/(i+1))
        error_est = ests_dev/sqrt(i+1)

        #print(str(i)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol))

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
        output_params['results'][i]['level_complexity'] = output_params['results'][i]['solver_iters']*flopsV(len(slvr.levels), slvr.levels, 0)

    output_params['results'][nr_levels-1]['level_complexity'] = output_params['results'][i]['solver_iters']*(ml.levels[nr_levels-1].A.shape[0]*ml.levels[nr_levels-1].A.shape[0])

    for i in range(nr_levels):
        output_params['total_complexity'] += output_params['results'][i]['level_complexity']

    # total trace
    for i in range(nr_levels):
        output_params['trace'] += output_params['results'][i]['ests_avg']

    print("")

    return output_params
