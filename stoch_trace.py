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
import png





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

    #print("Computing rough estimation of the trace ...")

    #print(solver_tol)
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

        # FIXME : building for Q i.e. applying g3
        x_size = int(x.shape[0]/2)
        x[x_size:] = -x[x_size:]

        #z,num_iters = solver_sparse(A,x,solver_tol,solver,solver_name)
        Abb = A.todense()
        z = np.dot(np.linalg.inv(Abb),x)
        z = np.asarray(z).reshape(-1)
        #print(num_iters)

        #print(num_iters)

        # FIXME : reversing back the application of g3
        x_size = int(x.shape[0]/2)
        x[x_size:] = -x[x_size:]

        e = np.vdot(x,z)
        ests[i] = e

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)

    # FIXME
    #rough_trace = -10

    #print("... done \n")

    print("** rough estimation of the trace : "+str(rough_trace))

    rough_trace = rough_trace.real
    print("** rough estimation of the trace : "+str(rough_trace))

    #exit(0)

    #rough_trace = 5.0

    # then, set a rough tolerance
    rough_trace_tol = abs(trace_tol*rough_trace)
    #rough_solver_tol = rough_trace_tol*lambda_min/N
    rough_solver_tol = abs(rough_trace_tol/N)

    #print(rough_trace_tol/N)
    #exit(0)

    #rough_solver_tol *= 0.50e+2
    rough_solver_tol = 1e-15
    #print(rough_solver_tol)

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

        # FIXME : building for Q i.e. applying g3
        x_size = int(x.shape[0]/2)
        x[x_size:] = -x[x_size:]

        #z,num_iters = solver_sparse(A,x,rough_solver_tol,solver,solver_name)
        Abb = A.todense()
        z = np.dot(np.linalg.inv(Abb),x)
        z = np.asarray(z).reshape(-1)
        num_iters = 0

        solver_iters += num_iters

        # FIXME : reversing back the application of g3
        x_size = int(x.shape[0]/2)
        x[x_size:] = -x[x_size:]

        e = np.vdot(x,z)

        #print(num_iters)
        #print(e)

        ests[i] = e

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(i+1)]-ests_avg)))/(i+1))
        error_est = ests_dev/sqrt(i+1)

        print(str(i)+" .. "+str(ests_avg)+" .. "+str(rough_trace)+" .. "+str(error_est)+" .. "+str(rough_trace_tol)+" .. "+str(num_iters))

        # break condition
        if i>11 and error_est<rough_trace_tol:
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

    #max_nr_levels = params['max_nr_levels']
    max_nr_levels = 2

    print("\nRunning MG setup from finest level ...")
    if trace_ml_constr=='pyamg':
        #ml = pyamg.smoothed_aggregation_solver(A)
        #[ml, work] = adaptive_sa_solver(A, num_candidates=5, improvement_iters=5)
        [ml, work] = adaptive_sa_solver(A, num_candidates=2, candidate_iters=2, improvement_iters=3,
                                        strength='symmetric', aggregate='standard', max_levels=max_nr_levels)
    elif trace_ml_constr=='manual_aggregation':

        # TODO : get <aggr_size> from input params
        aggr_size = 4
        aggrs = [aggr_size for i in range(max_nr_levels-1)]

        dof = [2]
        # TODO : get <dof_size> from input params
        dof_size = 4
        [dof.append(dof_size) for i in range(max_nr_levels-1)]

        ml = manual_aggregation(A, dof=dof, aggrs=aggrs, max_levels=max_nr_levels, dim=2)

    else:
        raise Exception("The specified <trace_multilevel_constructor> does not exist.")
    print("... done")

    print("\nMultilevel information:")
    print(ml)

    # the actual number of levels
    nr_levels = len(ml.levels)

    #exit(0)

    #print("\nRunning MG setup for coarser levels ...")
    print("\nRunning MG setup for each level ...")
    ml_solvers = list()
    #ml_solvers.append(ml)
    #if nr_levels>2:
    for i in range(nr_levels):
        #mlx = pyamg.smoothed_aggregation_solver(ml.levels[i].A)
        #[mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=5, improvement_iters=5)
        #[mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=2, candidate_iters=2, improvement_iters=3,
        #                                 strength='symmetric', aggregate='standard', max_levels=max_nr_levels-i)
        [mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=2, candidate_iters=2, improvement_iters=3,
                                         strength='symmetric', aggregate='standard', max_levels=9)
        ml_solvers.append(mlx)
    print("... done")

    print("")

    for i in range(nr_levels):
        for j in range(len(ml_solvers[i].levels)-1):
            #write_png(csr_matrix(ml_solvers[i].levels[j].P,dtype=ml_solvers[i].levels[j].P.dtype),"P_"+str(i)+"_"+str(j)+".png")
            Pxx1 = np.copy(np.asmatrix(np.asarray(csr_matrix(ml_solvers[i].levels[j].P,dtype=ml_solvers[i].levels[j].P.dtype))))
            Pxx1 = Pxx1[0,0].todense()

            mat_size1_half = int(Pxx1.shape[0]/2)

            Pxx1[mat_size1_half:,:] = -Pxx1[mat_size1_half:,:]

            Pxx2 = np.copy(np.asmatrix(np.asarray(csr_matrix(ml_solvers[i].levels[j].P,dtype=ml_solvers[i].levels[j].P.dtype))))
            Pxx2 = Pxx2[0,0].todense()

            mat_size2_half = int(Pxx2.shape[1]/2)
            Pxx2[:,mat_size2_half:] = -Pxx2[:,mat_size2_half:]
            diffP = Pxx1-Pxx2
            print("measuring g3-compatibility at level "+str(i)+" : "+str( npnorm(diffP,ord='fro') ))

    print("")

    for j in range(nr_levels-1):
        #write_png(csr_matrix(ml_solvers[i].levels[j].P,dtype=ml_solvers[i].levels[j].P.dtype),"P_"+str(i)+"_"+str(j)+".png")
        Pxx1 = np.copy(np.asmatrix(np.asarray(csr_matrix(ml.levels[j].P,dtype=ml.levels[j].P.dtype))))
        Pxx1 = Pxx1[0,0].todense()

        mat_size1_half = int(Pxx1.shape[0]/2)

        Pxx1[mat_size1_half:,:] = -Pxx1[mat_size1_half:,:]

        Pxx2 = np.copy(np.asmatrix(np.asarray(csr_matrix(ml.levels[j].P,dtype=ml.levels[j].P.dtype))))
        Pxx2 = Pxx2[0,0].todense()

        mat_size2_half = int(Pxx2.shape[1]/2)
        Pxx2[:,mat_size2_half:] = -Pxx2[:,mat_size2_half:]
        diffP = Pxx1-Pxx2
        print("measuring g3-compatibility at level "+str(j)+" : "+str( npnorm(diffP,ord='fro') ))

    print("\nComputing rough estimation of the trace ...")
    # first of all, get a rough estimate of the trace via Hutchinson
    trace_params = dict()
    solver_params = dict()
    solver_params['name'] = "mg"
    solver_params['tol'] = solver_tol
    trace_params['solver_params'] = solver_params
    trace_params['tol'] = trace_tol
    trace_params['max_nr_ests'] = 10
    result = hutchinson(A, ml_solvers[0], trace_params)
    trace = result['trace']
    std_dev = result['std_dev']
    nr_ests = result['nr_ests']
    solver_iters = result['solver_iters']
    print("... done \n")

    rough_trace = trace

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

        # FIXME : building for Q i.e. applying g3
        x_size = int(x.shape[0]/2)
        x[x_size:] = -x[x_size:]

        #z,num_iters = solver_sparse(A,x,solver_tol,solver,solver_name)
        Abb = A.todense()
        z = np.dot(np.linalg.inv(Abb),x)
        z = np.asarray(z).reshape(-1)
        #print(num_iters)

        #print(num_iters)

        # FIXME : reversing back the application of g3
        x_size = int(x.shape[0]/2)
        x[x_size:] = -x[x_size:]

        e = np.vdot(x,z)
        ests[i] = e

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)

    # FIXME
    #rough_trace = -10

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
    level_trace_tol  = abs(trace_tol*rough_trace/sqrt(nr_levels))
    level_solver_tol = level_trace_tol/N

    level_solver_tol = 1e-15

    #level_solver_tol *= 0.50e+2

    print("")

    #print(level_solver_tol)

    #print(trace_tol)
    #print(level_trace_tol)

    #print("")

    #for i in range(nr_levels):
    #    print(ml.levels[i].A.shape)

    #print("")













    #Af = ml.levels[0].A
    #Pf = ml.levels[0].P
    #Ac = ml.levels[1].A

    #import scipy.io

    #scipy.io.savemat('Af.mat', mdict={'Af': Af})
    #scipy.io.savemat('Ac.mat', mdict={'Ac': Ac})
    #scipy.io.savemat('P.mat', mdict={'P': Pf})

    #exit(0)









    cummP = sp.sparse.identity(N)
    cummR = sp.sparse.identity(N)
    for i in range(nr_levels-1):

        # fine and coarse matrices
        Af = ml.levels[i].A
        Ac = ml.levels[i+1].A
        # P and R
        R = ml.levels[i].R
        P = ml.levels[i].P

        # ------------
        # ESTIMATING VARIANCE THEORETICALLY

        Qf = Af.copy()
        mat_size = int(Qf.shape[0]/2)
        Qf[mat_size:,:] = -Qf[mat_size:,:]
        Qc = Ac.copy()
        mat_size = int(Qc.shape[0]/2)
        Qc[mat_size:,:] = -Qc[mat_size:,:]

        Afinv = np.linalg.inv(Qf.copy().toarray())

        Acinv = np.linalg.inv(Qc.copy().toarray())

        diff_mats = Afinv-P*Acinv*R
        np.fill_diagonal(diff_mats,0)
        fro_norm1 = npnorm(diff_mats,ord='fro')
        fro_norm1 -= np.sum(np.square(np.diag(diff_mats)))
        np.fill_diagonal(diff_mats,0)
        fro_norm2 = npnorm(diff_mats,ord='fro')
        print("\nFrobenius norm of off-diagonal part : "+str(fro_norm2)+"\n")
        #print(fro_norm2)

        # ------------

        #exit(0)

        print("Computing for level "+str(i)+"...")

        ests = np.zeros(trace_max_nr_ests, dtype=Af.dtype)
        for j in range(trace_max_nr_ests):

            # generate a Rademacher vector
            x0 = np.random.randint(4, size=N)
            x0 *= 2
            x0 -= 1
            x0 = np.where(x0==3, -1j, x0)
            x0 = np.where(x0==5, 1j, x0)

            #print(x0.shape)
            #print(type(x0))

            x = cummR*x0

            # FIXME : building for Q i.e. applying g3
            x_size = int(x.shape[0]/2)
            x[x_size:] = -x[x_size:]

            np_Af = Af.todense()

            #z,num_iters = solver_sparse(Af,x,level_solver_tol,solver,"cg")
            #z,num_iters = solver_sparse(Af,x,level_solver_tol,ml_solvers[i],"mg")
            z = np.dot(np.linalg.inv(np_Af),x)
            z = z.transpose()
            #print(z.shape)
            num_iters = 0
            #exit(0)

            #exact_sol = np.dot(np.linalg.inv(np_Af),x)
            #print("** error in solves : "+str(np.linalg.norm(z-exact_sol)))

            output_params['results'][i]['solver_iters'] += num_iters

            # FIXME : reversing back the application of g3
            x_size = int(x.shape[0]/2)
            x[x_size:] = -x[x_size:]

            xc = R*x

            # FIXME : building for Q i.e. applying g3
            xc_size = int(xc.shape[0]/2)
            xc[xc_size:] = -xc[xc_size:]

            # for the last level, there is no MG
            #y,num_iters = solver_sparse(Ac,xc,level_solver_tol,solver,"cg")

            #if (i+1)==(nr_levels-1):
            #    y,num_iters = solver_sparse(Ac,xc,level_solver_tol,solver,"cg")
            #else:
            #    y,num_iters = solver_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],"mg")

            np_Ac = Ac.todense()

            #y,num_iters = solver_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],"mg")
            y = np.dot(np.linalg.inv(np_Ac),xc)
            y = y.transpose()
            num_iters = 0

            output_params['results'][i+1]['solver_iters'] += num_iters

            #print(num_iters)

            xx1 = np.asarray(cummP*z).reshape(-1)
            xx2 = np.asarray(cummP*P*y).reshape(-1)

            e1 = np.vdot(x0,xx1)
            e2 = np.vdot(x0,xx2)

            #print(e1)
            #print(e2)

            ests[j] = e1-e2

            # average of estimates
            ests_avg = np.sum(ests[0:(j+1)])/(j+1)
            # and standard deviation
            ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(j+1)]-ests_avg)))/(j+1))
            error_est = ests_dev/sqrt(j+1)

            print(str(j)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol))

            # break condition
            if j>11 and error_est<level_trace_tol:
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
    for i in range(trace_max_nr_ests):

        # generate a Rademacher vector
        xc = np.random.randint(4, size=Nc)
        xc *= 2
        xc -= 1
        xc = np.where(xc==3, -1j, xc)
        xc = np.where(xc==5, 1j, xc)

        x1 = cummP*xc
        x2 = cummR*x1

        # FIXME : building for Q i.e. applying g3
        x2_size = int(x2.shape[0]/2)
        x2[x2_size:] = -x2[x2_size:]

        np_Ac = Ac.todense()
        #y,num_iters = solver_sparse(Ac,x2,level_solver_tol,solver,"cg")
        #y,num_iters = solver_sparse(Ac,x2,level_solver_tol,ml_solvers[nr_levels-1],"mg")
        y = np.dot(np.linalg.inv(np_Ac),x2)
        y = y.transpose()
        num_iters = 0

        y = np.asarray(y).reshape(-1)

        output_params['results'][nr_levels-1]['solver_iters'] += num_iters

        #print(num_iters)

        ests[i] = np.vdot(xc,y)

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(i+1)]-ests_avg)))/(i+1))
        error_est = ests_dev/sqrt(i+1)

        print(str(i)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol))

        # break condition
        if i>11 and error_est<level_trace_tol:
            break

    output_params['results'][nr_levels-1]['nr_ests'] += i

    # set trace and standard deviation
    output_params['results'][nr_levels-1]['ests_avg'] = ests_avg
    output_params['results'][nr_levels-1]['ests_dev'] = ests_dev

    for i in range(nr_levels):
        #output_params['results'][i]['level_complexity'] += output_params['results'][i]['solver_iters']*ml_solvers[i].cycle_complexity()
        slvr = ml_solvers[i]
        output_params['results'][i]['level_complexity'] += output_params['results'][i]['solver_iters']*flopsV(len(slvr.levels), slvr.levels, 0)

    for i in range(nr_levels):
        output_params['total_complexity'] += output_params['results'][i]['level_complexity']

    #print(np.trace(np.linalg.inv(Ac.todense())))

    # total trace
    for i in range(nr_levels):
        output_params['trace'] += output_params['results'][i]['ests_avg']

    print("")

    #return (ests_avg,ests_dev,i,solver_iters)
    return output_params
