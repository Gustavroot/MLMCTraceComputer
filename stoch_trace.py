# Stochastic methods for the computation of the trace of the inverse

import numpy as np
import scipy as sp
from solver import solver_sparse
from math import sqrt, pow
import pyamg
from utils import flopsV
from pyamg.aggregation.adaptive import adaptive_sa_solver
from aggregation import manual_aggregation
from scipy.sparse import csr_matrix
from numpy.linalg import norm as npnorm
from scipy.sparse.linalg import norm
import png
from numpy.linalg import eigh
from scipy.sparse import identity

from scipy.sparse.linalg import svds,eigsh


# ---------------------------------

# specific to LQCD

class LevelML:
    R = 0
    P = 0
    A = 0
    Q = 0

class SimpleML:
    levels = []

    def __str__(self):
        return "For manual aggregation, printing <ml> is under construction"

# ---------------------------------

def gamma3_application(v):
    v_size = int(v.shape[0]/2)
    v[v_size:] = -v[v_size:]
    return v

def gamma5_application(v,l):

    dof = [12,160]
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

# ---------------------------------

# compute tr(A^{-1}) via Hutchinson
def hutchinson(A, solver, params):

    print( np.trace( np.linalg.inv( A.todense() ) ) )

    # TODO : check input params !

    max_nr_levels = params['max_nr_levels']

    # solver params
    solver_name = params['solver_params']['name']
    solver_tol = params['solver_params']['tol']

    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']

    use_Q = params['use_Q']

    # size of the problem
    N = A.shape[0]

    solver_tol = 1e-5

    if use_Q:
        print("Constructing sparse Q ...")
        Q = A.copy()
        mat_size = int(Q.shape[0]/2)
        Q[mat_size:,:] = -Q[mat_size:,:]
        print("... done")

    # compute the SVD (for low-rank part of deflation)
    np.random.seed(65432)
    nr_deflat_vctrs = params['nr_deflat_vctrs']
    print("Computing SVD (finest level) ...")
    if use_Q:
        Sy,Ux = eigsh( Q,k=nr_deflat_vctrs,which='LM',tol=1.0e-5,sigma=0.0 )
        Vx = np.copy(Ux)
    else:
        diffA = A-A.getH()
        diffA_norm = norm( diffA,ord='fro' )
        if diffA_norm<1.0e-14:
            Sy,Ux = eigsh( A,k=nr_deflat_vctrs,which='LM',tol=1.0e-5,sigma=0.0 )
            Vx = np.copy(Ux)
        else:
            Ux,Sy,Vy = svds(A, k=nr_deflat_vctrs, which='SM', tol=1.0e-5)
            Vx = Vy.transpose().conjugate()
    Sx = np.diag(Sy)
    print("... done")

    # compute low-rank part of deflation
    small_A = np.dot(Vx.transpose().conjugate(),Ux) * np.linalg.inv(Sx)
    tr1 = np.trace(small_A)

    np.random.seed(123456)

    # pre-compute a rough estimation of the trace, to set then a tolerance
    nr_rough_iters = 10
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)
    # main Hutchinson loop
    for i in range(nr_rough_iters):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1
        #x = np.random.randint(4, size=N)
        #x *= 2
        #x -= 1
        #x = np.where(x==3, -1j, x)
        #x = np.where(x==5, 1j, x)

        x = x.astype(A.dtype)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0)

        z,num_iters = solver_sparse(A,x,solver_tol,solver,solver_name)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0)

        e = np.vdot(x,z)
        ests[i] = e

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)

    print("\n** rough estimation of the trace : "+str(rough_trace))
    print("")

    # then, set a rough tolerance
    rough_trace_tol = abs(trace_tol*rough_trace)
    #rough_solver_tol = rough_trace_tol*lambda_min/N
    #rough_solver_tol = abs(rough_trace_tol/N)

    rough_solver_tol = 1e-5

    solver_iters = 0
    ests = np.zeros(trace_max_nr_ests, dtype=A.dtype)
    # main Hutchinson loop
    for i in range(trace_max_nr_ests):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1
        #x = np.random.randint(4, size=N)
        #x *= 2
        #x -= 1
        #x = np.where(x==3, -1j, x)
        #x = np.where(x==5, 1j, x)

        x = x.astype(A.dtype)

        # deflating Vx from x
        x_def = x - np.dot(Vx,np.dot(Vx.transpose().conjugate(),x))

        if use_Q:
            if params['problem_name']=='schwinger':
                x_def = gamma3_application(x_def)
            elif params['problem_name']=='LQCD':
                x_def = gamma5_application(x_def,0)

        z,num_iters = solver_sparse(A,x_def,rough_solver_tol,solver,solver_name)

        solver_iters += num_iters

        e = np.vdot(x,z)

        ests[i] = e

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(   np.sum(   np.square(np.abs(ests[0:(i+1)]-ests_avg))   )/(i+1)   )
        error_est = ests_dev/sqrt(i+1)

        #print(str(i)+" .. "+str(ests_avg)+" .. "+str(rough_trace)+" .. "+str(error_est)+" .. "+str(rough_trace_tol)+" .. "+str(num_iters))

        # break condition
        if i>10 and error_est<rough_trace_tol:
            break

    result = dict()
    result['trace'] = ests_avg+tr1
    result['std_dev'] = ests_dev
    result['nr_ests'] = i
    result['solver_iters'] = solver_iters
    if solver_name=='mg':
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

    print("\nConstruction of P and A at all levels (from finest level) ...")

    if trace_ml_constr=='pyamg':
        if params['aggregation_type']=='SA':
            ml = pyamg.smoothed_aggregation_solver( A,max_levels=max_nr_levels )
        elif params['aggregation_type']=='ASA':
            [ml, work] = adaptive_sa_solver(A, num_candidates=2, candidate_iters=2, improvement_iters=3,
                                            strength='symmetric', aggregate='standard', max_levels=max_nr_levels)
        #[ml, work] = adaptive_sa_solver(A, num_candidates=5, improvement_iters=5)

    # specific to Schwinger
    elif trace_ml_constr=='manual_aggregation':

        # TODO : get <aggr_size> from input params
        # TODO : get <dof_size> from input params

        aggrs = [2*2,8*8]
        dof = [2,2,6,2,2,2]

        ml = manual_aggregation(A, dof=dof, aggrs=aggrs, max_levels=max_nr_levels, dim=2)

    # specific to LQCD
    elif trace_ml_constr=='from_files':

        import scipy.io as sio

        ml = SimpleML()

        # load A at each level
        for i in range(max_nr_levels):
            ml.levels.append(LevelML())
            if i==0:
                mat_contents = sio.loadmat('LQCD_A'+str(i+1)+'.mat')
                Axx = mat_contents['A'+str(i+1)]
                ml.levels[i].A = Axx.copy()

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

        # build the other A's
        for i in range(1,max_nr_levels):
            ml.levels[i].A = ml.levels[i-1].R*ml.levels[i-1].A*ml.levels[i-1].P

    else:
        raise Exception("The specified <trace_multilevel_constructor> does not exist.")

    print("... done")

    print("\nMultilevel information:")
    print(ml)

    # the actual number of levels
    nr_levels = len(ml.levels)

    if nr_levels<3:
        raise Exception("Use three or more levels.")

    for i in range(nr_levels):
        print("size(A"+str(i)+") = "+str(ml.levels[i].A.shape[0])+"x"+str(ml.levels[i].A.shape[1]))

    print("")

    # FIXME : theoretical esimation goes here

    print("\nCreating solver with PyAMG for each level ...")

    ml_solvers = list()
    for i in range(nr_levels-1):
        [mlx, work] = adaptive_sa_solver(ml.levels[i].A, num_candidates=2, candidate_iters=2, improvement_iters=3,
                                         strength='symmetric', aggregate='standard', max_levels=9)
        ml_solvers.append(mlx)

    print("... done")

    solver_tol = 1e-5

    print("\nComputing rough estimation of the trace ...")

    np.random.seed(123456)

    # pre-compute a rough estimate of the trace, to set then a tolerance
    nr_rough_iters = 10
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)
    # main Hutchinson loop
    for i in range(nr_rough_iters):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1
        #x = np.random.randint(4, size=N)
        #x *= 2
        #x -= 1
        #x = np.where(x==3, -1j, x)
        #x = np.where(x==5, 1j, x)

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

    # setting to zero the counters and results to be returned
    output_params = dict()
    output_params['nr_levels'] = nr_levels
    output_params['trace'] = 0.0
    output_params['total_complexity'] = 0.0
    output_params['std_dev'] = 0.0
    output_params['results'] = list()
    for i in range(nr_levels):
        output_params['results'].append(dict())
        output_params['results'][i]['solver_iters'] = 0
        output_params['results'][i]['nr_ests'] = 0
        output_params['results'][i]['ests_avg'] = 0.0
        output_params['results'][i]['ests_dev'] = 0.0
        output_params['results'][i]['level_complexity'] = 0.0

    # compute level differences
    #level_trace_tol  = abs(trace_tol*rough_trace/sqrt(nr_levels-1))
    #level_solver_tol = level_trace_tol/N
    #level_solver_tol = 1e-9/sqrt(nr_levels)
    level_solver_tol = 1e-5

    print("")

    tol_fraction0 = 0.45
    tol_fraction1 = 0.45

    cummP = sp.sparse.identity(N)
    cummR = sp.sparse.identity(N)

    # coarsest-level inverse
    Acc = ml.levels[nr_levels-1].A
    Ncc = Acc.shape[0]
    np_Acc = Acc.todense()
    np_Acc_inv = np.linalg.inv(np_Acc)

    for i in range(nr_levels-1):

        if i==0 : tol_fctr = sqrt(tol_fraction0)
        elif i==1 : tol_fctr = sqrt(tol_fraction1)
        # e.g. sqrt(0.45), sqrt(0.45), sqrt(0.1*(1.0/(nl-2)))
        else : tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-2)

        level_trace_tol  = abs(trace_tol*rough_trace*tol_fctr)

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
            x0 = np.random.randint(2, size=N)
            x0 *= 2
            x0 -= 1
            #x = np.random.randint(4, size=N)
            #x *= 2
            #x -= 1
            #x = np.where(x==3, -1j, x)
            #x = np.where(x==5, 1j, x)

            x0 = x0.astype(A.dtype)

            x = cummR*x0

            if use_Q:
                if params['problem_name']=='schwinger':
                    x = gamma3_application(x)
                elif params['problem_name']=='LQCD':
                    x = gamma5_application(x,i)

            z,num_iters = solver_sparse(Af,x,level_solver_tol,ml_solvers[i],"mg")

            output_params['results'][i]['solver_iters'] += num_iters

            # reverting back the application of gamma
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

            if (i+1)==(nr_levels-1):
                # solve directly
                #np_Ac = Ac.todense()
                y = np.dot(np_Acc_inv,xc)
                y = np.asarray(y).reshape(-1)
                num_iters = 1
            else:
                y,num_iters = solver_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],"mg")
            #y,num_iters = solver_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],"mg")

            output_params['results'][i+1]['solver_iters'] += num_iters

            e1 = np.vdot(x0,cummP*z)
            e2 = np.vdot(x0,cummP*P*y)

            ests[j] = e1-e2

            # average of estimates
            ests_avg = np.sum(ests[0:(j+1)])/(j+1)
            # and standard deviation
            ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(j+1)]-ests_avg)))/(j+1))
            error_est = ests_dev/sqrt(j+1)

            #print(str(j)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol))

            # break condition
            if j>10 and error_est<level_trace_tol:
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

    # in case the coarsest matrix is 1x1
    if Acc.shape[0]==1:

        output_params['results'][nr_levels-1]['nr_ests'] += 1
        # set trace and standard deviation
        output_params['results'][nr_levels-1]['ests_avg'] = 1.0/csr_matrix(Acc)[0,0]
        output_params['results'][nr_levels-1]['ests_dev'] = 0

    else:

        ests = np.zeros(trace_max_nr_ests, dtype=Acc.dtype)

        tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-2)
        level_trace_tol  = abs(trace_tol*rough_trace*tol_fctr)

        # ----- extracting eigenvectors for deflation

        if use_Q:
            print("Constructing sparse Q ...")
            Qcc = Acc.copy()
            mat_size = int(Qcc.shape[0]/2)
            Qcc[mat_size:,:] = -Qcc[mat_size:,:]
            print("... done")

        # compute the SVD (for low-rank part of deflation)
        np.random.seed(65432)
        # more deflation vectors for the coarsest level
        nr_deflat_vctrs = 16

        if nr_deflat_vctrs>(Acc.shape[0]-2) : nr_deflat_vctrs=Acc.shape[0]-2
        print("Computing SVD (coarsest level) ...")
        if use_Q:
            Sy,Ux = eigsh( Qcc,k=nr_deflat_vctrs,which='LM',tol=1.0e-5,sigma=0.0 )
            Vx = np.copy(Ux)
        else:
            diffA = Acc-Acc.getH()
            diffA_norm = norm( diffA,ord='fro' )
            if diffA_norm<1.0e-14:
                Sy,Ux = eigsh( Acc,k=nr_deflat_vctrs,which='LM',tol=1.0e-5,sigma=0.0 )
                Vx = np.copy(Ux)
            else:
                Ux,Sy,Vy = svds(Acc, k=nr_deflat_vctrs, which='SM', tol=1.0e-5)
                Vx = Vy.transpose().conjugate()
        Sx = np.diag(Sy)
        print("... done")

        # compute low-rank part of deflation
        small_A1 = cummP*Ux
        small_A2 = cummR*small_A1
        small_A3 = np.dot(Vx.transpose().conjugate(),small_A2)
        small_A = small_A3*np.linalg.inv(Sx)
        tr1c = np.trace(small_A)

        # -------------------------

        for i in range(trace_max_nr_ests):

            # generate a Rademacher vector

            xc = np.random.randint(2, size=Ncc)
            xc *= 2
            xc -= 1
            #x = np.random.randint(4, size=N)
            #x *= 2
            #x -= 1
            #x = np.where(x==3, -1j, x)
            #x = np.where(x==5, 1j, x)

            xc = xc.astype(A.dtype)

            x1 = cummP*xc
            x2 = cummR*x1

            # deflating Vx from x2
            x2_def = x2 - np.dot(Vx,np.dot(Vx.transpose().conjugate(),x2))

            if use_Q:
                if params['problem_name']=='schwinger':
                    x2_def = gamma3_application(x2_def)
                elif params['problem_name']=='LQCD':
                    x2_def = gamma5_application(x2_def,nr_levels-1)

            #y,num_iters = solver_sparse(Ac,x2,level_solver_tol,ml_solvers[nr_levels-1],"mg")
            y = np.dot(np_Acc_inv,x2_def)
            y = np.asarray(y).reshape(-1)
            num_iters = 1

            output_params['results'][nr_levels-1]['solver_iters'] += num_iters

            ests[i] = np.vdot(xc,y)

            # average of estimates
            ests_avg = np.sum(ests[0:(i+1)])/(i+1)
            # and standard deviation
            ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(i+1)]-ests_avg)))/(i+1))
            error_est = ests_dev/sqrt(i+1)

            #print(str(i)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol))

            # break condition
            if i>10 and error_est<level_trace_tol:
                break

        output_params['results'][nr_levels-1]['nr_ests'] += i
        # set trace and standard deviation
        output_params['results'][nr_levels-1]['ests_avg'] = ests_avg+tr1c
        output_params['results'][nr_levels-1]['ests_dev'] = ests_dev

    for i in range(nr_levels-1):
        #output_params['results'][i]['level_complexity'] += output_params['results'][i]['solver_iters']*ml_solvers[i].cycle_complexity()
        slvr = ml_solvers[i]
        #print( "flops/cycle = "+str(flopsV(len(slvr.levels), slvr.levels, 0)) )
        output_params['results'][i]['level_complexity'] = output_params['results'][i]['solver_iters']*flopsV(len(slvr.levels), slvr.levels, 0)

    output_params['results'][nr_levels-1]['level_complexity'] = output_params['results'][nr_levels-1]['solver_iters']*(ml.levels[nr_levels-1].A.shape[0]*ml.levels[nr_levels-1].A.shape[0])

    #print( output_params['results'][nr_levels-1]['solver_iters'] * (ml.levels[nr_levels-1].A.shape[0]*ml.levels[nr_levels-1].A.shape[0]) )

    for i in range(nr_levels):
        output_params['total_complexity'] += output_params['results'][i]['level_complexity']

    # total trace
    for i in range(nr_levels):
        output_params['trace'] += output_params['results'][i]['ests_avg']

    print("")

    return output_params






# -----

    """
    #print(npnorm(A.todense()-ml.levels[0].A.todense(),'fro'))

    A1xxx = ml.levels[0].A.copy().todense()
    A2xxx = ml.levels[1].A.copy().todense()
    Q1 = A1xxx.copy()
    Q2 = A2xxx.copy()
    for i in range(Q1.shape[1]) : Q1[:,i] = gamma3_application(A1xxx[:,i])
    for i in range(Q2.shape[1]) : Q2[:,i] = gamma3_application(A2xxx[:,i])

    Bx1 = Q1.copy()
    Bx2 = Q2.copy()
    #Bx3 = ml.levels[2].A.copy().todense()

    Cx1 = ml.levels[0].P.copy()
    Dx1 = ml.levels[0].R.copy()
    #Cx2 = ml.levels[1].P.copy()
    #Dx2 = ml.levels[1].R.copy()

    Bx1inv = np.linalg.inv(Bx1)
    Bx2inv = np.linalg.inv(Bx2)
    #Bx3inv = np.linalg.inv(Bx3)

    Bx2invproj = Cx1*Bx2inv*Dx1
    #Bx3invproj = Cx2*Bx3inv*Dx2

    diffxinv1 = Bx1inv - Bx2invproj
    diffxinv1xx = diffxinv1 + diffxinv1.transpose()

    #diffxinv2 = Bx2inv - Bx3invproj
    #diffxinv2xx = diffxinv2 + diffxinv2.transpose()

    offdiag_fro_norm1 = npnorm( diffxinv1xx-np.diag(np.diag(diffxinv1xx)), ord='fro' )
    #offdiag_fro_norm2 = npnorm( diffxinv2xx-np.diag(np.diag(diffxinv2xx)), ord='fro' )

    print("\nTheoretical estimation of variance, diff at level 1 : "+str(0.5*offdiag_fro_norm1*offdiag_fro_norm1))
    #print("\nTheoretical estimation of variance, diff at level 2 : "+str(0.5*offdiag_fro_norm2*offdiag_fro_norm2))

    # -------------

    pure_x = Bx1inv+Bx1inv.transpose()
    offdiag_fro_norm = npnorm( pure_x-np.diag(np.diag(pure_x)), ord='fro' )
    print("\nTheoretical estimation of variance, pure : "+str(0.5*offdiag_fro_norm*offdiag_fro_norm))

    exit(0)
    """
