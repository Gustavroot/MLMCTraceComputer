import numpy as np
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import fgmres


ml = []
level_nr = 0
total_levels = 0


def one_mg_step( b ):

    global ml
    global level_nr
    global total_levels

    level_id = total_levels-level_nr

    #print( "nr levels = "+str(len(ml.levels)) )
    #print( "level nr = "+str(level_nr) )

    rs = [ np.zeros(ml.levels[i].A.shape[0],dtype=ml.levels[i].A.dtype) for i in range(level_nr,total_levels) ]
    bs = [ np.zeros(ml.levels[i].A.shape[0],dtype=ml.levels[i].A.dtype) for i in range(level_nr,total_levels) ]
    xs = [ np.zeros(ml.levels[i].A.shape[0],dtype=ml.levels[i].A.dtype) for i in range(level_nr,total_levels) ]

    #print(b.shape)
    #print(bs[0].shape)

    bs[0][:] = b[:]

    # go down in the V-cycle
    for i in range(level_id-1):
        # 1. build the residual
        rs[i] = bs[i]-ml.levels[i+level_nr].A*xs[i]
        # 2. smooth
        e, exitCode = lgmres( ml.levels[i+level_nr].A,rs[i],maxiter=2 )
        # 3. update solution
        xs[i] += e
        # 4. update residual
        rs[i] = bs[i]-ml.levels[i+level_nr].A*xs[i]
        # 5. restrict residual
        bs[i+1] = ml.levels[i+level_nr].R*rs[i]

    # coarsest level solve
    xs[i], exitCode = lgmres( ml.levels[i+level_nr].A,bs[i],tol=1.0e-4 )

    # go up in the V-cycle
    for i in range(level_id-2,-1,-1):
        # 1. interpolate and update
        xs[i] += ml.levels[i+level_nr].P*xs[i+1]
        # 2. build the residual
        rs[i] = bs[i]-ml.levels[i+level_nr].A*xs[i]
        # 3. smooth
        e, exitCode = lgmres( ml.levels[i+level_nr].A,rs[i],maxiter=2 )
        # 4. update solution
        xs[i] += e

    return xs[0]



def mg_solve( A,b,tol ):

    #x = one_mg_step( b )
    #print( np.linalg.norm(b-A*x)/np.linalg.norm(b) )

    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    if A.shape[0]<1000:
        maxiter = A.shape[0]
    else:
        maxiter = 1000

    lop = LinearOperator(A.shape, matvec=one_mg_step)
    x,exitCode = fgmres( A,b,tol=tol,M=lop,callback=callback,maxiter=maxiter )

    return (x,num_iters)
