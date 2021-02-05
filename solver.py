import scipy as sp
import pyamg
import numpy as np
import warnings



def solver_sparse(A, b, tol, solver, solver_label):
    num_iters = 0

    warnings.simplefilter("ignore")
    #warnings.simplefilter("always")

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    # call the solver on your data
    if solver_label=="cg" or solver_label=="gmres":
        x,_ = solver(A, b, callback=callback, tol=tol)
    elif solver_label=="mg":
        #x = solver.solve(b, accel='cg', callback=callback)
        x = solver.solve(b, accel='gmres', callback=callback, tol=tol, maxiter=b.shape[0])
        #x = solver.solve(b, callback=callback)

        #print(tol)

        #rel_res = np.linalg.norm(b - A*x)/np.linalg.norm(b)

        #print(rel_res)
        #print(np.linalg.norm(b))
        #print(np.linalg.norm(x))
        #print(num_iters)
        #exit(0)

    return (x,num_iters)


def loadSolver(solver_name, A=None, B=None):

    # FIXME
    max_nr_levels = 9

    if solver_name=='cg':
        return sp.sparse.linalg.cg

    elif solver_name=='gmres':
        return sp.sparse.linalg.gmres

    elif solver_name=='mg':
        #ml = pyamg.smoothed_aggregation_solver(A,B)
        #[ml, work] = adaptive_sa_solver(A, num_candidates=5, improvement_iters=5)
        [ml, work] = pyamg.aggregation.adaptive.adaptive_sa_solver(A, num_candidates=2, candidate_iters=2, improvement_iters=3,
                                                                   strength='symmetric', aggregate='standard', max_levels=9)
        #ml = pyamg.smoothed_aggregation_solver(A)
        return ml

    else:
        raise Exception("From <loadSolver(...)> : the chosen <solver_name> is not supported.")
