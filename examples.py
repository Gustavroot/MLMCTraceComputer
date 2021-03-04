from matrix import loadMatrix
from solver import loadSolver
import scipy as sp
import numpy as np
from stoch_trace import hutchinson, mlmc



# this example assumes the matrix is Hermitian, and computes via deflated Hutchinson
def EXAMPLE_001(params):

    print("\n----------------------------------------------------------")
    print("Example 01 : computing tr(A^{-1}) with deflated Hutchinson")
    print("----------------------------------------------------------\n")

    # checking input params
    if 'matrix' not in params:
        raise Exception('From <EXAMPLE_001(...)> : specify a matrix type.')
    else:
        matrix_name = params['matrix']
    if 'solver' not in params:
        # choosing cg as default
        solver_name = 'cg'
    else:
        solver_name = params['solver']
    if 'matrix_params' not in params:
        raise Exception("From <EXAMPLE_001(...)> : you need to provide the params of the matrix.")

    # extracting matrix
    A,B = loadMatrix(matrix_name, params['matrix_params'])

    # setting solver
    solver = loadSolver(solver_name, A=A, B=B)

    # TODO : check more input params

    trace_tol = params['trace_tol']
    trace_use_Q = params['trace_use_Q']
    solver_tol = params['solver_tol']
    max_nr_levels = params['max_nr_levels']

    trace_params = dict()
    solver_params = dict()
    solver_params['name'] = solver_name
    solver_params['tol'] = solver_tol
    trace_params['solver_params'] = solver_params
    trace_params['tol'] = trace_tol
    trace_params['max_nr_ests'] = 1000000
    trace_params['use_Q'] = trace_use_Q
    trace_params['max_nr_levels'] = max_nr_levels
    trace_params['problem_name'] = params['matrix_params']['problem_name']
    trace_params['nr_deflat_vctrs'] = params['nr_deflat_vctrs']
    #trace_params['aggregation_type'] = params['aggregation_type']
    result = hutchinson(A, solver, trace_params)
    trace = result['trace']
    std_dev = result['std_dev']
    nr_ests = result['nr_ests']
    solver_iters = result['solver_iters']

    if solver_name=='mg':
        total_complexity = result['total_complexity']

    print(" -- matrix : "+matrix_name)
    print(" -- matrix size : "+str(A.shape[0])+"x"+str(A.shape[1]))
    print(" -- tr(A^{-1}) = "+str(trace))
    if solver_name=='mg':
        cmplxity = total_complexity/(1.0e+6)
        print(" -- total MG complexity = "+str(cmplxity)+" MFLOPS")
    print(" -- std dev = "+str(std_dev))
    print(" -- var = "+str(std_dev*std_dev))
    print(" -- number of estimates = "+str(nr_ests))
    print(" -- solver iters = "+str(solver_iters))

    print("\n")


# this example assumes the matrix is Hermitian, and computes via MLMC
def EXAMPLE_002(params):

    print("\n-------------------------------------------")
    print("Example 02 : computing tr(A^{-1}) with MLMC")
    print("-------------------------------------------\n")

    # checking input params
    if 'matrix' not in params:
        raise Exception('From <EXAMPLE_002(...)> : specify a matrix type.')
    else:
        matrix_name = params['matrix']
    if 'solver' not in params:
        # choosing cg as default
        solver_name = 'cg'
    else:
        solver_name = params['solver']
    if 'matrix_params' not in params:
        raise Exception("From <EXAMPLE_002(...)> : you need to provide the params of the matrix.")

    # extracting matrix
    A,B = loadMatrix(matrix_name, params['matrix_params'])

    # setting solver -- in this example, simply load CG (or whatever "plain/simple" solver is specified)
    solver = loadSolver(solver_name)

    # TODO : check more input params

    trace_tol = params['trace_tol']
    trace_use_Q = params['trace_use_Q']
    solver_tol = params['solver_tol']
    solver_lambda_min = params['solver_lambda_min']
    max_nr_levels = params['max_nr_levels']
    trace_ml_constr = params['trace_multilevel_construction']

    trace_params = dict()
    solver_params = dict()
    solver_params['name'] = solver_name
    solver_params['tol'] = solver_tol
    solver_params['lambda_min'] = solver_lambda_min
    trace_params['solver_params'] = solver_params
    trace_params['tol'] = trace_tol
    trace_params['max_nr_ests'] = 1000000
    trace_params['max_nr_levels'] = max_nr_levels
    trace_params['multilevel_construction'] = trace_ml_constr
    trace_params['use_Q'] = trace_use_Q
    trace_params['problem_name'] = params['matrix_params']['problem_name']
    trace_params['aggregation_type'] = params['aggregation_type']
    result = mlmc(A, solver, trace_params)

    print(" -- matrix : "+matrix_name)
    print(" -- matrix size : "+str(A.shape[0])+"x"+str(A.shape[1]))
    print(" -- tr(A^{-1}) = "+str(result['trace']))
    cmplxity = result['total_complexity']/(1.0e+6)
    print(" -- total MG complexity = "+str(cmplxity)+" MFLOPS")
    #print(" -- std dev = "+str(result['std_dev']))
    print(" -- std dev = ---")
    for i in range(result['nr_levels']):
        print(" -- level : "+str(i))
        print(" \t-- number of estimates = "+str(result['results'][i]['nr_ests']))
        print(" \t-- solver iters = "+str(result['results'][i]['solver_iters']))
        print(" \t-- trace = "+str(result['results'][i]['ests_avg']))
        print(" \t-- std dev = "+str(result['results'][i]['ests_dev']))
        print(" \t-- var = "+str(result['results'][i]['ests_dev'] * result['results'][i]['ests_dev']))
        #if i<(result['nr_levels']-1):
        cmplxity = result['results'][i]['level_complexity']/(1.0e+6)
        print("\t-- level MG complexity = "+str(cmplxity)+" MFLOPS")

    print("\n")
