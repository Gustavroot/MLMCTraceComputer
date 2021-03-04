from examples import EXAMPLE_001, EXAMPLE_002
import numpy as np



# -------------------------------------------------------------

def set_params(example_name):

    if example_name=='2dlaplace':

        # this example computes the chosen matrix via MLMC

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        matrix_params['N'] = 64
        params['matrix'] = '2dlaplace'
        matrix_params['problem_name'] = '2dlaplace'
        params['trace_tol'] = 0.1e-1
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'pyamg'
        params['max_nr_levels'] = 3
        params['nr_deflat_vctrs'] = 2

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

    elif example_name=='graph':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify

        # with G the matrix from Suite Sparse, then, we are computing
        # here tr(A^{-1}) with A = a1*I + a2*G
        N = 472
        matrix_params['N'] = N
        params['matrix'] = 'spd_Erdos971.mat'
        matrix_params['problem_name'] = 'graph'
        matrix_params['a1'] = 1.0
        #matrix_params['a2'] = -1.0/(N*N - 1.0)
        matrix_params['a2'] = -0.065
        #matrix_params['problem_name'] = 'diffusion2D'
        params['trace_tol'] = 0.3e-1
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'pyamg'
        params['max_nr_levels'] = 3
        params['nr_deflat_vctrs'] = 2

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

# -------------------------------------------------------------


# MLMC

# Schwinger 16^2
def G101():

    # this example computes the chosen matrix via MLMC

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 16
    params['matrix'] = 'mat_schwinger16x16b3phasenum11000.mat'
    matrix_params['problem_name'] = 'schwinger'
    params['trace_tol'] = 0.3e-1
    params['trace_use_Q'] = True
    params['trace_multilevel_construction'] = 'manual_aggregation'
    params['max_nr_levels'] = 3

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2

    EXAMPLE_002(params)


# Schwinger 128^2
def G102():

    # this example computes the chosen matrix via MLMC

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 128
    params['matrix'] = 'mat_schwinger128x128b3phasenum11000.mat'
    matrix_params['problem_name'] = 'schwinger'
    params['trace_tol'] = 0.3e-1
    params['trace_use_Q'] = True
    params['trace_multilevel_construction'] = 'manual_aggregation'
    params['max_nr_levels'] = 3

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2

    EXAMPLE_002(params)


# Gauge Laplacian
def G103():

    # this example computes the chosen matrix via MLMC

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 16
    matrix_params['alpha'] = 1.0
    matrix_params['beta'] = 0.0015
    params['matrix'] = 'gauge_laplacian'
    matrix_params['problem_name'] = 'gauge_laplacian'
    params['trace_tol'] = 0.3e-1
    params['trace_use_Q'] = False
    params['trace_multilevel_construction'] = 'pyamg'
    params['max_nr_levels'] = 3

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'ASA'

    EXAMPLE_002(params)


# Laplace 2D
def G104():

    # this example computes the chosen matrix via MLMC

    params = set_params('2dlaplace')

    # fixed parameters
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'ASA'

    EXAMPLE_002(params)


# Linear Elasticity
def G105():

    # this example computes the chosen matrix via MLMC

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 64
    params['matrix'] = 'linear_elasticity'
    matrix_params['problem_name'] = 'linear_elasticity'
    params['trace_tol'] = 0.3e-1
    params['trace_use_Q'] = False
    params['trace_multilevel_construction'] = 'pyamg'
    params['max_nr_levels'] = 3

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'ASA'

    EXAMPLE_002(params)


# diffusion
def G106():

    # this example computes the chosen matrix via MLMC

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 64
    params['matrix'] = 'diffusion2D'
    matrix_params['problem_name'] = 'diffusion2D'
    params['trace_tol'] = 0.3e-1
    params['trace_use_Q'] = False
    params['trace_multilevel_construction'] = 'pyamg'
    params['max_nr_levels'] = 3

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'ASA'

    EXAMPLE_002(params)


# undirected graph
def G107():

    # this example computes the chosen matrix via MLMC

    params = set_params('graph')

    # fixed parameters
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'SA'

    EXAMPLE_002(params)


# LQCD
def G108():

    # this example computes the chosen matrix via MLMC

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 0
    params['matrix'] = 'LQCD_A1.mat'
    matrix_params['problem_name'] = 'LQCD'
    matrix_params['problem_name'] = 'LQCD'
    params['trace_tol'] = 0.3e-1
    params['trace_use_Q'] = True
    params['trace_multilevel_construction'] = 'manual_aggregation'
    params['max_nr_levels'] = 3

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2

    EXAMPLE_002(params)


# -------------------------------------------------------------

# deflated Hutchinson

# Schwinger 16^2
def G201():

    # this example computes the chosen matrix via deflated Hutchinson

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 16
    params['matrix'] = 'mat_schwinger16x16b3phasenum11000.mat'
    matrix_params['problem_name'] = 'schwinger'
    params['trace_tol'] = 0.3e-1
    params['nr_deflat_vctrs'] = 2
    params['trace_use_Q'] = True

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)


# Schwinger 128^2
def G202():

    # this example computes the chosen matrix via deflated Hutchinson

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 128
    params['matrix'] = 'mat_schwinger128x128b3phasenum11000.mat'
    matrix_params['problem_name'] = 'schwinger'
    params['trace_tol'] = 0.3e-1
    params['nr_deflat_vctrs'] = 2
    params['trace_use_Q'] = True

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)


# Gauge Laplacian
def G203():

    # this example computes the chosen matrix via deflated Hutchinson

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 16
    matrix_params['alpha'] = 1.0
    matrix_params['beta'] = 0.0015
    params['matrix'] = 'gauge_laplacian'
    matrix_params['problem_name'] = 'gauge_laplacian'
    params['trace_tol'] = 0.3e-1
    params['nr_deflat_vctrs'] = 2
    params['trace_use_Q'] = False

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)


# Laplace 2D
def G204():

    # this example computes the chosen matrix via deflated Hutchinson

    params = set_params('2dlaplace')

    # fixed parameters
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)


# Linear Elasticity
def G205():

    # this example computes the chosen matrix via deflated Hutchinson

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 64
    params['matrix'] = 'linear_elasticity'
    matrix_params['problem_name'] = 'linear_elasticity'
    params['trace_tol'] = 0.3e-1
    params['nr_deflat_vctrs'] = 2
    params['trace_use_Q'] = False

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)


# diffusion
def G206():

    # this example computes the chosen matrix via deflated Hutchinson

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 64
    params['matrix'] = 'diffusion2D'
    matrix_params['problem_name'] = 'diffusion2D'
    params['trace_tol'] = 0.3e-1
    params['nr_deflat_vctrs'] = 2
    params['trace_use_Q'] = False

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)


# undirected graph
def G207():

    # this example computes the chosen matrix via deflated Hutchinson

    params = set_params('graph')

    # fixed parameters
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)


# LQCD
def G208():

    # this example computes the chosen matrix via deflated Hutchinson

    np.random.seed(51234)

    params = dict()
    matrix_params = dict()

    # to modify
    matrix_params['N'] = 0
    params['matrix'] = 'LQCD_A1.mat'
    matrix_params['problem_name'] = 'LQCD'
    params['trace_tol'] = 0.3e-1
    params['nr_deflat_vctrs'] = 2
    params['trace_use_Q'] = True

    # fixed parameters
    params['matrix_params'] = matrix_params
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)
