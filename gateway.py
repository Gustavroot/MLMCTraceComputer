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
        matrix_params['N'] = 128
        params['matrix'] = '2dlaplace'
        matrix_params['problem_name'] = '2dlaplace'
        params['trace_tol'] = 1.0e-3
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'pyamg'
        params['max_nr_levels'] = 4
        params['nr_deflat_vctrs'] = 8

        params['coarsest_level_directly'] = True
        params['accuracy_eigvs'] = 'high'

        params['aggrs'] = None
        params['dof'] = None

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

    if example_name=='3dlaplace':

        # this example computes the chosen matrix via MLMC

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        matrix_params['N'] = 128
        params['matrix'] = '3dlaplace'
        matrix_params['problem_name'] = '3dlaplace'
        params['trace_tol'] = 1.0e-4
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'pyamg'
        params['max_nr_levels'] = 6
        params['nr_deflat_vctrs'] = 32

        params['coarsest_level_directly'] = True
        params['accuracy_eigvs'] = 'high'

        params['aggrs'] = None
        params['dof'] = None

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
        #N = 472
        matrix_params['N'] = 0
        params['matrix'] = 'spd_Erdos971.mat'
        matrix_params['problem_name'] = 'graph'
        matrix_params['a1'] = 1.0
        #matrix_params['a2'] = -1.0/(N*N - 1.0)
        matrix_params['a2'] = 1.0
        #matrix_params['a2'] = -0.065
        #matrix_params['problem_name'] = 'diffusion2D'
        params['trace_tol'] = 0.3e-1
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'pyamg'
        params['max_nr_levels'] = 3
        params['nr_deflat_vctrs'] = 2

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

    elif example_name=='estrada_index':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify

        # with B the matrix from Suite Sparse, then, we are computing
        # here tr(L^{-1}) with L = a1*I + a2*B
        #N = 472
        matrix_params['N'] = 0
        #params['matrix'] = 'spd_G56.mat'
        #params['matrix'] = 'spd_Erdos02.mat'
        params['matrix'] = 'spd_USpowerGrid.mat'
        matrix_params['problem_name'] = 'estrada_index'
        matrix_params['a1'] = 1.01
        #matrix_params['a2'] = -1.0/(N*N - 1.0)
        matrix_params['a2'] = -1.0
        #matrix_params['a2'] = -0.065
        #matrix_params['problem_name'] = 'diffusion2D'
        params['trace_tol'] = 3.0e-3
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'pyamg'
        params['max_nr_levels'] = 3
        # this parameter is changed to compare against MLMC , nr_defl = 2,32,128
        params['nr_deflat_vctrs'] = 64

        params['coarsest_level_directly'] = True
        params['accuracy_eigvs'] = 'high'

        #params['function'] = 'inverse'
        params['function'] = 'exponential'

        params['aggrs'] = None
        params['dof'] = None

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

    elif example_name=='schwinger128':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        matrix_params['N'] = 128
        params['matrix'] = 'mat_schwinger128x128b3phasenum11000.mat'
        matrix_params['problem_name'] = 'schwinger'
        params['trace_tol'] = 2.5e-3
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'manual_aggregation'
        #params['trace_multilevel_construction'] = 'pyamg'
        params['max_nr_levels'] = 4
        params['coarsest_level_directly'] = True
        params['accuracy_eigvs'] = 'high'

        params['aggrs'] = [2*2,8*8,2*2]
        params['dof'] = [2,2,8,8]
        params['nr_deflat_vctrs'] = params['dof'][1]

        #matrix_params['mass'] = -0.13750
        #matrix_params['mass'] = -0.13
        matrix_params['mass'] = 0.0

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

    elif example_name=='schwinger16':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        matrix_params['N'] = 16
        params['matrix'] = 'mat_schwinger16x16b3phasenum11000.mat'
        matrix_params['problem_name'] = 'schwinger'
        params['trace_tol'] = 2.0e-2
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'manual_aggregation'
        params['max_nr_levels'] = 3
        params['coarsest_level_directly'] = True
        params['accuracy_eigvs'] = 'high'
        params['nr_deflat_vctrs'] = 2

        #matrix_params['mass'] = -0.8940
        matrix_params['mass'] = -0.95

        params['aggrs'] = [2*2,2*2,2*2]
        params['dof'] = [2,2,2,2]

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

    elif example_name=='LQCDsmall':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        matrix_params['N'] = 0
        params['matrix'] = 'LQCD_A1small.mat'
        matrix_params['problem_name'] = 'LQCD'
        params['trace_tol'] = 0.5e-0
        params['trace_use_Q'] = True
        params['trace_multilevel_construction'] = 'manual_aggregation'
        params['max_nr_levels'] = 3
        params['coarsest_level_directly'] = True
        params['accuracy_eigvs'] = 'high'
        #params['nr_deflat_vctrs'] = 2

        matrix_params['mass'] = 0.0

        params['aggrs'] = [2*2*2*2,2*2*2*2]
        params['dof'] = [12,24,32]
        params['nr_deflat_vctrs'] = params['dof'][0]

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params


    elif example_name=='LQCDlarge':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        matrix_params['N'] = 0
        params['matrix'] = 'LQCD_A1large.mat'
        matrix_params['problem_name'] = 'LQCD'
        params['trace_tol'] = 0.8e-0
        params['trace_use_Q'] = True
        params['trace_multilevel_construction'] = 'manual_aggregation'
        params['max_nr_levels'] = 3
        params['coarsest_level_directly'] = True
        params['accuracy_eigvs'] = 'low'
        #params['nr_deflat_vctrs'] = 2

        # 0.57305644+0.2178393j
        matrix_params['mass'] = -0.5

        params['aggrs'] = [4*4*4*4,2*2*2*2]
        params['dof'] = [12,24,32]
        params['nr_deflat_vctrs'] = params['dof'][1]

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

    elif example_name=='GL':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        matrix_params['N'] = 128
        matrix_params['alpha'] = 1.0
        matrix_params['beta'] = 0.001
        params['matrix'] = 'gauge_laplacian'
        matrix_params['problem_name'] = 'gauge_laplacian'
        params['trace_tol'] = 0.5e-3
        params['trace_use_Q'] = False
        params['trace_multilevel_construction'] = 'pyamg'
        params['max_nr_levels'] = 4
        params['coarsest_level_directly'] = True
        params['accuracy_eigvs'] = 'high'

        params['aggrs'] = None
        params['dof'] = None

        params['nr_deflat_vctrs'] = 8

        # fixed parameters
        params['matrix_params'] = matrix_params

        return params

    else:
        raise Exception("Non-existent option for example type.")


# -------------------------------------------------------------


# MLMC

# Schwinger 16^2
def G101():

    # this example computes the chosen matrix via MLMC

    params = set_params('schwinger16')

    # fixed params
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_002(params)


# Schwinger 128^2
def G102():

    # this example computes the chosen matrix via MLMC

    params = set_params('schwinger128')

    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'ASA'

    EXAMPLE_002(params)


# Gauge Laplacian
def G103():

    # this example computes the chosen matrix via MLMC

    params = set_params('GL')

    # fixed parameters
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

    params = set_params('LQCDsmall')

    # fixed parameters
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_002(params)


# LQCD
def G109():

    # this example computes the chosen matrix via MLMC

    params = set_params('LQCDlarge')

    # fixed parameters
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_002(params)


# Laplace 3D
def G110():

    # this example computes the chosen matrix via MLMC

    params = set_params('3dlaplace')

    # fixed parameters
    params['solver'] = 'cg'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'ASA'

    EXAMPLE_002(params)


# Extrada Index
def G111():

    # this example computes the chosen matrix via MLMC

    params = set_params('estrada_index')

    # fixed parameters
    params['solver'] = 'gmres'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'SA'

    EXAMPLE_002(params)


# -------------------------------------------------------------

# deflated Hutchinson

# Schwinger 16^2
def G201():

    # this example computes the chosen matrix via deflated Hutchinson

    params = set_params('schwinger16')

    # fixed parameters
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_001(params)


# Schwinger 128^2
def G202():

    # this example computes the chosen matrix via deflated Hutchinson

    params = set_params('schwinger128')

    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)


# Gauge Laplacian
def G203():

    # this example computes the chosen matrix via deflated Hutchinson

    params = set_params('GL')

    # fixed parameters
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

    params = set_params('LQCDsmall')

    # fixed parameters
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_001(params)


# LQCD
def G209():

    # this example computes the chosen matrix via deflated Hutchinson

    params = set_params('LQCDlarge')

    # fixed parameters
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_001(params)


# Laplace 3D
def G210():

    # this example computes the chosen matrix via deflated Hutchinson

    params = set_params('3dlaplace')

    # fixed parameters
    params['solver'] = 'mg'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_001(params)


# Estrada Index
def G211():

    # this example computes the chosen matrix via deflated Hutchinson

    params = set_params('estrada_index')

    # fixed parameters
    params['spec_function'] = 'mg'
    params['function_tol'] = 1e-3
    params['max_nr_levels'] = 4

    EXAMPLE_001(params)
