from examples import EXAMPLE_001, EXAMPLE_002
import numpy as np



# main section
if __name__=='__main__':

    # --------------------------------
    np.random.seed(51234)
    # this second example computes the chosen matrix via MLMC

    params = dict()
    matrix_params = dict()

    # options of matrices for testing :

    # 1. for Gauge Laplacian
    #matrix_params['N'] = 16
    #matrix_params['alpha'] = 1.0
    #matrix_params['beta'] = 0.0015
    #params['matrix'] = 'gauge_laplacian'

    # 2. for Laplace 2D
    #matrix_params['N'] = 16
    #params['matrix'] = '2dlaplace'

    # 3. for Linear Elasticity
    #matrix_params['N'] = 64
    #params['matrix'] = 'linear_elasticity'

    # 4. for diffusion
    #matrix_params['N'] = 64
    #params['matrix'] = 'diffusion2D'

    # 5. for Schwinger
    #matrix_params['N'] = 128
    #params['matrix'] = 'mat_schwinger128x128b3phasenum11000.mat'
    matrix_params['N'] = 16
    params['matrix'] = 'mat_schwinger16x16b3phasenum11000.mat'

    params['matrix_params'] = matrix_params
    params['solver'] = 'cg'
    params['trace_tol'] = 1.0e-2
    params['trace_use_Q'] = True
    params['trace_multilevel_construction'] = 'manual_aggregation'
    params['solver_tol'] = 1e-3
    params['solver_lambda_min'] = 1e-2
    params['max_nr_levels'] = 15
    EXAMPLE_002(params)

    # --------------------------------
    np.random.seed(51234)
    # this first example computes the chosen matrix via Hutchinson

    params = dict()
    matrix_params = dict()

    # options of matrices for testing :

    # 1. for Gauge Laplacian
    #matrix_params['N'] = 16
    #matrix_params['alpha'] = 1.0
    #matrix_params['beta'] = 0.0015
    #params['matrix'] = 'gauge_laplacian'

    # 2. for Laplace 2D
    #matrix_params['N'] = 16
    #params['matrix'] = '2dlaplace'

    # 3. for Linear Elasticity
    #matrix_params['N'] = 64
    #params['matrix'] = 'linear_elasticity'

    # 4. for diffusion
    #matrix_params['N'] = 64
    #params['matrix'] = 'diffusion2D'

    # 5. for Schwinger
    #matrix_params['N'] = 128
    #params['matrix'] = 'mat_schwinger128x128b3phasenum11000.mat'
    matrix_params['N'] = 16
    params['matrix'] = 'mat_schwinger16x16b3phasenum11000.mat'

    params['matrix_params'] = matrix_params
    params['solver'] = 'mg'
    params['trace_tol'] = 1.0e-2
    params['trace_use_Q'] = True
    params['solver_tol'] = 1e-3
    params['max_nr_levels'] = 15

    EXAMPLE_001(params)
