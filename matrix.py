# All the resources needed to load and manipulate matrices
import warnings
from scipy.sparse import identity



def loadMatrix(matrix_name, params):

    warnings.simplefilter("ignore")

    if matrix_name=='gauge_laplacian':
        from pyamg.gallery import gauge_laplacian

        # TODO : add params check here !

        N = params['N']
        alpha = params['alpha']
        beta = params['beta']

        return (gauge_laplacian(N, alpha, beta),None)

    elif matrix_name=='2dlaplace':
        from pyamg.gallery import poisson

        N = params['N']

        return (poisson((N,N), format='csr'),None)

    elif matrix_name=='linear_elasticity':
        from pyamg.gallery import linear_elasticity

        N = params['N']

        return linear_elasticity((N,N), format='bsr')

    elif matrix_name=='diffusion2D':

        from pyamg.gallery.diffusion import diffusion_stencil_2d
        from pyamg.gallery import stencil_grid

        N = params['N']

        stencil = diffusion_stencil_2d()
        return (stencil_grid(stencil, (N,N), format='csr'),None)

    elif len(matrix_name.split('_'))==2:

        m = 1.5

        # get filename and load
        filename = matrix_name.split('_')
        if not filename[0]=='mat':
            raise Exception("From <loadMatrix(...)> : the format of the filename should be <mat_X.mat>")

        import scipy.io as sio
        mat_contents = sio.loadmat(matrix_name)

        #print(mat_contents['S'].shape)
        A = mat_contents['S']
        A += m*identity(A.shape[0], dtype=A.dtype)

        #print(matrix_name)
        #exit(0)

        return (A,None)

    else:
        raise Exception("From <loadMatrix(...)> : unknown value for input variable <matrix_name>")
