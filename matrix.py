# All the resources needed to load and manipulate matrices
import warnings
from scipy.sparse import identity
from scipy.sparse import csr_matrix
import matplotlib.pylab as plt



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

    elif len(matrix_name.split('_'))==2 and matrix_name.split('_')[0]=='spd':

        import scipy.io as sio

        mat_contents = sio.loadmat(matrix_name)
        Axx = mat_contents['Problem']['A'][0,0]
        Ax = csr_matrix(Axx)

        #A = params['a1']*identity(Ax.shape[0],dtype=Ax.dtype) + params['a2']*Ax
        N = Ax.shape[0]
        A = params['a1']*identity(Ax.shape[0],dtype=Ax.dtype) + ( -(1.0/(N*N-1.0)) ) *Ax

        #plt.spy(A)
        #plt.show()

        return (A,None)


    elif len(matrix_name.split('_'))==2:

        #m = -0.8854 # this value is for a 16^2 lattice
        #m = -0.115 # this value is for a 128^2 lattice
        #m = -0.0899

        import scipy.io as sio

        filename = matrix_name.split('_')

        # check if LQCD
        if filename[0]=='LQCD':
            mat_contents = sio.loadmat(matrix_name)
            A = mat_contents['A1']

            m = 1.0
            A += m*identity(A.shape[0], dtype=A.dtype)

            return (A,None)

        # Schwinger, 16^2
        #m = -0.8940

        # Schwinger, 128^2
        #m = -0.12750

        m = -0.13750
        #m = -0.13250
        #m = 0.0

        # get filename and load
        if not filename[0]=='mat':
            raise Exception("From <loadMatrix(...)> : the format of the filename should be <mat_X.mat>")

        mat_contents = sio.loadmat(matrix_name)

        A = mat_contents['S']
        A += m*identity(A.shape[0], dtype=A.dtype)

        # FIXME : this application of g3 is not needed in general !!
        if matrix_name=='mat_schwinger16x16b3phasenum11000.mat':
            mat_size = int(A.shape[0]/2)
            A[mat_size:,:] = -A[mat_size:,:]

        #print(matrix_name)
        #exit(0)

        #print("matrix loaded!")

        return (A,None)

    else:
        raise Exception("From <loadMatrix(...)> : unknown value for input variable <matrix_name>")
