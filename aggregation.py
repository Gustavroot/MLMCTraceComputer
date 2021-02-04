from pyamg.aggregation.adaptive import adaptive_sa_solver
from scipy.sparse.linalg import eigsh, eigs
from math import pow, sqrt
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import norm
from numpy.linalg import norm as npnorm
import png





class LevelML:
    R = 0
    P = 0
    A = 0

class SimpleML:
    levels = []

    def __str__(self):
        return "For manual aggregation, printing <ml> is under construction"



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



# <dof> :   per level (except the last one, of course), this is a list of
#           the number of degrees of freedom for the next level
# <aggrs> : per level, this is the block size. So, if at a certain level the
#           value is 4, then the aggregates are of size 4^d, where d is the
#           dimensionality of the physical problem
def manual_aggregation(A, dof=[2,2,2], aggrs=[2,2], max_levels=3, dim=2):

    # assuming a (roughly) minimum coarsest-level size for the matrix
    min_coarsest_size = 8

    if max_levels>2:
        raise Exception("FIXME : this code needs to be fixed for levels>2; the aggregation"+
                        " is being done wrong! Spin-wise separation is being done right from the"+
                        " first to the second level, but not correct afterwards")

    # TODO : check what is the actual maximum number of levels possible. For
    #        now, just assume max_levels is possible

    Al = A.copy()

    As = list()
    Ps = list()
    Rs = list()

    # at level 0
    ml = SimpleML()
    ml.levels.append(LevelML())
    ml.levels[0].A = Al.copy()

    print("")

    for i in range(max_levels-1):

        #mat_size = int(Al.shape[0]/2)
        #Al[mat_size:] = -Al[mat_size:]

        print("\teigensolving at level "+str(i)+" ...")
        eigvals,eig_vecs = eigs(Al, k=dof[i+1], which='SM', return_eigenvectors=True, tol=1e-9)
        print("\t... done")

        #mat_size = int(Al.shape[0]/2)
        #Al[mat_size:] = -Al[mat_size:]

        print("\tconstructing P at level "+str(i)+" ...")

        aggr_size = aggrs[i]*aggrs[i]*dof[i]
        aggr_size_half = int(aggr_size/2)
        nr_aggrs = int(Al.shape[0]/aggr_size)

        P_size_n = Al.shape[0]
        P_size_m = nr_aggrs*dof[i+1]*2
        Px = np.zeros((P_size_n,P_size_m), dtype=Al.dtype)

        # this is a for loop over aggregates
        for j in range(nr_aggrs):
            # this is a for loop over eigenvectors
            for k in range(dof[i+1]):
                # this is a for loop over half of the entries, spin 0
                for w in range(aggr_size_half):
                    # even entries
                    aggr_eigvectr_ptr = j*aggr_size+2*w
                    #ii_ptr = j*aggr_size+w
                    ii_ptr = j*aggr_size+2*w
                    jj_ptr = j*dof[i+1]*2+k
                    Px[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]
            # this is a for loop over eigenvectors
            for k in range(dof[i+1]):
                # this is a for loop over half of the entries, spin 1
                for w in range(aggr_size_half):
                    # odd entries
                    aggr_eigvectr_ptr = j*aggr_size+2*w+1
                    #ii_ptr = j*aggr_size+aggr_size_half+w
                    ii_ptr = j*aggr_size+2*w+1
                    jj_ptr = j*dof[i+1]*2+dof[i+1]+k
                    Px[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]

        print("\t... done")

        # ------------------------------------------------------------------------------------
        # perform a per-aggregate orthonormalization - apply plain CGS
        print("\torthonormalizing by aggregate in P at level "+str(i)+" ...")
        # spin 0
        for j in range(nr_aggrs):
            for k in range(dof[i+1]):
                ii_off_1 = j*aggr_size
                #ii_off_2 = ii_off_1+aggr_size_half
                ii_off_2 = ii_off_1+aggr_size
                jj_off = j*dof[i+1]*2
                # vk = Px[ii_off_1:ii_off_2,jj_off+k]
                rs = []
                for w in range(k):
                    rs.append(np.vdot(Px[ii_off_1:ii_off_2,jj_off+w],Px[ii_off_1:ii_off_2,jj_off+k]))
                for w in range(k):
                    Px[ii_off_1:ii_off_2,jj_off+k] -= rs[w]*Px[ii_off_1:ii_off_2,jj_off+w]
                Px[ii_off_1:ii_off_2,jj_off+k] /= sqrt(np.vdot(Px[ii_off_1:ii_off_2,jj_off+k],Px[ii_off_1:ii_off_2,jj_off+k]))
        # spin 1
        for j in range(nr_aggrs):
            for k in range(dof[i+1]):
                #ii_off_1 = j*aggr_size+aggr_size_half
                #ii_off_2 = ii_off_1+aggr_size_half
                ii_off_1 = j*aggr_size
                ii_off_2 = ii_off_1+aggr_size
                jj_off = j*dof[i+1]*2+dof[i+1]
                # vk = Px[ii_off_1:ii_off_2,jj_off+k]
                rs = []
                for w in range(k):
                    rs.append(np.vdot(Px[ii_off_1:ii_off_2,jj_off+w],Px[ii_off_1:ii_off_2,jj_off+k]))
                for w in range(k):
                    Px[ii_off_1:ii_off_2,jj_off+k] -= rs[w]*Px[ii_off_1:ii_off_2,jj_off+w]
                Px[ii_off_1:ii_off_2,jj_off+k] /= sqrt(np.vdot(Px[ii_off_1:ii_off_2,jj_off+k],Px[ii_off_1:ii_off_2,jj_off+k]))
        print("\t... done")
        # ------------------------------------------------------------------------------------

        #Pl2 = csr_matrix(Px, dtype=Px.dtype)
        #write_png(Pl2,"P2_"+str(i)+".png")

        # check Gamma3-compability here !!
        P1 = np.copy(Px)
        mat_size1_half = int(P1.shape[0]/2)
        P1[mat_size1_half:,:] = -P1[mat_size1_half:,:]
        P2 = np.copy(Px)
        mat_size2_half = int(P1.shape[1]/2)
        P2[:,mat_size2_half:] = -P2[:,mat_size2_half:]
        diffP = P1-P2
        print("\tmeasuring g3-compatibility at level "+str(i)+" : "+str( npnorm(diffP,ord='fro') ))

        Pl = csr_matrix(Px, dtype=Px.dtype)

        ml.levels[i].P = Pl.copy()

        print("\tconstructing R at level "+str(i)+" ...")

        # set Rl = Pl^H
        Rl = Pl.copy()
        Rl = Rl.conjugate()
        Rl = Rl.transpose()

        ml.levels[i].R = Rl.copy()

        print("\t... done")

        axx = Rl*Pl
        bxx = identity(Pl.shape[1],dtype=Pl.dtype)
        cxx = axx-bxx
        print("\torthonormality of P at level "+str(i)+" = "+str( norm(axx-bxx,ord='fro')) )

        print("\tconstructing A at level "+str(i+1)+" ...")

        Ax = Rl*Al*Pl
        Al = Ax.copy()

        ml.levels.append(LevelML())
        ml.levels[i+1].A = Al.copy()

        Ax = Ax.getH()
        print("\thermiticity of A at level "+str(i+1)+" = "+str( norm(Ax-Al,ord='fro')) )

        mat_size_half = int(Al.shape[0]/2)
        g3Al = Al.copy()
        g3Al[mat_size_half:,:] = -g3Al[mat_size_half:,:]
        g3Ax = g3Al.copy()
        g3Ax = g3Ax.getH()
        print("\thermiticity of g3*A at level "+str(i+1)+" = "+str( norm(g3Ax-g3Al,ord='fro')) )

        print("\t... done")

        print("")

        if Al.shape[0] <= min_coarsest_size: break

    return ml
