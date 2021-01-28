from pyamg.aggregation.adaptive import adaptive_sa_solver
from scipy.sparse.linalg import eigs
from math import pow
import numpy as np
from scipy.sparse import csr_matrix




# <dof> :   per level (except the last one, of course), this is a list of
#           the number of degrees of freedom for the next level
# <aggrs> : per level, this is the block size. So, if at a certain level the
#           value is 4, then the aggregates are of size 4^d, where d is the
#           dimensionality of the physical problem
def manual_aggregation(A, dof=[2,2,2], aggrs=[2,2], max_levels=3, dim=2):

    # TODO : check what is the actual maximum number of levels possible. For
    #        now, just assume max_levels is possible

    Al = A.copy()

    As = list()
    Ps = list()
    Rs = list()

    As.append(A.copy())

    for i in range(max_levels-1):

        print("\teigensolving at level "+str(i)+" ...")
        # TODO : extract actual eigenvalues of Al
        #eigvals,eig_vecs = eigs(Al, k=dof[i+1], which='SM', return_eigenvectors=True, tol=1e-1)
        eig_vecs = np.zeros((Al.shape[0],dof[i+1]))
        print("\t... done")

        print("\tconstructing P at level "+str(i)+" ...")

        aggr_size = aggrs[i]*aggrs[i]*dof[i]
        aggr_size_half = int(aggr_size/2)
        nr_aggrs = int(Al.shape[0]/aggr_size)

        #print(aggr_size)
        #print(nr_aggrs)
        #print(Al.shape)

        P_size_n = Al.shape[0]
        P_size_m = nr_aggrs*dof[i+1]*2
        #Pl = csr_matrix((P_size_n,P_size_m), dtype=Al.dtype)
        Pl = np.zeros((P_size_n,P_size_m), dtype=Al.dtype)

        # this is a for loop over aggregates
        for j in range(nr_aggrs):

            # this is a for loop over eigenvectors
            for k in range(dof[i+1]):

                # this is a for loop over half of the entries, spin 0
                for w in range(aggr_size_half):
                    # even entries
                    aggr_eigvectr_ptr = j*aggr_size+2*w

                    ii_ptr = j*aggr_size+w
                    jj_ptr = j*dof[i+1]*2+k

                    Pl[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]

            # this is a for loop over eigenvectors
            for k in range(dof[i+1]):

                # this is a for loop over half of the entries, spin 1
                for w in range(aggr_size_half):
                    # odd entries
                    aggr_eigvectr_ptr = j*aggr_size+2*w+1

                    ii_ptr = j*aggr_size+aggr_size_half+w
                    jj_ptr = j*dof[i+1]*2+dof[i+1]+k

                    Pl[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]

        print("\t"+str(Pl.shape))

        print("\t... done")

        print("\tconstructing R at level "+str(i)+" ...")

        # set Rl = Pl^H
        #Rl = Pl.copy()
        #Rl = Rl.conjugate()
        #Rl = Rl.transpose()

        #print("\t"+str(Rl.shape))

        print("\t... done")

        print("\tconstructing A at level "+str(i+1)+" ...")

        #Ax = Rl*Al*Pl
        #Al = Ax.copy()

        print("\t"+str(Al.shape))

        print("\t... done")

        #break

    #[ml, work] = adaptive_sa_solver(A, num_candidates=2, candidate_iters=2, improvement_iters=3,
    #                                strength='symmetric', aggregate='standard', max_levels=max_levels)

    return 0
