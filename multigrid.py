import numpy as np
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import fgmres

from pyamg.aggregation.adaptive import adaptive_sa_solver
from scipy.sparse.linalg import eigsh, eigs
from math import pow, sqrt
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import norm
from numpy.linalg import norm as npnorm

ml = []
level_nr = 0
total_levels = 0
coarsest_iters = 0
coarsest_iters_tot = 0
coarsest_iters_avg = 0
nr_calls = 0
smoother_iters = 2

coarsest_lev_iters = [0,0,0,0,0,0,0,0,0,0]







class LevelML:
    R = 0
    P = 0
    A = 0
    Q = 0

class SimpleML:
    levels = []

    def __str__(self):
        for idx,level in enumerate(self.levels[:-1]):
            print("Level: "+str(idx))
            print("\tsize(R) = "+str(level.R.shape))
            print("\tsize(P) = "+str(level.P.shape))
            print("\tsize(A) = "+str(level.A.shape))
















def one_mg_step( b ):

    global ml
    global level_nr
    global total_levels
    global coarsest_iters
    global coarsest_iters_tot
    global coarsest_iters_avg
    global nr_calls
    #global smoother_iters

    global coarsest_lev_iters

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
        e, exitCode = lgmres( ml.levels[i+level_nr].A,rs[i],tol=1.0e-20,maxiter=smoother_iters )
        # 3. update solution
        xs[i] += e
        # 4. update residual
        rs[i] = bs[i]-ml.levels[i+level_nr].A*xs[i]
        # 5. restrict residual
        bs[i+1] = ml.levels[i+level_nr].R*rs[i]

    # coarsest level solve
    #print(coarsest_iters_tot)
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    xs[i], exitCode = lgmres( ml.levels[i+level_nr].A,bs[i],tol=1.0e-4,callback=callback )
    # FIXME : number 30 hardcoded
    #xs[i], exitCode = lgmres( ml.levels[i+level_nr].A,bs[i],tol=1.0e-30,callback=callback,maxiter=30 )
    coarsest_lev_iters[level_nr] += num_iters
    #print(num_iters)
    coarsest_iters = num_iters
    nr_calls += 1
    coarsest_iters_tot += coarsest_iters
    #print(coarsest_iters_tot)
    #print("")
    coarsest_iters_avg = coarsest_iters_tot/nr_calls

    # go up in the V-cycle
    for i in range(level_id-2,-1,-1):
        # 1. interpolate and update
        xs[i] += ml.levels[i+level_nr].P*xs[i+1]
        # 2. build the residual
        rs[i] = bs[i]-ml.levels[i+level_nr].A*xs[i]
        # 3. smooth
        e, exitCode = lgmres( ml.levels[i+level_nr].A,rs[i],tol=1.0e-20,maxiter=smoother_iters )
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




# ----------------------------------------------------------------------------------------------



# solver class
class MG:

    def __init__(self,A,smooth_iters=2):
        # This parameter changes in MLMC, and it indicates the level from which
        # we want to perform a solve
        self.level_nr = 0
        # This is the multigrid hierarchy, of type SimpleML
        self.ml = []
        self.A = A
        # Solution from the last call to the solver
        self.x = []
        self.num_iters = 0

        self.total_levels = 0
        self.coarsest_iters = 0
        self.coarsest_iters_tot = 0
        self.coarsest_iters_avg = 0
        self.nr_calls = 0

        self.smooth_iters = smooth_iters
        
        self.coarsest_lev_iters = [0,0,0,0,0,0,0,0,0,0]

    # <dof> :   per level (except the last one, of course), this is a list of
    #           the number of degrees of freedom for the next level
    # <aggrs> : per level, this is the block size. So, if at a certain level the
    #           value is 4, then the aggregates are of size 4^d, where d is the
    #           dimensionality of the physical problem
    def setup(self,dof=[2,4,4],aggrs=[2*2,2*2],max_levels=3,dim=2,acc_eigvs='low',sys_type='schwinger'):

        # assuming a (roughly) minimum coarsest-level size for the matrix
        min_coarsest_size = 1

        # TODO : check what is the actual maximum number of levels possible. For
        #        now, just assume max_levels is possible

        Al = self.A.copy()

        As = list()
        Ps = list()
        Rs = list()

        # at level 0
        ml = SimpleML()
        ml.levels.append(LevelML())
        ml.levels[0].A = Al.copy()

        print("")

        for i in range(max_levels-1):

            # use Q
            #mat_size = int(Al.shape[0]/2)
            #Al[mat_size:] = -Al[mat_size:]

            print("\tNonzeros = "+str(Al.count_nonzero()))
            print("\tsize(A) = "+str(Al.shape))
    
            print("\teigensolving at level "+str(i)+" ...")
    
            nt = 1

            if acc_eigvs == 'low':
                tolx = tol=1.0e-1
                ncvx = nt*dof[i+1]+2
            elif acc_eigvs == 'high':
                tolx = tol=1.0e-5
                ncvx = None
            else:
                raise Exception("<accuracy_mg_eigvs> does not have a possible value.")
    
            # FIXME : hardcoded value for eigensolving tolerance for now
            tolx = 1.0e-5
    
            #eigvals,eig_vecsx = eigsh(Al, k=nt*dof[i+1], which='SM', return_eigenvectors=True, tol=1e-5, maxiter=1000000)
            #eigvals,eig_vecsx = eigs(Al, k=nt*dof[i+1], which='SM', return_eigenvectors=True, tol=1e-2, maxiter=1000000)

            #eigvals,eig_vecsx = eigsh( Al, k=1, which='SR', tol=tolx, maxiter=1000000 )
            #print(eigvals)
            #exit(0)

            if i<3:
                eigvals,eig_vecsx = eigs( Al, k=nt*dof[i+1], which='LM', tol=tolx, maxiter=1000000, sigma=0.0, ncv=ncvx )
                #eigvals,eig_vecsx = eigsh( Al, k=nt*dof[i+1], which='SM', tol=tolx, maxiter=1000000 )
            else:
                #eigvals,eig_vecsx = eigs( Al, k=nt*dof[i+1], which='LM', tol=1.0e-5, maxiter=1000000, sigma=0.0 )
                eigvals,eig_vecsx = eigs( Al, k=nt*dof[i+1], which='LM', tol=tolx, maxiter=1000000, sigma=0.0 )
                #eigvals,eig_vecsx = eigsh( Al, k=nt*dof[i+1], which='SM', tol=1.0e-5, maxiter=1000000 )

            #print(eigvals)
            #exit(0)

            eig_vecs = np.zeros((Al.shape[0],dof[i+1]), dtype=Al.dtype)

            #coeffs = [ 1.0/float(k+1) for k in range(nt) ]
            coeffs = [ 1.0 for k in range(nt) ]

            for j in range(dof[i+1]):
                for k in range(nt):
                    eig_vecs[:,j] += eig_vecsx[:,nt*j+k]
                    #eig_vecs[:,j] += coeffs[k]*eig_vecsx[:,j+dof[i+1]*k]

            print("\t... done")

            # use Q
            #mat_size = int(Al.shape[0]/2)
            #Al[mat_size:] = -Al[mat_size:]

            print("\tconstructing P at level "+str(i)+" ...")

            #aggr_size = aggrs[i]*aggrs[i]*dof[i]

            if i==0 : aggr_size = aggrs[i]*dof[i]
            else : aggr_size = aggrs[i]*dof[i]*2

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
                    for w in range(int(aggr_size_half/(dof[i]/2))):
                        for z in range(int(dof[i]/2)):
                            # even entries
                            aggr_eigvectr_ptr = j*aggr_size+w*dof[i]+z
                            #ii_ptr = j*aggr_size+w
                            #ii_ptr = j*aggr_size+2*w
                            ii_ptr = j*aggr_size+w*dof[i]+z
                            jj_ptr = j*dof[i+1]*2+k
                            Px[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]

                # this is a for loop over eigenvectors
                for k in range(dof[i+1]):
                    # this is a for loop over half of the entries, spin 1
                    for w in range(int(aggr_size_half/(dof[i]/2))):
                        for z in range(int(dof[i]/2)):
                            # odd entries
                            aggr_eigvectr_ptr = j*aggr_size+w*dof[i]+int(dof[i]/2)+z
                            #ii_ptr = j*aggr_size+aggr_size_half+w
                            ii_ptr = j*aggr_size+w*dof[i]+int(dof[i]/2)+z
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

            Pl = csr_matrix(Px, dtype=Px.dtype)

            ml.levels[i].P = Pl.copy()

            print("\tconstructing R at level "+str(i)+" ...")

            # set Rl = Pl^H
            Rl = Pl.copy()
            Rl = Rl.conjugate()
            Rl = Rl.transpose()

            print("\t... done")
    
            ml.levels[i].R = Rl.copy()
    
            Ax = Rl*Al*Pl
            Al = Ax.copy()

            ml.levels.append(LevelML())
            ml.levels[i+1].A = Al.copy()

            """
            if sys_type=='schwinger':

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

                axx = Rl*Pl
                bxx = identity(Pl.shape[1],dtype=Pl.dtype)
                cxx = axx-bxx
                print("\torthonormality of P at level "+str(i)+" = "+str( norm(axx-bxx,ord='fro')) )

                print("\tconstructing A at level "+str(i+1)+" ...")

                Ax = Ax.getH()
                print("\thermiticity of A at level "+str(i+1)+" = "+str( norm(Ax-Al,ord='fro')) )

                mat_size_half = int(Al.shape[0]/2)
                g3Al = Al.copy()
                g3Al[mat_size_half:,:] = -g3Al[mat_size_half:,:]
                g3Ax = g3Al.copy()
                g3Ax = g3Ax.getH()
                print("\thermiticity of g3*A at level "+str(i+1)+" = "+str( norm(g3Ax-g3Al,ord='fro')) )

                print("\t... done")

                if Al.shape[0] <= min_coarsest_size: break

            print("")
            """

        print("\tNonzeros = "+str(Al.count_nonzero()))
        print("\tsize(A) = "+str(Al.shape))

        # creating Q -- Schwinger specific
        #for i in range(len(ml.levels)):
        #    half_size = int(ml.levels[i].A.shape[0]/2)
        #    ml.levels[i].Q = ml.levels[i].A.copy()
        #    ml.levels[i].Q[mat_size_half:,:] = -ml.levels[i].Q[mat_size_half:,:]

        self.ml = ml













    def solve(self,A,b,tol):

        num_iters = 0
        def callback(xk):
            nonlocal num_iters
            num_iters += 1

        if A.shape[0]<1000:
            maxiter = A.shape[0]
        else:
            maxiter = 1000

        lop = LinearOperator(A.shape, matvec=self.one_mg_step)
        self.x,exitCode = fgmres(A,b,tol=tol,M=lop,callback=callback,maxiter=maxiter)

        self.num_iters = num_iters






    def one_mg_step(self,b):

        #global ml
        #global level_nr
        #global total_levels
        #global coarsest_iters
        #global coarsest_iters_tot
        #global coarsest_iters_avg
        #global nr_calls
        #global smoother_iters    
        #global coarsest_lev_iters

        level_id = self.total_levels-self.level_nr

        rs = [ np.zeros(self.ml.levels[i].A.shape[0],dtype=self.ml.levels[i].A.dtype) for i in range(self.level_nr,self.total_levels) ]
        bs = [ np.zeros(self.ml.levels[i].A.shape[0],dtype=self.ml.levels[i].A.dtype) for i in range(self.level_nr,self.total_levels) ]
        xs = [ np.zeros(self.ml.levels[i].A.shape[0],dtype=self.ml.levels[i].A.dtype) for i in range(self.level_nr,self.total_levels) ]

        bs[0][:] = b[:]

        # go down in the V-cycle
        for i in range(level_id-1):
            # 1. build the residual
            rs[i] = bs[i]-self.ml.levels[i+self.level_nr].A*xs[i]
            # 2. smooth
            e, exitCode = lgmres( self.ml.levels[i+self.level_nr].A,rs[i],tol=1.0e-20,maxiter=self.smooth_iters )
            # 3. update solution
            xs[i] += e
            # 4. update residual
            rs[i] = bs[i]-self.ml.levels[i+self.level_nr].A*xs[i]
            # 5. restrict residual
            bs[i+1] = self.ml.levels[i+self.level_nr].R*rs[i]
    
        # coarsest level solve
        #print(coarsest_iters_tot)
        num_iters = 0
        def callback(xk):
            nonlocal num_iters
            num_iters += 1
        xs[i], exitCode = lgmres( self.ml.levels[i+self.level_nr].A,bs[i],tol=1.0e-4,callback=callback )
        self.coarsest_lev_iters[self.level_nr] += num_iters
        self.coarsest_iters = num_iters
        self.nr_calls += 1
        self.coarsest_iters_tot += self.coarsest_iters
        self.coarsest_iters_avg = self.coarsest_iters_tot/self.nr_calls
    
        # go up in the V-cycle
        for i in range(level_id-2,-1,-1):
            # 1. interpolate and update
            xs[i] += self.ml.levels[i+self.level_nr].P*xs[i+1]
            # 2. build the residual
            rs[i] = bs[i]-self.ml.levels[i+self.level_nr].A*xs[i]    
            # 3. smooth    
            e, exitCode = lgmres( self.ml.levels[i+self.level_nr].A,rs[i],tol=1.0e-20,maxiter=self.smooth_iters )
            # 4. update solution
            xs[i] += e
    
        return xs[0]






























    def __str__(self):
        str_out = ""
        for idx,level in enumerate(self.ml.levels):
            str_out += "Level: "+str(idx)+"\n"
            if idx<(len(self.ml.levels)-1): str_out += "\tsize(R) = "+str(level.R.shape)+"\n"
            if idx<(len(self.ml.levels)-1): str_out += "\tsize(P) = "+str(level.P.shape)+"\n"
            str_out += "\tsize(A) = "+str(level.A.shape)+"\n"
        return str_out
