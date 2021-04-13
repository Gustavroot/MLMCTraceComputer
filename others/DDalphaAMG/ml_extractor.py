import subprocess
import numpy as np
from scipy.sparse import csr_matrix
import scipy.io


# A0 : (4^4)*12 = 3072
# A1 : (2^4)*24*2 = 768
# P1 : 3072x768

nr_levels = 2

sizes_A = [(49152,49152)]
#sizes_A = [(3072,3072),(768,768)]

#sizes_Q = [(49152,49152),(768,768)]
#sizes_P = [(49152,768)]
#sizes_Q = [(3072,3072),(1536,1536)]
#sizes_P = [(49152,768)]

#sizes_A = [(4,4),(4,4)]
#sizes_P = [(4,4)]

As = list()
Qs = list()
Ps = list()

# do A first
for i in range(nr_levels):

    if i>0 : break

    print("----------")
    print("A, level="+str(i))
    print("----------")

    As.append(np.zeros((sizes_A[i][0],sizes_A[i][1]),dtype='complex128'))

    if i==0 : round_size = 3072
    else : round_size = 768
    rounds = int(sizes_A[i][1]/round_size)
    for j in range(rounds):

        print("Running DD-alphaAMG ... ("+str(j*round_size)+" to "+str((j+1)*round_size)+")")
        process = subprocess.check_output(['sh','run','-i','sample_devel.ini','A',str(i),str(j*round_size),str((j+1)*round_size)], universal_newlines=True)
        print("... done")
        output = process.split("\n")

        print("Parsing data ...")
        jj_ctr = 0
        for line in output:
            if line[:7]=='PRINTER':
                line_str = line[10:]
                if not line_str[:6]=='val = ': print(line_str)
                if line_str[:6]=='val = ':
                    line_val = line_str[6:]
                    line_val = '-'.join(line_val.split("+-"))
                    line_vals = line_val.split(' -- ')
                    line_vals.pop()

                    line_vals2 = [complex(elm) for elm in line_vals]
                    line_vals = np.array(line_vals2).reshape(-1)
                    As[i][:,jj_ctr+j*round_size] = line_vals[:]

                    jj_ctr += 1
        print("... done")

    print("Packing into MATLAB sparse format ...")
    # A , level i
    Mx = csr_matrix(As[i], dtype=As[i].dtype)
    scipy.io.savemat('LQCD_A'+str(i+1)+'.mat', mdict={'A'+str(i+1): Mx})
    print("... done")


"""
# then, do Q
for i in range(nr_levels):

    #if i==0 : continue

    print("----------")
    print("Q, level="+str(i))
    print("----------")

    Qs.append(np.zeros((sizes_Q[i][0],sizes_Q[i][1]),dtype='complex128'))

    if i==0 : round_size = 1024
    else : round_size = 256
    rounds = int(sizes_Q[i][1]/round_size)
    for j in range(rounds):

        print("Running DD-alphaAMG ... ("+str(j*round_size)+" to "+str((j+1)*round_size)+")")
        process = subprocess.check_output(['sh','run','-i','sample_devel.ini','Q',str(i),str(j*round_size),str((j+1)*round_size)], universal_newlines=True)
        print("... done")
        output = process.split("\n")

        print("Parsing data ...")
        jj_ctr = 0
        for line in output:
            if line[:7]=='PRINTER':
                line_str = line[10:]
                if not line_str[:6]=='val = ': print(line_str)
                if line_str[:6]=='val = ':
                    line_val = line_str[6:]
                    line_val = '-'.join(line_val.split("+-"))
                    line_vals = line_val.split(' -- ')
                    line_vals.pop()

                    line_vals2 = [complex(elm) for elm in line_vals]
                    line_vals = np.array(line_vals2).reshape(-1)
                    Qs[i][:,jj_ctr+j*round_size] = line_vals[:]

                    jj_ctr += 1
        print("... done")

    print("Packing into MATLAB sparse format ...")
    # Q , level i
    Mx = csr_matrix(Qs[i], dtype=Qs[i].dtype)
    scipy.io.savemat('LQCD_Q'+str(i+1)+'.mat', mdict={'Q'+str(i+1): Mx})
    print("... done")
"""



"""
# and finally, do P
for i in range(nr_levels-1):

    print("----------")
    print("P, level="+str(i))
    print("----------")

    Ps.append(np.zeros((sizes_P[i][0],sizes_P[i][1]),dtype='complex128'))

    if i==0 : round_size = 768
    else : round_size = 768
    rounds = int(sizes_P[i][1]/round_size)
    for j in range(rounds):

        print("Running DD-alphaAMG ... ("+str(j*round_size)+" to "+str((j+1)*round_size)+")")
        process = subprocess.check_output(['bash','run','-i','sample_devel.ini','P',str(i),str(j*round_size),str((j+1)*round_size)], universal_newlines=True)
        print("... done")
        #output = process.stdout.split("\n")
        output = process.split("\n")

        print("Parsing data ...")
        jj_ctr = 0
        for line in output:
            if line[:7]=='PRINTER':
                line_str = line[10:]
                if not line_str[:6]=='val = ': print(line_str)
                if line_str[:6]=='val = ':
                    line_val = line_str[6:]
                    line_val = '-'.join(line_val.split("+-"))
                    line_vals = line_val.split(' -- ')
                    line_vals.pop()

                    line_vals2 = [complex(elm) for elm in line_vals]
                    line_vals = np.array(line_vals2).reshape(-1)
                    Ps[i][:,jj_ctr+j*round_size] = line_vals[:]

                    jj_ctr += 1
        print("... done")

    print("Packing into MATLAB sparse format ...")
    Mx = csr_matrix(Ps[i], dtype=Ps[i].dtype)
    scipy.io.savemat('LQCD_P'+str(i+1)+'.mat', mdict={'P'+str(i+1): Mx})
    print("... done")
"""


#print(Ps[0])

# ---------------------------

#import scipy.io

#print("Packing into MATLAB sparse format ...")
# A , 0
#Mx = csr_matrix(As[0], dtype=As[0].dtype)
#scipy.io.savemat('LQCD_A1.mat', mdict={'A1': Mx})

# A , 1
#Mx = csr_matrix(As[1], dtype=As[1].dtype)
#scipy.io.savemat('LQCD_A2.mat', mdict={'A2': Mx})

# Q , 0
#Mx = csr_matrix(Qs[0], dtype=Qs[0].dtype)
#scipy.io.savemat('LQCD_Q1.mat', mdict={'Q1': Mx})

# Q , 1
#Mx = csr_matrix(Qs[1], dtype=Qs[1].dtype)
#scipy.io.savemat('LQCD_Q2.mat', mdict={'Q2': Mx})

# P , 0
#Mx = csr_matrix(Ps[0], dtype=Ps[0].dtype)
#scipy.io.savemat('LQCD_P1.mat', mdict={'P1': Mx})
#print("... done")
