import subprocess
import numpy as np
from scipy.sparse import csr_matrix


# A0 : (4^4)*12 = 3072
# A1 : (2^4)*24*2 = 768
# P1 : 3072x768

nr_levels = 2
sizes_A = [(3072,3072),(768,768)]
sizes_P = [(3072,768)]
#sizes_A = [(4,4),(4,4)]
#sizes_P = [(4,4)]

As = list()
Ps = list()

# do A first
for i in range(nr_levels):

    As.append(np.zeros((sizes_A[i][0],sizes_A[i][1]),dtype='complex128'))

    process = subprocess.check_output(['sh','run','-i','sample_devel.ini','A',str(i)], universal_newlines=True)
    #output = process.stdout.split("\n")
    output = process.split("\n")

    #print(output)
    #exit(0)

    jj_ctr = 0
    for line in output:
        if line[:7]=='PRINTER':
            line_str = line[10:]
            if not line_str[:6]=='val = ': print(line_str)
            if line_str[:6]=='val = ':
                line_val = line_str[6:]
                line_vals = line_val.split(' -- ')
                line_vals.pop()
                for count,valx in enumerate(line_vals):
                    if len(valx.split("+-"))==2: valx = ''.join(valx.split("+"))
                    xxval = complex(valx)
                    As[i][count,jj_ctr] = xxval
                jj_ctr += 1

#print(As[0])
#print(As[1])



# then, do P
for i in range(nr_levels-1):

    Ps.append(np.zeros((sizes_P[i][0],sizes_P[i][1]),dtype='complex128'))

    process = subprocess.check_output(['bash','run','-i','sample_devel.ini','P',str(i)], universal_newlines=True)
    #output = process.stdout.split("\n")
    output = process.split("\n")

    jj_ctr = 0
    for line in output:
        if line[:7]=='PRINTER':
            line_str = line[10:]
            if not line_str[:6]=='val = ': print(line_str)
            if line_str[:6]=='val = ':
                line_val = line_str[6:]
                line_vals = line_val.split(' -- ')
                line_vals.pop()
                for count,valx in enumerate(line_vals):
                    if len(valx.split("+-"))==2: valx = ''.join(valx.split("+"))
                    xxval = complex(valx)
                    Ps[i][count,jj_ctr] = xxval
                jj_ctr += 1

#print(Ps[0])

# ---------------------------

import scipy.io

# A , 0
Mx = csr_matrix(As[0], dtype=As[0].dtype)
scipy.io.savemat('Af.mat', mdict={'Af': Mx})

# A , 1
Mx = csr_matrix(As[1], dtype=As[1].dtype)
scipy.io.savemat('Ac.mat', mdict={'Ac': Mx})

# P , 0
Mx = csr_matrix(Ps[0], dtype=Ps[0].dtype)
scipy.io.savemat('P.mat', mdict={'P': Mx})
