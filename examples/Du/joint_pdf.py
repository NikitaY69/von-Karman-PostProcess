import numpy as np
import dask.array as da
import time
import itertools
import pandas as pd
import ast
import sys

# checking the device
try:
    import cupy as cp
    cp.array([0])
    gpu = True
    from vkcupp import Stats
except:
    gpu = False
    from vkpp import Stats

# slurm arrays params
stats = ['full', 'penal', 'interior', 'bulk']
fields = ['Du_l1', 'Du_2']
runs = list(itertools.product(range(4), range(2)))
# run set
a = int(sys.argv[1])-1
i,j = runs[a]
stat = stats[i]
field = fields[j]

# run params
params = pd.read_csv('misc/miscs.csv')
params.set_index('Stat', inplace=True)

# omega
omega_p = ast.literal_eval(params.loc[stat]['omega'])
w_avg = omega_p[0]
w_range = np.array(omega_p[1])
# field
field_p = ast.literal_eval(params.loc[stat][field])
f_avg = field_p[0]
f_range = np.array(field_p[1])

epsilon = 0.045
bags = 28
bins = 2000
ranges = [f_range/epsilon, w_range/w_avg]
s = slice(None)

# Loading db
db_path = '/mnt/beegfs/projects/sfemans_ai/connectors/data_hugues.pkl'
pp = Stats(db_path, stat)                    # getting the PostProcessing module
data = pp.db                                   # database dictionnary
P, N = pp.dims[2], pp.dims[3]                  # p,r dimensions in data struct

# Defining the rechunked+delayed fields
f = data[field]['field'].rechunk((1,1,P,N//14))          
w = data['omega']['field'].rechunk((3,1,P,N//14)) 

# Sequential workloads
t00 = time.time()
for t in range(bags):
    print('####################################################################')
    print('Frame %d' %t)
    t0 = time.time()

    if stat == 'penal':
        ti = time.time()
        pp.set_penal(s, [t], s, s)
        print('Computed penal in %f s' % (time.time()-ti))

    ti = time.time()
    f_c = f[:, [t], :, :].compute()
    print('Loaded %s in %f s' % (field, (time.time()-ti)))
    if gpu:
        ti = time.time()
        f_c = cp.array(f_c, copy=False)
        print('Transfered field to CUDA in %f' % (time.time()-ti))
    
    ti = time.time()
    w_c = w[:, [t], :, :].compute()
    print('Loaded omega in %f s' % (time.time()-ti))
    if gpu:
        ti = time.time()
        w_c = cp.array(w_c, copy=False)
        print('Transfered field to CUDA in %f' % (time.time()-ti))

    ti = time.time()
    w_c = pp.norm(w_c)
    print('Computed norm in %f s' % (time.time()-ti))

    ti = time.time()
    joint, _ = pp.joint_pdf(f_c/epsilon, w_c/w_avg, bins=bins, ranges=ranges, \
                            log=False, save='%s/%s/iter_%d.dat' % (field, stat, t))
    print('Computed joint pdf in %f s' % (time.time()-ti))

    print('Computed everything in %f s' % (time.time()-t0))

print('\n########################################################################')
print('Computed over %d frames in %f s' % (bags, (time.time()-t00)))