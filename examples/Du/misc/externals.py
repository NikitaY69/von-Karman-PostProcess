import numpy as np
import dask.array as da
import time
import itertools
import pandas as pd
import ast
import sys
import pickle 

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
fields = ['Du_l1', 'Du_2', 'omega']
runs = list(itertools.product(range(4), range(3)))
# run set
a = int(sys.argv[1])-1
i,j = runs[a]
stat = stats[i]
field = fields[j]

# creating saving file
try:
    # loading
    pd.read_csv('miscs.csv', index_col='Stat')
except FileNotFoundError:    
    # creating vanilla file
        d = {field:[[None, None] for _ in range(len(stats))] for field in fields}
        d['Stat'] = stats
        params = pd.DataFrame(d)
        params.to_csv('miscs.csv', index=False)

# run globals
bags = 28
s = slice(None)

# loading db
db_path = '/mnt/beegfs/projects/sfemans_ai/connectors/data_hugues.pkl'
pp = Stats(db_path, stat)
data = pp.db

d = data[field]['field'].shape[0]
f = data[field]['field'].rechunk((d, 1, pp.dims[2], pp.dims[3]//14))

# run
ranges = []
avgs = []
t00 = time.time()
for t in range(2):
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
    
    if d == 3:
        ti = time.time()
        f_c = pp.norm(f_c)
        print('Computed norm in %f s' % (time.time()-ti))

    ti = time.time()
    t_range = pp.get_range(f_c)
    ranges.append(t_range)
    print('Computed range in %f' % (time.time()-ti))

    ti = time.time()
    t_avg = pp.mean(f_c, type='both')
    avgs.append(t_avg)
    print('Computed mean in %f s' % (time.time()-ti))

    print('Computed everything in %f s' % (time.time()-t0))

if gpu:
    avgs = [v.get() for v in avgs]
ranges = np.ravel(np.array(ranges))
tot_range = [ranges.min(), ranges.max()]
avg = np.sum(avgs)/bags

# updating saving file
params = pd.read_csv('miscs.csv', index_col='Stat')
params.loc[stat][field] = ast.literal_eval(params.loc[stat][field])
params.loc[stat][field][0] = avg
params.loc[stat][field][1] = tot_range
params.to_csv('miscs.csv', mode='r+')

print('\n########################################################################')
print('%s_min, %s_min = %f, %f' % (field, field, tot_range[0], tot_range[1]))
print('%s avg = %f' % (field, avg))
print('########################################################################')
print('Computed over %d frames in %f s' % (bags, (time.time()-t00)))