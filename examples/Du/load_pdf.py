from vkpp import Stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import time
import itertools
import pandas as pd
import ast
import sys

# plotting params
mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.linewidth'] = 1.5   
N = 256
colors = [(0,0,1,0.75), (0,0,0.5), (0,1,1), (0,0.5,0), (1,1,0), (1,0.5,0), (0.5,0,0), (1,0,0,0.75)]
cmap_name = 'btr_rainbow'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=N)
plt.register_cmap(cmap=cmap)

# slurm arrays params
calcs = ['corr', 'joint']
stats = ['full', 'penal', 'interior', 'bulk']
fields = ['Du_l1', 'Du_2']
runs = list(itertools.product(range(2), range(4), range(2)))
# run set
a = int(sys.argv[1])-1
i,j,k = runs[a] # field - stat - calc
field = fields[i]
stat = stats[j]
calc = calcs[k]

# arrays dictionnaries
specs = {'corr': {'cmap':'gist_ncar', 'qty': r'\mathcal{C}'}, \
         'joint': {'cmap': 'btr_rainbow', 'qty': r'f'}}
scales = {'Du_l1': r'\ell=1.06\eta', 'Du_2': r'\ell=26.5\eta'}

real_lims = {'Du_l1': {stat: [-30, 40] for stat in stats},\
             'Du_2': {stat: [-20,35] for stat in stats},\
             'omega': {stat: [0, 30] for stat in stats}}
real_lims['Du_l1']['interior'] = [-25,35]
real_lims['Du_l1']['bulk'] = [-5,8]
real_lims['Du_2']['interior'] = [-20, 35]
real_lims['Du_2']['bulk'] = [-5,15]
real_lims['omega']['interior'] = [0,25]
real_lims['omega']['bulk'] = [0,9.5]
# these limits have been found empirically AFTER visualization
# for most fields, the real limits are almost similar between full, penal and interior

cm = specs[calc]['cmap']
qty = specs[calc]['qty']
scale = scales[field]
lim_x = real_lims[field][stat]
lim_y = real_lims['omega'][stat]

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

# Loading db
db_path = '/mnt/beegfs/projects/sfemans_ai/connectors/data_hugues.pkl'
pp = Stats(db_path, stat)

edges = pp.compute_edges(bins, ranges)
X, Y = np.meshgrid(edges[0], edges[1])

joints = []
for n in range(bags):
    hist = np.fromfile('%s/%s/iter_%d.dat' % (field, stat, n))
    joint = pp.load_reshaped(hist, bins)
    joints.append(joint)

# full hist
full = (np.sum(np.array(joints), axis=0)/bags)
if calc == 'corr':
    full = pp.correlations(full, bins, ranges)
else:
    full = np.log10(full)

# finding visible colorbar range
full[full == -np.inf] = np.nan
n_bins, v = np.histogram(full[~np.isnan(full)], bins=100)
cutoff = bins
vmin = v[:-1][n_bins>cutoff][0]
if calc == 'corr':
    vmax = v[1:][::-1][(n_bins>cutoff)[::-1]][0]
    print('vmax = %f' % vmax)
print('vmin = %f' % vmin)

# Plotting
if calc == 'joint':
    fH = plt.pcolormesh(X, Y, full, vmin=vmin)
else:
    divnorm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    fH = plt.pcolormesh(X, Y, full, norm=divnorm)

plt.xlabel(r'$\frac{\mathscr{D}^u}{\epsilon}$')
plt.ylabel(r'$\frac{\omega}{\left\langle\omega\right\rangle}$')
cbar = plt.colorbar(fH)
cbar.set_label(r'$\log_{10}\left(%s\right)$' % qty)
plt.set_cmap(cm)
plt.xlim(lim_x)
plt.ylim(lim_y)

plt.title(r'$%s\left(\frac{\mathscr{D}^u}{\epsilon}, ' %qty+ \
          r'\frac{\omega}{\left\langle\omega\right\rangle}\right)$ (%s)' %stat)
plt.savefig('figs/%s/%s_%s_omega_%s.jpg' %(field, calc, field, stat), dpi=1200)