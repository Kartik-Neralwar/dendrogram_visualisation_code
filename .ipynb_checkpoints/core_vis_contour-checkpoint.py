# %matplotlib widget
#contour of cores on imshow density plot -- gridded using numpy hist
import time
from yt.units import *
start_time = time.time()
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm, ListedColormap, rgb2hex
import pandas as pd
import time
import mpld3
import matplotlib
import sys

dust_to_gas = 0.01
mol_hydrogen_ratio = 2.0
microturbulence_speed = 1e4 # cgs
gamma = 5.0/3.0 # Note gamma is not constant and this function is an approximation.
helium_mass_fraction = 0.284 # Default mass fraction in Gizmo

# number fraction of target species
molecular_abundance = 1.5 * 10**-6 # abundance of 13CO # for now : AS per casi 3d paper. old assumption: We consider that the gas is H2 and then obtain the 13CO abundance using the $\frac{12C}{13C}$ = 60 and $\frac{CO}{H2}$ = 10$^{-4}$ factors specified in SEDIGISM 2017 paper.

# Mask abundance based on accreted particles
mask_abundance = False

snap = 10 #snapshot number #int(sys.argv[1])
bins = 1000 #resolution
x_range = (20,60) #ranges for simulation box --> centered at 50,50,50

proj_ax = 2 #axis along which to project

plt.rcParams['legend.title_fontsize'] = 24 #legend's title
hdf_path = '/u/kneralwar/ptmp_link/starforge/M2e4_alpha2_fiducial/snapshot_'+str(snap).zfill(3)+'.hdf5' #path to the snapshot hdf5 file

f = h5py.File(hdf_path, 'r')
data=f['PartType0']

h2dens = data['Density'][()] * data['MolecularMassFraction'][()] * data['NeutralHydrogenAbundance'][()] *(1-helium_mass_fraction)/(mol_hydrogen_ratio*mh)
h2dens = h2dens / 1.47712717e+22 #conversion factor taken from yt by dividing the h2 density here and obtained using yt
x = data['Coordinates'][()][:,0]
y = data['Coordinates'][()][:,1]
z = data['Coordinates'][()][:,2]

data_path2 = '/u/kneralwar/ptmp_link/dendrogram_cores_starforge/M2e4a2_fiducial/_dendrograms/' #path to dendrogram fits files
data2 = fits.getdata(data_path2 + 'M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42_snapshot_'+str(snap).zfill(3)+'_min_val1e3_res1e-3.fits', 1) #dendrogram fits data
indx_map2 = fits.getdata(data_path2 + 'M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42_snapshot_'+str(snap).zfill(3)+'_min_val1e3_res1e-3.fits', 2) #dendrogram mask
indx_map = indx_map2.astype('float64')

core_table = pd.read_pickle('/u/kneralwar/ptmp_link/dendrogram_cores_starforge/M2e4a2_fiducial/_summary_files/M2e4a2_ffmass_coord_bins_2024_05_03.pkl') #table of core properties with core ids
snapshot = (core_table['ID']).str[:5].astype(int)
core_id_1 = (core_table['ID']).str[5:].astype(int)
radius = core_table['Reff [pc]'][snapshot == snap]
x_core_table = core_table['x'][snapshot == snap]
y_core_table = core_table['y'][snapshot == snap]
z_core_table = core_table['z'][snapshot == snap]
global_bin = core_table['global_bin']

core_id = core_id_1[(snapshot == snap) & ((global_bin == 0) | (global_bin == 1) | (global_bin == 2) | (global_bin == 3))]
global_bin_new = global_bin[(snapshot == snap) & ((global_bin == 0) | (global_bin == 1) | (global_bin == 2) | (global_bin == 3))]


core_id_n = core_id_1[(snapshot == snap) & (global_bin == 0)]
core_id_l = core_id_1[(snapshot == snap) & (global_bin == 1)]
core_id_m = core_id_1[(snapshot == snap) & (global_bin == 2)]
core_id_h = core_id_1[(snapshot == snap) & (global_bin == 3)]
# core_id_n = core_id_1[(snapshot == snap) & (global_bin == 4)]


dens_g, edges = np.histogramdd((x,y,z), bins = bins, range = (x_range,x_range,x_range), weights = data2)
indx_map_g, edges_1 = np.histogramdd((x,y,z), bins = bins, range = (x_range,x_range,x_range), weights = indx_map2)

x_arr = edges[0]#np.linspace(0, bins, bins+1)
y_arr = edges[1]#np.linspace(0, bins, bins+1)
x_core, y_core = np.meshgrid(((x_arr[1:] + x_arr[:-1]) /2), ((y_arr[1:] + y_arr[:-1]) /2))

color_list_ff = ['#008080','#FFD700','#4B0082','#808000']

fig = plt.figure(figsize = (14,14))
ax = fig.add_subplot(111)
ax.set_facecolor(rgb2hex(matplotlib.cm.get_cmap('Greys')(0))) # bg color for subplot - color using matplotlib.cm.get_cmap('viridis')[0] to hex
plt.imshow((np.nansum(dens_g, axis = proj_ax)), norm=LogNorm(), vmin = 1e2, vmax = 1e10, alpha = 1, cmap = 'Greys', origin = 'lower', extent = (x_range[0],x_range[1],x_range[0],x_range[1]))
# plt.colorbar()

for i,k in zip(core_id[::], global_bin_new[::]):
    core_dens = np.where(indx_map == i, data2, np.nan)
    core_dens_g, edges_3 = np.histogramdd((x,y,z), bins = bins, range = (x_range,x_range,x_range), weights = core_dens)
    plt.contour(x_core, y_core, np.nansum(core_dens_g, axis = proj_ax), colors = color_list_ff[k], levels = 1, alpha = 1)

plt.scatter(50,50, color=color_list_ff[0],alpha = 0, label = 'No', marker=".", s=200, rasterized = True)
plt.scatter(50,50, color=color_list_ff[1],alpha = 0, label = 'Low', marker=".", s=200, rasterized = True)
plt.scatter(50,50, color=color_list_ff[2],alpha = 0, label = 'Moderate', marker=".", s=200, rasterized = True)
plt.scatter(50,50, color=color_list_ff[3],alpha = 0, label = 'High', marker=".", s=200, rasterized = True)

leg = plt.legend(fontsize = 24, title = 'Global Bins')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh._sizes =[500]
    
    
plt.xlabel('x [pc]', fontsize = 34)
plt.ylabel('y [pc]', fontsize = 34)

plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)

plt.savefig("/u/kneralwar/ptmp_link/cores_paper_figs/core_vis_contour/image" + str(snap) + ".pdf", bbox_inches = 'tight') #saving as pdf

#For saving an interactive html figure
# html_str = mpld3.fig_to_html(fig)
# Html_file= open("/u/kneralwar/ptmp_link/cores_paper_figs/core_vis_contour/image" + str(snap) + ".html","w")
# Html_file.write(html_str)
# Html_file.close()