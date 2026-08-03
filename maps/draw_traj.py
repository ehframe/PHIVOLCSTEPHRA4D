import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import glob
from scipy import stats
import itertools
from scipy import integrate
from pathlib import Path
import rasterio
import earthpy.spatial as es

main_dir = Path(__file__).resolve().parents[1]

dem_path = main_dir / "KANLAON" / 'kanlaonDEM' /"kanlaon_DEM.tif"

sitecsv = main_dir/ "2026709_files"  / 'totalload.csv'

trajpath = main_dir/'2026709_files' / '2026709' / 'traj2026709'

print(trajpath)
dem_borders = 10.0

# Load trajectory data
csvs = list(trajpath.glob('*mms-1.csv'))





VENT_X = 514495.0
VENT_Y = 1150889.0

pallet = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]


#LOAD DEM

with rasterio.open(dem_path) as dem:
    dem_data = dem.read(1)
    bounds = dem.bounds
    dem_data[dem_data <= 0] = np.nan

#convert bounds of dem to km
x_min = (bounds.left - VENT_X) / 1000.0
x_max = (bounds.right - VENT_X) / 1000.0
y_min = (bounds.bottom - VENT_Y) / 1000.0
y_max = (bounds.top - VENT_Y) / 1000.0

dem_extent = (x_min, x_max, y_min, y_max)



levels = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]

marker_colored = colors.ListedColormap(pallet[:len(levels)-1])
custom_norm = colors.BoundaryNorm(levels, marker_colored.N)

#hillshade
hillshade = es.hillshade(dem_data, azimuth=360, altitude=20)


#borders of dem
plot_xlim = (x_min - dem_borders, x_max + dem_borders)
plot_ylim = (y_min - dem_borders, y_max + dem_borders)


#PLOTTING#
if not csvs:
    print('no -mms files found')
else:
    for file_path in csvs:
        file_name = os.path.basename(file_path)
        print(file_name)
        d_name = file_name.replace('mms-1.csv', '').capitalize()
        traj = pd.read_csv(file_path) 

        #LOAD TRAJ DATA
        traj_x = traj['x0']
        traj_y = traj['y0']
        traj_z = traj['z0']
        #convert scatter data to km
        traj_x_km = (traj_x - VENT_X) / 1000.0
        traj_y_km = (traj_y - VENT_Y) / 1000.0
        #vent coordinates and load pallet
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(hillshade, cmap = 'gray', extent=dem_extent, origin="upper", alpha=0.8)

        sc = ax.scatter(traj_x_km, 
                traj_y_km, 
                c=traj_z, 
                cmap=marker_colored,
                norm=custom_norm,
                s = 30,
                linewidth=0.5)
        #sc = ax.plot()
        ax.plot(0, 0, marker='^', color='red', markersize=5,label = 'VENT')

        ax.set_aspect('equal')
        ax.set_xlim(*plot_xlim)
        ax.set_ylim(*plot_ylim)

        ax.set_xlabel('Distance from Vent (km)')
        ax.set_ylabel('Distance from Vent (km)')
        ax.set_title(f'TRAJECTORY OF ASH PARTICLES ({d_name})')

        '''dem_plot = ax.imshow(dem_data, cmap='terrain', alpha=0.5)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Total Load')'''
        cbar = fig.colorbar(sc, ax=ax, boundaries=levels)
        cbar.set_label('z0 (m)')
        
        outputdir = main_dir / '2026709_files' / 'trajectoriesplotted'
        outputdir.mkdir(parents=True, exist_ok=True)

        plt.savefig(outputdir / f'{d_name} traj.png', dpi=300)
        plt.close(fig)
print('rock with u')
