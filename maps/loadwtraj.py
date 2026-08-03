#importing all needed libraries

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
import rioxarray
from xrspatial import hillshade

main_dir = Path(__file__).resolve().parents[1]
dem_path = main_dir / "KANLAON" / 'kanlaonDEM' /"kanlaon_DEM.tif"
sitecsv = main_dir/ "2026709_files"  / 'totalload.csv'
trajpath = main_dir/'2026709_files' / '2026709' / 'traj2026709'
outputdir = main_dir / '2026709_files' / 'Load and Trajectory'
outputdir.mkdir(parents=True, exist_ok=True)
VENT_X = 514495.0
VENT_Y = 1150889.0
dem_borders = 40.0

#pallets
colorload = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]
#colorparticles = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]


#LOAD dem
dem = rioxarray.open_rasterio(dem_path, masked = True).squeeze()
dem = dem.rio.reproject('EPSG:32651')
left, bottom, right, top = dem.rio.bounds()
hillshade = hillshade(dem, azimuth= 180, angle_altitude= 45)

#convert bounds of dem to km
x_min = (left - VENT_X) / 1000.0
x_max = (right - VENT_X) / 1000.0
y_min = (bottom - VENT_Y) / 1000.0
y_max = (top - VENT_Y) / 1000.0
dem_extent = (x_min, x_max, y_min, y_max)

#borders of dem
plot_xlim = (x_min - dem_borders, x_max + dem_borders)
plot_ylim = (y_min - dem_borders, y_max + dem_borders)


#SITE LOAD DATA
load = pd.read_csv(sitecsv)
ld_x = load['Long']
ld_y = load['Lat']
ld_z = load['total_load']

#load data
ld_level = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]
ld_marker = colors.ListedColormap(colorload[:len(ld_level)-1])
ld_norm = colors.BoundaryNorm(ld_level, ld_marker.N)
ld_x_km = (ld_x - VENT_X) / 1000.0
ld_y_km = (ld_y - VENT_Y) / 1000.0


#ASH TRAJECTORY DATA
csvs = list(trajpath.glob('*mms-1.csv'))
at_level = [0, 1, 2, 3, 4, 5, 6, 7]
at_marker = colors.ListedColormap(colorload[:len(ld_level)-1])
at_norm = colors.BoundaryNorm(at_level, at_marker.N)


### PLOTTING ###
if not csvs:
    print('heehee beat it (walang -mms)')
else:
    for file_path in csvs:
        file_name = os.path.basename(file_path)
        print(file_name)
        d_name = file_name.replace('mms-1.csv', '').capitalize()
        traj = pd.read_csv(file_path)

        #load traj data
        traj_x_km = (traj['x0'] - VENT_X) / 1000.0
        traj_y_km = (traj['y0'] - VENT_Y) / 1000.0
        traj_z = traj['z0']/1000

        fig, ax = plt.subplots(figsize=(10, 10))
        #show dem
        ax.imshow(hillshade, cmap = 'gray', extent = dem_extent, origin = 'upper', alpha=0.8)

        #layer 1
        at = ax.scatter(traj_x_km, 
                traj_y_km, 
                c=traj_z, 
                cmap=at_marker,
                norm=at_norm,
                s = 30,
                marker = 'o',
                linewidth=0.5,
                label = 'Ash Trajectory'
                )
        ld = ax.scatter(ld_x_km,
                        ld_y_km,
                        c = ld_z,
                        cmap = ld_marker,
                        norm = ld_norm,
                        s = 30,
                        marker = 'o',
                        edgecolors='black',
                        linewidths=0.5,
                        label = 'Site load'
                        )
        
        ax.plot(0, 0, marker='^', color='red', markersize=5,label = 'VENT')

        ax.set_aspect('equal')
        ax.set_xlim(*plot_xlim)
        ax.set_ylim(*plot_ylim)

        ax.set_xlabel('Distance from Vent (km)')
        ax.set_ylabel('Distance from Vent (km)')
        ax.set_title(f'TRAJECTORY AND SITE LOAD ({d_name})')
        #cbar ash trajectory
        cbar_at = fig.colorbar(at, ax=ax, boundaries=at_level)
        cbar_at.set_label('z0 (km)')
        #cbar ground site
        cbar_ld = fig.colorbar(ld, ax=ax, boundaries=ld_level)
        cbar_ld.set_label('Total Load (kg/m^2)')
        plt.savefig(outputdir / f'{d_name}_combined_overlay.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print('omke na')