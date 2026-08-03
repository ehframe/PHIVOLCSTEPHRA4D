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
import earthpy.spatial as es
import geopandas as gpd
from shapely.geometry import Point

main_dir = Path(__file__).resolve().parents[1]
dem_path = main_dir / "KANLAON" / 'kanlaonDEM' /"kanlaon_DEM.tif"
sitecsv = main_dir/ "mapbuild" / 'totalload.csv'
trajpath = main_dir/'2026315'/'traj2026315'
shpdir = main_dir / 'mapbuild'/'shpfiles'
shpdir.mkdir(parents=True, exist_ok=True)


#utm zone 51N for kanlaon
utm = 'EPSG:32651'

# Load trajectory data
csvs = list(trajpath.glob('*mms-1.csv'))

print('load shape file loading')
load = pd.read_csv(sitecsv)

geom_load = [Point(x,y) for x, y in zip(load['Long'], load['Lat'])]
gdf_load = gpd.GeoDataFrame(load, geometry = geom_load, crs=utm)
gdf_load.to_file(shpdir/'siteload.shp')

if not csvs:
    print('no -mms found brodie')
else:
    for file_path in csvs:
        file_name = os.path.basename(file_path)
        d_name = file_name.replace('mms-1.csv', '').capitalize()
        print(d_name)

        traj = pd.read_csv(file_path)

        valid_coords = np.isfinite(traj['x0'], np.isfinite(traj['y0']))

        traj = traj[valid_coords]

        geom_traj = [Point(x,y) for x, y in zip(traj['x0'], traj['y0'])]
        gdf_traj = gpd.GeoDataFrame(traj, geometry = geom_traj, crs=utm)
        trajshpdir = (shpdir/ 'shptraj')
        trajshpdir.mkdir(parents=True, exist_ok = True)
        gdf_traj.to_file(trajshpdir/f'{d_name}_trajectory.shp')
    print('omke na')



