import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from pathlib import Path
import rioxarray
#from xrspatial import hillshade
from pyproj import Transformer
from mpl_toolkits.basemap import Basemap
# Define base paths
main_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(main_dir))

from variable import date, date_slash, date_clear, file_hour, start_hour, end_hour, edate

dem_path = main_dir / "KANLAON" / 'kanlaonDEM' / "kanlaon_DEM.tif"
trajpath = main_dir /f"{date_clear}_{start_hour:02d}0000_files"  / f'{date_clear}' / f'traj{date_clear}'
outputdir = main_dir / f"{date_clear}_{start_hour:02d}0000_files" / 'plotted trajectories'
outputdir.mkdir(parents=True, exist_ok=True)

# Vent location (UTM Zone 51N, meters)
VENT_X = 514495.0
VENT_Y = 1150889.0
dem_borders = 10.0  # km padding around DEM

#Transform UTM to Lat/Lon
converter = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
v_lon, v_lat = converter.transform(VENT_X, VENT_Y)


# Palette and levels setup
pallet = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]
levels = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
marker_colored = colors.ListedColormap(pallet[:len(levels)-1])
custom_norm = colors.BoundaryNorm(levels, marker_colored.N)

'''#LOAD dem
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

'''
#plot trajectories
csvs = list(trajpath.glob('*mms-1.csv'))

if not csvs:
    print('No -mms files found')
else:
    for file_path in csvs:
        file_name = file_path.name
        print(f"Processing: {file_name}")
        d_name = file_name.replace('mms-1.csv', '').capitalize()
        
        traj = pd.read_csv(file_path)
        
        # Relative coordinates in UTM
        p_lon, p_lat = converter.transform(traj['x0'].values, traj['y0'].values) 
        traj_z = traj['z0']
        


        # Create Plot
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        '''# Draw Hillshade as map background
        ax.imshow(hillshade, cmap='gray', extent=dem_extent, origin="upper", alpha=0.8)
        '''

        #map setup
        m = Basemap(
                    resolution="h",
                    projection="lcc",
                    lat_0=10.3968,
                    lon_0=123.3010,
                    llcrnrlon=122.800842,
                    llcrnrlat=10.199636,
                    urcrnrlon=123.801148,
                    urcrnrlat=10.594028,
                    ax=ax,
                )
        m.drawmapboundary(fill_color="#A6CAE0", zorder=0)
        m.fillcontinents(color="#FFFFFF", lake_color="#A6CAE0", zorder=1)
        m.drawcoastlines(linewidth=0.8, color="#333333", zorder=2)
        m.drawcountries(linewidth=1.0, zorder=3)
        m.drawstates(linewidth=0.5, zorder=3)


        map_x, map_y = m(p_lon, p_lat)
        # Plot Trajectory particles
        sc = ax.scatter(
            map_x, map_y,
            c=traj_z,
            cmap=marker_colored,
            norm=custom_norm, 
            s=30, linewidth=0.2, alpha=0.9
        )

        # Mark Kanlaon Vent
        vent_map_x, vent_map_y = m(v_lon, v_lat)
        m.plot(vent_map_x, vent_map_y, marker='^', color='red', markersize=8, label='VENT', zorder=5)
        # Add gridlines (parallels and meridians)
        m.drawparallels(
            np.arange(10.0, 11.0, 0.2),
            labels=[1, 0, 0, 0],
            fontsize=8,
            linewidth=0.3,
        )
        m.drawmeridians(
            np.arange(122.8, 124.0, 0.2),
            labels=[0, 0, 0, 1],
            fontsize=8,
            linewidth=0.3,
        )

        # Titles and Colorbar
        plt.title(
            f"TRAJECTORY OF ASH PARTICLES ({d_name})",
            fontweight="bold",
            pad=12,
        )
        ax.legend(loc="upper right")

        cbar = fig.colorbar(sc, ax=ax, boundaries=levels, shrink=0.8)
        cbar.set_label("z0 Elevation (m)", fontweight="bold")

        # Save and close figure
        plt.savefig(
            outputdir / f"{d_name}_traj.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

print("rock with u")