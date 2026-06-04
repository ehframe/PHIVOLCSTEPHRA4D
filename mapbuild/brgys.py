import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from pathlib import Path

main_dir = Path(__file__).resolve().parents[1]

VENT_X = 514495.0
VENT_Y = 1150889.0
dem_borders = 5.0

pallet = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]

bg = pd.read_csv(main_dir / "KANLAON" / "brgycindem_table.csv")
ld = pd.read_csv(main_dir / "2026315" / "w_rate" / "weight3_er2026315_C004_site.csv")
frl = pd.read_csv(main_dir / "4_load_ts_vt.csv")

#checking if the columns are the same
if 'site' not in frl.columns and(frl.columns[0] == ''):
    frl.rename(columns={frl.columns[0]: 'site'})
frl_col = [col for col in frl.columns if str(col).isdigit()]
time = frl.copy()
for col in frl.columns:
    time[col] = pd.to_numeric(time[col], errors='coerce').fillna(0)

time = time.copy()

total_sum= time[frl_col].sum(axis=1)

time['total_load'] = total_sum


group = time.groupby('site', as_index = False)['total_load'].sum(min_count=1)
#putting all tables in one site
combined = bg.merge(group, on='site', how='left')
combined['total_load'] = pd.to_numeric(combined['total_load'], errors='coerce').fillna(0)

combined.to_csv(main_dir / 'mapbuild/test.csv', index = False)

tst = pd.read_csv(main_dir / 'mapbuild/test.csv')

sc_x = tst["Long"]
sc_y = tst["Lat"]
sc_z = tst["total_load"]

levels = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]

marker_colored = colors.ListedColormap(pallet[:len(levels)-1])
custom_norm = colors.BoundaryNorm(levels, marker_colored.N)

###DEM LOADING###


import rasterio 
import earthpy.spatial as es
dem_path = main_dir / "KANLAON" / 'kanlaonDEM' /"kanlaon_DEM.tif"

with rasterio.open(dem_path) as dem:
    dem_data = dem.read(1)
    bounds = dem.bounds
    dem_data[dem_data<0] = np.nan

#convert bounds of dem to km
x_min = (bounds.left - VENT_X) / 1000.0
x_max = (bounds.right - VENT_X) / 1000.0
y_min = (bounds.bottom - VENT_Y) / 1000.0
y_max = (bounds.top - VENT_Y) / 1000.0

dem_extent = (x_min, x_max, y_min, y_max)

#convert scatter data to km
sc_x_km = (sc_x - VENT_X) / 1000.0
sc_y_km = (sc_y - VENT_Y) / 1000.0


#hillshade
hillshade = es.hillshade(dem_data, azimuth=360, altitude=45)


#borders of dem
plot_xlim = (x_min - dem_borders, x_max + dem_borders)
plot_ylim = (y_min - dem_borders, y_max + dem_borders)



###PLOTTING###

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(hillshade, cmap = 'gray', extent=dem_extent, origin="upper", alpha=0.8)

sc = ax.scatter(sc_x_km, 
                sc_y_km, 
                c=sc_z, 
                cmap=marker_colored,
                norm=custom_norm,
                s = 30,
                edgecolors='black', 
                linewidth=0.5)

ax.plot(0, 0, marker='^', color='red', markersize=20,label = 'VENT')

ax.set_aspect('equal')
ax.set_xlim(*plot_xlim)
ax.set_ylim(*plot_ylim)

ax.set_xlabel('Distance from Vent (km)')
ax.set_ylabel('Distance from Vent (km)')
ax.set_title('Brgy Total Load')

'''dem_plot = ax.imshow(dem_data, cmap='terrain', alpha=0.5)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Total Load')'''
cbar = fig.colorbar(sc, ax=ax, boundaries=levels)
cbar.set_label('Total Load (kg/m^2)')


plt.savefig(main_dir / 'mapbuild' / 'brgy_total_load.png', dpi=300)
plt.show()

