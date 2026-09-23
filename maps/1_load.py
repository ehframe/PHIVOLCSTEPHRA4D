import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from pathlib import Path
from pyproj import Transformer
from mpl_toolkits.basemap import Basemap

main_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(main_dir))

from variable import date, date_slash, date_clear, file_hour, start_hour, end_hour, edate

VENT_X = 514495.0
VENT_Y = 1150889.0

# Transformer for converting UTM Zone 51N (meters) to Lat/Lon degrees (WGS84)
converter = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
v_lon, v_lat = converter.transform(VENT_X, VENT_Y)

pallet = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]

bg = pd.read_csv(main_dir / "KANLAON" / "affected_sites_brgy_format.csv")
ld = pd.read_csv(main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / f"{date_clear}" / "w_rate" / f"weight3_er{date_clear}_C004_site.csv")
frl = pd.read_csv(main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / "4_load_ts_vt.csv")
mapout = main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / "mapoutputs"
os.makedirs(mapout, exist_ok=True)

# Checking if the columns are the same
bg.columns = bg.columns.str.strip()
bg = bg.rename(columns={col: 'site' for col in bg.columns if col.lower() == 'site'})

if 'site' not in frl.columns and (frl.columns[0] == ''):
    frl = frl.rename(columns={frl.columns[0]: 'site'})

frl_col = [col for col in frl.columns if str(col).isdigit()]
time = frl.copy()
for col in frl.columns:
    time[col] = pd.to_numeric(time[col], errors='coerce').fillna(0)

time = time.copy()
total_sum = time[frl_col].sum(axis=1)
time['total_load'] = total_sum

group = time.groupby('site', as_index=False)['total_load'].sum(min_count=1)

# Putting all tables in one site
combined = bg.merge(group, on='site', how='left')
combined['total_load'] = pd.to_numeric(combined['total_load'], errors='coerce').fillna(0)

combined.to_csv(main_dir / f"{mapout}/totalload.csv", index=False)
tst = pd.read_csv(main_dir / f"{mapout}/totalload.csv")

# Ensure valid data rows
tst = tst.dropna(subset=["Long", "Lat", "total_load"]).copy()

sc_x_raw = pd.to_numeric(tst["Long"], errors="coerce")
sc_y_raw = pd.to_numeric(tst["Lat"], errors="coerce")
sc_z = pd.to_numeric(tst["total_load"], errors="coerce")

# Auto-detect if coordinates are in UTM meters (>180) and convert to Lat/Lon degrees if needed
if len(sc_x_raw) > 0 and sc_x_raw.iloc[0] > 180:
    sc_x, sc_y = converter.transform(sc_x_raw.values, sc_y_raw.values)
else:
    sc_x, sc_y = sc_x_raw.values, sc_y_raw.values

levels = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]

marker_colored = colors.ListedColormap(pallet[:len(levels)-1])
custom_norm = colors.BoundaryNorm(levels, marker_colored.N)

### PLOTTING WITH BASEMAP ###

fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

# Map setup matching the geographic extent
m = Basemap(
    resolution="h",
    projection="lcc",
    lat_0=10.4120,          # Centered near Kanlaon Vent Latitude
    lon_0=123.2500,         # Centered across the point cluster
    llcrnrlon=123.05,       # Lower-Left Longitude
    llcrnrlat=10.25,        # Lower-Left Latitude
    urcrnrlon=123.45,       # Upper-Right Longitude
    urcrnrlat=10.55,        # Upper-Right Latitude
    ax=ax,
)
m.drawmapboundary(fill_color="#A6CAE0", zorder=0)
m.fillcontinents(color="#FFFFFF", lake_color="#A6CAE0", zorder=1)
m.drawcoastlines(linewidth=0.8, color="#333333", zorder=2)
m.drawcountries(linewidth=1.0, zorder=3)
m.drawstates(linewidth=0.5, zorder=3)

# Project geographic points (Long, Lat) to map plane coordinates
map_x, map_y = m(sc_x, sc_y)
vent_map_x, vent_map_y = m(v_lon, v_lat)

# Plot site load data (zorder=10 ensures points are drawn over map features)
sc = ax.scatter(
    map_x, 
    map_y, 
    c=sc_z, 
    cmap=marker_colored,
    norm=custom_norm,
    s=50,
    edgecolors='black', 
    linewidth=0.6,
    zorder=10
)

# Mark Kanlaon Vent
m.plot(vent_map_x, vent_map_y, marker='^', color='red', markersize=12, label='VENT', zorder=11)

# Add gridlines (parallels and meridians)
m.drawparallels(
    np.arange(10.0, 11.0, 0.1),
    labels=[1, 0, 0, 0],
    fontsize=8,
    linewidth=0.3,
)
m.drawmeridians(
    np.arange(122.8, 124.0, 0.1),
    labels=[0, 0, 0, 1],
    fontsize=8,
    linewidth=0.3,
)

# Title & Legend
ax.set_title('Load Distribution at Sites', fontweight='bold', pad=12)
ax.legend(loc="upper right")

cbar = fig.colorbar(sc, ax=ax, boundaries=levels, shrink=0.8, pad=0.03)
cbar.set_label('Total Load (kg/m²)', fontweight='bold')

plt.savefig(f"{mapout}/brgy_total_load.png", dpi=300, bbox_inches="tight")
plt.show()