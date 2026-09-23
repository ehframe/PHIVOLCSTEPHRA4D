from pathlib import Path
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

main_dir = Path(__file__).resolve().parents[0]

# Vent location in Lat/Lon
vent_lon, vent_lat = 123.13, 10.41

# 1. Open Dataset & Destagger Winds
ds = xr.open_dataset(
    main_dir / '20260709_0000' / 'extracted' / '2026-07-09_070000_raw.nc'
)

U, V = ds['U'].values, ds['V'].values
u_mass = 0.5 * (U[..., :-1] + U[..., 1:])
v_mass = 0.5 * (V[..., :-1, :] + V[..., 1:, :])

u_surf, v_surf = u_mass[0, 0, :, :], v_mass[0, 0, :, :]
wind_speed = np.sqrt(u_surf**2 + v_surf**2)

lats, lons = ds['XLAT'].values[0, :, :], ds['XLONG'].values[0, :, :]

# Zoom extent coordinates
llcrnrlon, urcrnrlon = 122.800842, 123.801148
llcrnrlat, urcrnrlat = 10.199636, 10.594028

# 2. Create Basemap Instance
plt.figure(figsize=(10, 8))
m = Basemap(
    projection='cyl',
    llcrnrlon=llcrnrlon,
    llcrnrlat=llcrnrlat,
    urcrnrlon=urcrnrlon,
    urcrnrlat=urcrnrlat,
    resolution='i',  # Use 'f' for full high-res shoreline detail if installed
)

m.drawcoastlines(linewidth=1.2)
m.drawcountries()

# Gridlines every 0.1 degrees for zoomed domain
m.drawparallels(
    np.arange(llcrnrlat, urcrnrlat, 0.1), labels=[1, 0, 0, 0], fmt='%.2f'
)
m.drawmeridians(
    np.arange(llcrnrlon, urcrnrlon, 0.2), labels=[0, 0, 0, 1], fmt='%.2f'
)

# Convert coordinates
x, y = m(lons, lats)
vx, vy = m(vent_lon, vent_lat)

# 3. Plot Quiver Field (stride = 2 for dense zoomed view)
skip = (slice(None, None, 2), slice(None, None, 2))

m.quiver(
    x[skip],
    y[skip],
    u_surf[skip],
    v_surf[skip],
    wind_speed[skip],
    cmap='viridis',
    scale=100,  # Smaller scale = larger/more visible vector arrows
    width=0.003,
)

# Plot Vent
m.plot(
    vx,
    vy,
    'r^',
    markersize=12,
    markeredgecolor='k',
    label='Vent Location',
    latlon=False,
)

plt.colorbar(label='Wind Speed (m/s)', shrink=0.7)
plt.title('Surface Wind Field & Vent Location', fontsize=12)
plt.legend(loc='upper right')
plt.show()