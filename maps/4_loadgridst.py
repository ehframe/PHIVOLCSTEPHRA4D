import sys
import os
from pathlib import Path
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
import rioxarray
from xrspatial import hillshade
main_dir = Path(__file__).resolve().parents[1]

sys.path.append(str(main_dir))

from variable import date, date_slash, date_clear, file_hour, start_hour, end_hour, edate
#dem_path = main_dir / "KANLAON" / "kanlaon_DEM.tif"
trajpath = main_dir / f"{date_clear}_{start_hour:02d}0000_files" / f"{date_clear}" / f"traj{date_clear}"
outpath = main_dir / f"{date_clear}_{start_hour:02d}0000_files" / "Load in grids (STATIC)"
outpath.mkdir(parents=True, exist_ok=True)

# Vent Coords
VENT_X = 514495.0
VENT_Y = 1150889.0

# coordinate converter from UTM to Lat/Lon
convert = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
v_lon, v_lat = convert.transform(VENT_X, VENT_Y)

# EXACT TRAJ-FALL PHYSICS PARAMETERS (Tephra4D Constants)
K_RISE = 300.0  # Plume rise horizontal diffusion (m^2/s)
K_DIFF = 100.0  # Horizontal eddy diffusion coefficient (m^2/s)
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)  # Atmospheric turbulence constant
FTT = 3600.0  # Falling Time Threshold in seconds (1 hour cutoff)

csvs = list(trajpath.glob("*mms-1.csv"))

'''# LOAD DEM AND INITIALIZE TERRAIN LAYOUT
dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
dem = dem.rio.reproject('EPSG:32651')  # UTM Zone 51N for Kanlaon
left, bottom, right, top = dem.rio.bounds()
hillshade_img = hillshade(dem, azimuth=180, angle_altitude=45)

# Convert DEM boundaries to kilometers relative to Vent
x_min = (left - VENT_X) / 1000.0
x_max = (right - VENT_X) / 1000.0
y_min = (bottom - VENT_Y) / 1000.0
y_max = (top - VENT_Y) / 1000.0
dem_extent = (x_min, x_max, y_min, y_max)

# Create computational 2D grid matrix blocks (60x60 grid cells)
x_grid = np.linspace(left, right, 60)
y_grid = np.linspace(bottom, top, 60)
X_m, Y_m = np.meshgrid(x_grid, y_grid)
# Pre-convert the meshgrid matrices to kilometers relative to Vent for fast plotting
X_km = (X_m - VENT_X) / 1000.0
Y_km = (Y_m - VENT_Y) / 1000.0'''

llcrnlon, urcrnrlon = 122.800842, 123.801148
llrcnlat, urcrnrlat = 10.199636, 10.594028

lon_utm = np.linspace(llcrnlon, urcrnrlon, 25)  # change 25 to any number to enlargen grid
lat_utm = np.linspace(llrcnlat, urcrnrlat, 25)
X_lon, Y_lat = np.meshgrid(lon_utm, lat_utm)

latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
X_m, Y_m = latlon_to_utm.transform(X_lon, Y_lat)

# DISCRETE COLORBAR BOUNDS CONFIGURATION
levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
base_cmap = plt.cm.get_cmap("YlOrRd")
colors_list = base_cmap(np.linspace(0.1, 1.0, len(levels) - 1))
cmap = mcolors.ListedColormap(colors_list)
cmap.set_under(color=(0, 0, 0, 0))  # Set values below minimum level transparent
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=False)

# TRAJ-FALL PHYSICS CONFIGURATION
if not csvs:
    print("No trajectory files found (*mms-1.csv)")
else:
    for file_path in csvs:
        file_name = os.path.basename(file_path)
        print(f"Generating cell-grid block simulation for: {file_name}")
        d_name = file_name.replace("mms-1.csv", "").capitalize()
        traj = pd.read_csv(file_path)

        x_centers = traj["x0"].values
        y_centers = traj["y0"].values

        if "time" in traj.columns:
            t_travel = traj["time"].values
        elif "t" in traj.columns:
            t_travel = traj["t"].values
        else:
            t_travel = np.linspace(60, 5400, len(x_centers))

        # Reset mass grid layer per run
        ash_accumulation_grid = np.zeros_like(X_m)

        for xc, yc, t in zip(x_centers, y_centers, t_travel):
            if not (np.isfinite(xc) and np.isfinite(yc)):
                continue

            t_calc = max(t, 1.0)

            # --- TRAJ-FALL VARIANCE (SIGMA) EQUATION ---
            if t_calc < FTT:
                sigma = 4.0 * K_RISE * t_calc
            else:
                sigma = (4.0 * K_RISE * FTT) + (
                    1.6 * C_TURB * (t_calc**2.5)
                )

            dx = X_m - xc
            dy = Y_m - yc
            nakami_exp = -(dx**2 + dy**2) / sigma

            if np.max(nakami_exp) > -40:
                weight_distribution = np.where(
                    nakami_exp > -40,
                    1.0 / (sigma * np.pi) * np.exp(nakami_exp) * 1000.0,
                    0.0,
                )
                ash_accumulation_grid += weight_distribution

        # figure and basemap
        fig, ax = plt.subplots(figsize=(10, 8))

        '''# Show terrain hillshade base map
        ax.imshow(hillshade_img, cmap='gray', extent=dem_extent, origin='upper', alpha=0.7)
        '''
        m = Basemap(
            resolution="h",
            projection="lcc",
            lat_0=v_lat,
            lon_0=v_lon,
            llcrnrlon=llcrnlon,
            llcrnrlat=llrcnlat,
            urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat,
            ax=ax,
        )
        m.drawmapboundary(fill_color="#A6CAE0", zorder=0)
        m.fillcontinents(color="#FFFFFF", lake_color="#A6CAE0", zorder=1)
        m.drawcoastlines(linewidth=0.8, color="#333333", zorder=2)
        m.drawcountries(linewidth=1.0, zorder=3)
        m.drawstates(linewidth=0.5, zorder=3)
        # 2d grid to lat/lon into basemap
        X_map, Y_map = m(X_lon, Y_lat)
        max_load = ash_accumulation_grid.max()
        if max_load > 0:
            # Mask out values lower than the minimum colorbar level threshold
            masked_grid = np.where(
                ash_accumulation_grid >= levels[0],
                ash_accumulation_grid,
                np.nan,
            )

            # Render the entire block layout grid instantly
            mesh = m.pcolormesh(
                X_map,
                Y_map,
                masked_grid,
                cmap=cmap,
                norm=norm,
                alpha=0.75,
                edgecolor="none",
                linewidth=0.2,
                shading="nearest",
            )

            # Map the colorbar explicitly to the output mesh layout with discrete tick marks
            cbar = fig.colorbar(
                mesh, ax=ax, shrink=0.7, pad=0.04, ticks=levels
            )
            cbar.set_label(
                "Ash Concentration",
                rotation=270,
                labelpad=15,
                fontweight="bold",
            )

        # Mark the Kanlaon crater vent position at local origin (0,0)
        v_map_x, v_map_y = m(v_lon, v_lat)

        m.plot(
            v_map_x,
            v_map_y,
            "^r",
            markersize=11,
            mew=2,
            label="Kanlaon Vent",
        )

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

        plt.title(
            f"Trajectory Plume Accumulation ({d_name})",
            fontweight="bold",
            pad=12,
        )

        ax.legend(loc="upper right")

        plt.savefig(
            outpath / f"{d_name}_block_plume_model.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

print("rock w u")