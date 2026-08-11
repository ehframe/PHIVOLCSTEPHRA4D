import os
from pathlib import Path
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

# -------------------------------------------------------------------
# 1. PATHS AND SITE PARAMETERS
# -------------------------------------------------------------------
main_dir = Path(__file__).resolve().parents[2]
trajpath = main_dir / "2026709_files" / "2026709" / "traj2026709"
outpath = main_dir / "2026709_files" / "grid_height_static"
outpath.mkdir(parents=True, exist_ok=True)

VENT_X = 514495.0
VENT_Y = 1150889.0

# Coordinate Converters
convert = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)

v_lon, v_lat = convert.transform(VENT_X, VENT_Y)

# Dispersion Physics Constants
K_RISE = 300.0
K_DIFF = 100.0
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)
FTT = 3600.0

# Map Boundaries & Grid Mesh Setup
llcrnlon, urcrnrlon = 122.800842, 123.801148
llrcnlat, urcrnrlat = 10.199636, 10.594028

lon_utm = np.linspace(llcrnlon, urcrnrlon, 25)
lat_utm = np.linspace(llrcnlat, urcrnrlat, 25)
X_lon, Y_lat = np.meshgrid(lon_utm, lat_utm)
X_m, Y_m = latlon_to_utm.transform(X_lon, Y_lat)

# Discrete Colorbar Bounds for Height (in kilometers ASL)
height_levels = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
base_cmap = plt.cm.get_cmap("terrain")
colors_list = base_cmap(np.linspace(0.2, 0.95, len(height_levels) - 1))
cmap = mcolors.ListedColormap(colors_list)
cmap.set_under(color=(0, 0, 0, 0))
norm = mcolors.BoundaryNorm(
    boundaries=height_levels, ncolors=cmap.N, clip=False
)

csvs = list(trajpath.glob("*mms-1.csv"))

# -------------------------------------------------------------------
# 2. CALCULATION AND PLOTTING
# -------------------------------------------------------------------
if not csvs:
    print("No trajectory files found (*mms-1.csv)")
else:
    for file_path in csvs:
        file_name = os.path.basename(file_path)
        print(f"Generating static height grid for: {file_name}")
        d_name = file_name.replace("mms-1.csv", "").capitalize()
        traj = pd.read_csv(file_path)

        x_centers = traj["x0"].values if "x0" in traj.columns else traj["x"].values
        y_centers = traj["y0"].values if "y0" in traj.columns else traj["y"].values

        # Retrieve altitude (z column, converted to kilometers)
        if "z0" in traj.columns:
            z_centers = traj["z0"].values / 1000.0
        elif "z" in traj.columns:
            z_centers = traj["z"].values / 1000.0
        else:
            z_centers = np.ones_like(x_centers) * 5.0

        if "time" in traj.columns:
            t_travel = traj["time"].values
        elif "t" in traj.columns:
            t_travel = traj["t"].values
        else:
            t_travel = np.linspace(60, 5400, len(x_centers))

        weight_sum = np.zeros_like(X_m)
        weighted_height_sum = np.zeros_like(X_m)

        for xc, yc, zc, t in zip(x_centers, y_centers, z_centers, t_travel):
            if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(zc)):
                continue

            t_calc = max(t, 1.0)
            if t_calc < FTT:
                sigma = 4.0 * K_RISE * t_calc
            else:
                sigma = (4.0 * K_RISE * FTT) + (1.6 * C_TURB * (t_calc**2.5))

            dx = X_m - xc
            dy = Y_m - yc
            nakami_exp = -(dx**2 + dy**2) / sigma

            if np.max(nakami_exp) > -40:
                w = np.where(
                    nakami_exp > -40,
                    1.0 / (sigma * np.pi) * np.exp(nakami_exp) * 1000.0,
                    0.0,
                )
                weight_sum += w
                weighted_height_sum += w * zc

        # Compute effective average height per grid cell
        height_grid = np.where(
            weight_sum > 1e-6, weighted_height_sum / weight_sum, np.nan
        )

        fig, ax = plt.subplots(figsize=(10, 8))

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

        X_map, Y_map = m(X_lon, Y_lat)

        mesh = m.pcolormesh(
            X_map,
            Y_map,
            height_grid,
            cmap=cmap,
            norm=norm,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.2,
            shading="nearest",
            zorder=4,
        )

        cbar = fig.colorbar(
            mesh, ax=ax, shrink=0.7, pad=0.04, ticks=height_levels
        )
        cbar.set_label(
            "Ash Plume Height (km ASL)",
            rotation=270,
            labelpad=15,
            fontweight="bold",
        )

        v_map_x, v_map_y = m(v_lon, v_lat)
        m.plot(
            v_map_x,
            v_map_y,
            "^r",
            markersize=11,
            mew=2,
            label="Kanlaon Vent",
            zorder=5,
        )

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
            f"Ash Plume Height Distribution ({d_name})",
            fontweight="bold",
            pad=12,
        )
        ax.legend(loc="upper right")

        plt.savefig(
            outpath / f"{d_name}_static_height_grid.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

print("Static height maps generated successfully!")