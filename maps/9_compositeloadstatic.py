from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")  # Non-GUI backend for saving image files
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

# -------------------------------------------------------------------
# 1. SETUP & CONSTANTS
# -------------------------------------------------------------------
main_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(main_dir))

from variable import date, date_slash, date_clear, file_hour, start_hour, end_hour, edate

trajpath = main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / f"{date_clear}" / f"traj{date_clear}"
outpath = main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / "Load in grids"
outpath.mkdir(parents=True, exist_ok=True)

VENT_X = 514495.0
VENT_Y = 1150889.0

# Tephra4D Dispersion Constants
K_RISE = 300.0
K_DIFF = 100.0
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)

latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
utm_to_latlon = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

v_lon, v_lat = utm_to_latlon.transform(VENT_X, VENT_Y)

# Map Bounds
llcrnlon, urcrnrlon = 122.800842, 124.105784
llrcnlat, urcrnrlat = 10.199636, 10.690102

# Grid Mesh Setup
lon_utm = np.linspace(llcrnlon, urcrnrlon, 60)
lat_utm = np.linspace(llrcnlat, urcrnrlat, 40)
X_lon, Y_lat = np.meshgrid(lon_utm, lat_utm)
X_m, Y_m = latlon_to_utm.transform(X_lon, Y_lat)

# Discrete Colorbar Setup
levels = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
base_cmap = plt.cm.get_cmap("YlOrRd")
colors_list = base_cmap(np.linspace(0.1, 1.0, len(levels) - 1))
cmap = mcolors.ListedColormap(colors_list)
cmap.set_under(color=(0, 0, 0, 0))  # Transparent values below minimum level
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=False)

csvs = list(trajpath.glob("*mms-1.csv"))
if not csvs:
    raise FileNotFoundError("No trajectory files found in directory.")

# -------------------------------------------------------------------
# 2. LOAD & INTEGRATE TOTAL LOAD ACROSS ALL TRAJECTORIES
# -------------------------------------------------------------------
print(f"Loading and processing {len(csvs)} trajectory files into a static load grid...")

total_load_grid = np.zeros_like(X_m)

for file_path in csvs:
    df = pd.read_csv(file_path)

    if "t0" in df.columns:
        df_time = df["t0"].to_numpy()
    elif "time" in df.columns:
        df_time = df["time"].to_numpy()
    elif "t" in df.columns:
        df_time = df["t"].to_numpy()
    else:
        df_time = np.linspace(0.32, 3600, len(df))

    x_pts = df["x0"].to_numpy() if "x0" in df.columns else df["x"].to_numpy()
    y_pts = df["y0"].to_numpy() if "y0" in df.columns else df["y"].to_numpy()

    for xc, yc, t in zip(x_pts, y_pts, df_time):
        if not (np.isfinite(xc) and np.isfinite(yc)):
            continue

        t0_val = max(t, 0.1)

        sigma1 = (4.0 * K_RISE * t0_val)
        sigma2 = (4.0 * K_RISE * t0_val) + (4.0 * K_DIFF * t0_val)
        sigma = max(sigma1, sigma2)

        dx = X_m - xc
        dy = Y_m - yc
        nakami_exp = -(dx**2 + dy**2) / sigma

        if np.max(nakami_exp) > -40:
            weight = np.where(
                nakami_exp > -40,
                (1.0 / (sigma * np.pi)) * np.exp(nakami_exp) * 1000.0,
                0.0,
            )
            total_load_grid += weight  # Accumulate into static composite grid

# -------------------------------------------------------------------
# 3. BASEMAP VISUALIZATION & SAVE
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

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
v_map_x, v_map_y = m(v_lon, v_lat)

m.plot(
    v_map_x,
    v_map_y,
    marker="^",
    markersize=12,
    color="red",
    label="Kanlaon Vent",
    zorder=5,
)

ax.set_title(
    f"Total Ash Load (Composite) — {date_clear}",
    fontsize=14,
    fontweight="bold",
)

m.drawparallels(
    np.arange(10.0, 11.0, 0.2),
    labels=[1, 0, 0, 0],
    fontsize=9,
    linewidth=0.3,
    color="gray",
)
m.drawmeridians(
    np.arange(122.8, 124.0, 0.2),
    labels=[0, 0, 0, 1],
    fontsize=9,
    linewidth=0.3,
    color="gray",
)

masked_load = np.where(total_load_grid >= levels[0], total_load_grid, np.nan)

mesh = m.pcolormesh(
    X_map,
    Y_map,
    masked_load,
    cmap=cmap,
    norm=norm,
    alpha=0.85,
    edgecolor="none",
    linewidth=0.1,
    shading="nearest",
    zorder=4,
)

cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.04, ticks=levels)
cbar.set_label(
    "Ash Concentration",
    rotation=270,
    labelpad=15,
    fontweight="bold",
)
ax.legend(loc="upper right")

output_image = outpath / "static_composite_total_load.png"
plt.savefig(output_image, bbox_inches="tight")
plt.close(fig)

print(f"-> Saved high-resolution static composite plot: {output_image}")