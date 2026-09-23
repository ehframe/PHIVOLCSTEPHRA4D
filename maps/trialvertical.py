from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.animation as anim
import matplotlib.colors as mcolors
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

trajpath = main_dir / f"{date_clear}_{start_hour:02d}0000_files" / f"{date_clear}" / f"traj{date_clear}"
outpath = main_dir / f"{date_clear}_{start_hour:02d}0000_files" / "Load in grids"
outpath.mkdir(parents=True, exist_ok=True)

# Kanlaon Vent Coordinates & Elevation
VENT_X = 514495.0
VENT_Y = 1150889.0
VENT_ELEV_M = 2435.0                  # Elevation in meters
VENT_ELEV_KM = VENT_ELEV_M / 1000.0   # 2.435 km ASL

# Tephra4D Dispersion Constants
K_RISE = 300.0  # K_rise (m^2/s)
K_DIFF = 100.0  # Horizontal diffusion K (m^2/s)
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)  # C factor
TIME_WINDOW = 60.0  # Active particle window (seconds)

latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
utm_to_latlon = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

v_lon, v_lat = utm_to_latlon.transform(VENT_X, VENT_Y)

# Vertical Cross-Section Grid Setup (Distance vs. Height ASL)
MAX_DIST_KM = 100.0   # Max horizontal range from vent in km
MAX_HEIGHT_KM = 20.0  # Max altitude in km ASL

dist_vec_m = np.linspace(0, MAX_DIST_KM * 1000.0, 100)      # Distance grid (m)
height_vec_m = np.linspace(0, MAX_HEIGHT_KM * 1000.0, 80)   # Height grid (m ASL)

DIST_M, HEIGHT_M = np.meshgrid(dist_vec_m, height_vec_m)
DIST_KM = DIST_M / 1000.0
HEIGHT_KM = HEIGHT_M / 1000.0

# Discrete Height/Concentration Colorbar Setup
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
# 2. LOAD & COMPOSITE ALL TRAJECTORIES (WITH VENT ELEVATION)
# -------------------------------------------------------------------
print(f"Loading and compositing {len(csvs)} trajectory files with vent elevation ({VENT_ELEV_KM} km ASL)...")

all_dist_list, all_z_list, all_t_list = [], [], []

for file_path in csvs:
    df = pd.read_csv(file_path)

    # Detect travel time column
    if "t0" in df.columns:
        df_time = df["t0"].to_numpy()
    elif "time" in df.columns:
        df_time = df["time"].to_numpy()
    elif "t" in df.columns:
        df_time = df["t"].to_numpy()
    else:
        df_time = np.linspace(0.32, 3600, len(df))

    # Detect horizontal coordinates
    x_pts = df["x0"].to_numpy() if "x0" in df.columns else df["x"].to_numpy()
    y_pts = df["y0"].to_numpy() if "y0" in df.columns else df["y"].to_numpy()

    # Detect height / altitude column
    if "z0" in df.columns:
        raw_z = df["z0"].to_numpy()
    elif "z" in df.columns:
        raw_z = df["z"].to_numpy()
    elif "height" in df.columns:
        raw_z = df["height"].to_numpy()
    elif "alt" in df.columns:
        raw_z = df["alt"].to_numpy()
    else:
        raw_z = np.zeros_like(x_pts)

    # Convert z to meters ASL (add vent elevation if z is relative to vent/AGL)
    if raw_z.max() < 1000.0:  # If values are in km AGL/ASL
        z_pts_m = (raw_z * 1000.0) + VENT_ELEV_M if raw_z.min() < VENT_ELEV_KM else raw_z * 1000.0
    else:                     # If values are in meters
        z_pts_m = raw_z + VENT_ELEV_M if raw_z.min() < VENT_ELEV_M else raw_z

    # Calculate radial distance from vent in meters
    dist_from_vent = np.sqrt((x_pts - VENT_X)**2 + (y_pts - VENT_Y)**2)

    all_dist_list.append(dist_from_vent)
    all_z_list.append(z_pts_m)
    all_t_list.append(df_time)

# Combine into master coordinate arrays
all_dist = np.concatenate(all_dist_list)
all_z = np.concatenate(all_z_list)
all_t = np.concatenate(all_t_list)

max_time = min(all_t.max(), 7200.0)
min_time = all_t.min()
num_frames = 30
time_steps = np.linspace(min_time, max_time, num_frames)

# -------------------------------------------------------------------
# 3. VERTICAL CROSS-SECTION VISUALIZATION SETUP
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

ax.set_facecolor("#F8F9FA")
ax.grid(True, linestyle="--", linewidth=0.4, color="gray", alpha=0.6)

ax.set_xlim(0, MAX_DIST_KM)
ax.set_ylim(0, MAX_HEIGHT_KM)

ax.set_xlabel("Distance from Vent (km)", fontsize=10, fontweight="bold")
ax.set_ylabel("Height / Altitude ASL (km)", fontsize=10, fontweight="bold")

# Plot terrain profile slope extending from vent down to sea level
dist_profile = [0, 20, MAX_DIST_KM]
elev_profile = [VENT_ELEV_KM, 0.2, 0.0]
ax.fill_between(dist_profile, elev_profile, color="#8B5A2B", alpha=0.35, label="Ground Surface", zorder=3)

# Plot Kanlaon Vent at 2.435 km ASL altitude
ax.plot(
    0,
    VENT_ELEV_KM,
    marker="^",
    markersize=12,
    color="red",
    label=f"Kanlaon Vent ({VENT_ELEV_KM} km ASL)",
    zorder=5,
)

title_text = ax.set_title(
    f"Vertical Ash Propagation Composite — Time: {int(time_steps[0])}s",
    fontsize=12,
    fontweight="bold",
)

initial_grid = np.full_like(DIST_M, np.nan)

# Create 2D pcolormesh mesh grid (Distance vs. Height ASL)
mesh = ax.pcolormesh(
    DIST_KM,
    HEIGHT_KM,
    initial_grid,
    cmap=cmap,
    norm=norm,
    alpha=0.85,
    edgecolor="none ",
    linewidth=0.1,
    shading="nearest",
    zorder=4,
)

cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03, ticks=levels)
cbar.set_label(
    "Total Active Ash Concentration",
    rotation=270,
    labelpad=15,
    fontweight="bold",
)
ax.legend(loc="upper right")

# -------------------------------------------------------------------
# 4. ANIMATION UPDATE LOOP
# -------------------------------------------------------------------
def update(frame):
    current_time = time_steps[frame]


    # Active time window filter
    time_mask = all_t <= current_time
    ash_grid = np.zeros_like(DIST_M)

    if np.any(time_mask):
        dist_c_arr = all_dist[time_mask]
        zc_arr = all_z[time_mask]
        t_arr = all_t[time_mask]

        for dc, zc, t in zip(dist_c_arr, zc_arr, t_arr):
            if not (np.isfinite(dc) and np.isfinite(zc)):
                continue

            t0_val = max(t, 0.1)
            t_diff = max(current_time - t0_val, 0.0)

            # Gaussian dispersion parameters
            sigma1 = (4.0 * K_RISE * t0_val) + (
                1.6 * C_TURB * (t_diff**2.5)
            )
            sigma2 = (4.0 * K_RISE * t0_val) + (4.0 * K_DIFF * t_diff)
            sigma = max(sigma1, sigma2)

            # Spatial delta (distance from vent and height ASL)
            ddist = DIST_M - dc
            dheight = HEIGHT_M - zc
            nakami_exp = -(ddist**2 + dheight**2) / sigma

            if np.max(nakami_exp) > -40:    
                weight = np.where(
                    nakami_exp > -40,
                    (1.0 / (sigma * np.pi))
                    * np.exp(nakami_exp)
                    * 1000.0,
                    0.0,
                )
                ash_grid += weight  # Accumulate concentration across active particles

    masked = np.where(ash_grid >= levels[0], ash_grid, np.nan)
    mesh.set_array(masked.ravel())
    title_text.set_text(
        f"Vertical Ash Propagation Composite — Time: {int(current_time)}s"
    )

    return mesh, title_text

animation = anim.FuncAnimation(
    fig, update, frames=num_frames, interval=200, blit=False
)
gif_file = outpath / "height_composite_ash_propagation.gif"
animation.save(gif_file, writer="pillow", fps=5)

plt.close(fig)
print(f"-> Saved vertical height composite animation: {gif_file}")