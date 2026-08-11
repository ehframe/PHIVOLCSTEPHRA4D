from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Non-GUI backend for saving GIFs
import matplotlib.animation as anim
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

# -------------------------------------------------------------------
# 1. SETUP & PARAMETERS
# -------------------------------------------------------------------
main_dir = Path(__file__).resolve().parents[1]
trajpath = main_dir / "2026709_files" / "2026709" / "traj2026709"
outpath = main_dir / "2026709_files" / "grid_height_animations"
outpath.mkdir(parents=True, exist_ok=True)

VENT_X = 514495.0
VENT_Y = 1150889.0

# Tephra4D Dispersion Constants
K_RISE = 300.0
K_DIFF = 100.0
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)
TIME_WINDOW = 60.0  # Active time window (seconds)

latlon_to_utm = Transformer.from_crs(
    "EPSG:4326", "EPSG:32651", always_xy=True
)
utm_to_latlon = Transformer.from_crs(
    "EPSG:32651", "EPSG:4326", always_xy=True
)

v_lon, v_lat = utm_to_latlon.transform(VENT_X, VENT_Y)

# Map Bounds
llcrnlon, urcrnrlon = 122.800842, 123.801148
llrcnlat, urcrnrlat = 10.199636, 10.594028

lon_utm = np.linspace(llcrnlon, urcrnrlon, 25)
lat_utm = np.linspace(llrcnlat, urcrnrlat, 25)
X_lon, Y_lat = np.meshgrid(lon_utm, lat_utm)
X_m, Y_m = latlon_to_utm.transform(X_lon, Y_lat)

# Custom Discrete Height Colorbar Levels (km ASL)
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
# 2. ANIMATION GENERATION
# -------------------------------------------------------------------
for file_path in csvs:
    d_name = file_path.stem.replace("mms-1", "").capitalize()
    print(f"Generating height animation GIF for: {d_name}")

    df = pd.read_csv(file_path)

    df_time = (
        df["t0"].to_numpy()
        if "t0" in df.columns
        else df["time"].to_numpy()
        if "time" in df.columns
        else df["t"].to_numpy()
        if "t" in df.columns
        else np.linspace(0.32, 3600, len(df))
    )

    x_pts = (
        df["x0"].to_numpy() if "x0" in df.columns else df["x"].to_numpy()
    )
    y_pts = (
        df["y0"].to_numpy() if "y0" in df.columns else df["y"].to_numpy()
    )

    if "z0" in df.columns:
        z_pts = df["z0"].to_numpy() / 1000.0
    elif "z" in df.columns:
        z_pts = df["z"].to_numpy() / 1000.0
    else:
        z_pts = np.ones_like(x_pts) * 5.0

    max_time = min(df_time.max(), 3600.0)
    num_frames = 30
    time_steps = np.linspace(df_time.min(), max_time, num_frames)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

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
        markersize=10,
        color="red",
        label="Vent Location",
        zorder=5,
    )

    title_text = ax.set_title(
        f"Ash Airborne Height ({d_name}) — Time: {int(time_steps[0])}s",
        fontsize=12,
        fontweight="bold",
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

    initial_grid = np.full_like(X_m, np.nan)
    mesh = m.pcolormesh(
        X_map,
        Y_map,
        initial_grid,
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
        "Active Ash Height (km ASL)",
        rotation=270,
        labelpad=15,
        fontweight="bold",
    )

    def update(frame):
        current_time = time_steps[frame]

        time_mask = (df_time >= current_time - TIME_WINDOW) & (
            df_time <= current_time + TIME_WINDOW
        )

        weight_sum = np.zeros_like(X_m)
        weighted_height_sum = np.zeros_like(X_m)

        if np.any(time_mask):
            xc_arr = x_pts[time_mask]
            yc_arr = y_pts[time_mask]
            zc_arr = z_pts[time_mask]
            t_arr = df_time[time_mask]

            for xc, yc, zc, t in zip(xc_arr, yc_arr, zc_arr, t_arr):
                if not (
                    np.isfinite(xc) and np.isfinite(yc) and np.isfinite(zc)
                ):
                    continue

                t0_val = max(t, 0.1)
                t_diff = max(current_time - t0_val, 0.0)

                sigma1 = (4.0 * K_RISE * t0_val) + (
                    1.6 * C_TURB * (t_diff**2.5)
                )
                sigma2 = (4.0 * K_RISE * t0_val) + (4.0 * K_DIFF * t_diff)
                sigma = max(sigma1, sigma2)

                dx = X_m - xc
                dy = Y_m - yc
                nakami_exp = -(dx**2 + dy**2) / sigma

                if np.max(nakami_exp) > -40:
                    w = np.where(
                        nakami_exp > -40,
                        (1.0 / (sigma * np.pi))
                        * np.exp(nakami_exp)
                        * 1000.0,
                        0.0,
                    )
                    weight_sum += w
                    weighted_height_sum += w * zc

        height_grid = np.where(
            weight_sum > 1e-6, weighted_height_sum / weight_sum, np.nan
        )

        mesh.set_array(height_grid.ravel())
        title_text.set_text(
            f"Ash Airborne Height ({d_name}) — Time: {int(current_time)}s"
        )

        return mesh, title_text

    animation = anim.FuncAnimation(
        fig, update, frames=num_frames, interval=200, blit=False
    )
    gif_file = outpath / f"{d_name}_height_propagation.gif"
    animation.save(gif_file, writer="pillow", fps=5)

    plt.close(fig)
    print(f"-> Saved GIF: {gif_file}")