from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import io
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from pyproj import Transformer
import requests
from scipy.interpolate import RegularGridInterpolator

# -------------------------------------------------------------------
# 1. SETUP & CONSTANTS
# -------------------------------------------------------------------
main_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(main_dir))

from variable import (
    date,
    date_slash,
    date_clear,
    file_hour,
    start_hour,
    end_hour,
    edate,
)

trajpath = (
    main_dir
    / f"{date_clear}_{start_hour:02d}0000_files"
    / f"{date_clear}"
    / f"traj{date_clear}"
)
outpath = (
    main_dir / f"{date_clear}_{start_hour:02d}0000_files" / "Load in grids" / "3D"
)
outpath.mkdir(parents=True, exist_ok=True)

VENT_X = 514495.0
VENT_Y = 1150889.0
VENT_ELEV_KM = 2.435

K_RISE = 300.0
K_DIFF = 100.0
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)
V_SETTLE_KM_S = 0.0008

latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
utm_to_latlon = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

v_lon, v_lat = utm_to_latlon.transform(VENT_X, VENT_Y)

llcrnlon, urcrnrlon = 122.800842, 124.105784
llrcnlat, urcrnrlat = 10.199636, 10.690102

lon_utm = np.linspace(llcrnlon, urcrnrlon, 140)
lat_utm = np.linspace(llrcnlat, urcrnrlat, 100)
X_lon, Y_lat = np.meshgrid(lon_utm, lat_utm)
X_m, Y_m = latlon_to_utm.transform(X_lon, Y_lat)

levels = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
cmin, cmax = levels[0], levels[-1]
norm_levels = [(lvl - cmin) / (cmax - cmin) for lvl in levels]

hex_colors = [
    "#FFFF00",
    "#FFD700",
    "#FF9900",
    "#FF5500",
    "#FF0000",
    "#CC0044",
    "#990066",
    "#660066",
    "#330033",
]
plotly_colorscale = [[0.0, "rgba(0,0,0,0)"]]
for n_val, hex_c in zip(norm_levels, hex_colors):
    plotly_colorscale.append([n_val, hex_c])

csvs = list(trajpath.glob("*mms-1.csv"))
if not csvs:
    raise FileNotFoundError("No trajectory files found in directory.")

# -------------------------------------------------------------------
# 2. LOAD TRAJECTORY DATA
# -------------------------------------------------------------------
print(f"Loading {len(csvs)} trajectory files...")
all_x_list, all_y_list, all_z_list, all_t_list = [], [], [], []

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

    if "x0" in df.columns and "y0" in df.columns:
        x_pts, y_pts = df["x0"].to_numpy(), df["y0"].to_numpy()
    else:
        x_pts, y_pts = df["x"].to_numpy(), df["y"].to_numpy()

    if "z0" in df.columns:
        z_pts = (
            df["z0"].to_numpy() / 1000.0
            if df["z0"].max() > 100
            else df["z0"].to_numpy()
        )
    elif "h" in df.columns:
        z_pts = (
            df["h"].to_numpy() / 1000.0
            if df["h"].max() > 100
            else df["h"].to_numpy()
        )
    elif "z" in df.columns:
        z_pts = (
            df["z"].to_numpy() / 1000.0
            if df["z"].max() > 100
            else df["z"].to_numpy()
        )
    else:
        z_pts = np.full_like(x_pts, VENT_ELEV_KM)

    all_x_list.append(x_pts)
    all_y_list.append(y_pts)
    all_z_list.append(z_pts)
    all_t_list.append(df_time)

all_x = np.concatenate(all_x_list)
all_y = np.concatenate(all_y_list)
all_z = np.concatenate(all_z_list)
all_t = np.concatenate(all_t_list)

max_time = min(all_t.max(), 7200.0)
min_time = all_t.min()
num_frames = 30
time_steps = np.linspace(min_time, max_time, num_frames)
max_z_plume = max(float(all_z.max()), VENT_ELEV_KM + 1.0)

# -------------------------------------------------------------------
# 3. TERRAIN & BASEMAP
# -------------------------------------------------------------------
def lonlat_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_lonlat(xtile, ytile, zoom):
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    return lon_deg, math.degrees(lat_rad)


def fetch_terrain_grid_km(lon_grid, lat_grid, zoom=10):
    min_lon, max_lon = lon_grid.min(), lon_grid.max()
    min_lat, max_lat = lat_grid.min(), lat_grid.max()

    x_min, y_max = lonlat_to_tile(min_lon, min_lat, zoom)
    x_max, y_min = lonlat_to_tile(max_lon, max_lat, zoom)

    stitched_img = Image.new(
        "RGB", ((x_max - x_min + 1) * 256, (y_max - y_min + 1) * 256)
    )

    for i, x in enumerate(range(x_min, x_max + 1)):
        for j, y in enumerate(range(y_min, y_max + 1)):
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png"
            try:
                resp = requests.get(url, timeout=10)
                tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                stitched_img.paste(tile_img, (i * 256, j * 256))
            except Exception:
                pass

    img_arr = np.array(stitched_img, dtype=np.float32)
    elevation_m = (
        img_arr[:, :, 0] * 256.0 + img_arr[:, :, 1] + (img_arr[:, :, 2] / 256.0)
    ) - 32768.0

    tile_min_lon, tile_max_lat = tile_to_lonlat(x_min, y_min, zoom)
    tile_max_lon, tile_min_lat = tile_to_lonlat(x_max + 1, y_max + 1, zoom)

    interp = RegularGridInterpolator(
        (
            np.linspace(tile_max_lat, tile_min_lat, elevation_m.shape[0])[::-1],
            np.linspace(tile_min_lon, tile_max_lon, elevation_m.shape[1]),
        ),
        np.flipud(elevation_m),
        bounds_error=False,
        fill_value=0.0,
    )

    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return np.clip(interp(pts).reshape(lat_grid.shape) / 1000.0, 0.0, None)


terrain_z_km = fetch_terrain_grid_km(X_lon, Y_lat, zoom=10)
ground_footprint_z = terrain_z_km + 0.015

fig_temp, ax_temp = plt.subplots()
m = Basemap(
    resolution="h",
    projection="cyl",
    llcrnrlon=llcrnlon,
    llcrnrlat=llrcnlat,
    urcrnrlon=urcrnrlon,
    urcrnrlat=urcrnrlat,
    ax=ax_temp,
)
coastlines = m.drawcoastlines()
coast_traces = []
for path in coastlines.get_paths():
    verts = path.vertices
    coast_traces.append(
        go.Scatter3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=np.zeros(len(verts)),
            mode="lines",
            line=dict(color="#000000", width=3),
            showlegend=False,
            hoverinfo="none",
        )
    )
plt.close(fig_temp)

# -------------------------------------------------------------------
# 4. PLUME SURFACE & VERTICAL COLUMN CALCULATION
# -------------------------------------------------------------------
def compute_atmospheric_plume(current_time):
    time_mask = all_t <= current_time
    ash_grid = np.zeros_like(X_m)
    height_grid = np.zeros_like(X_m)
    weight_sum = np.zeros_like(X_m)

    if np.any(time_mask):
        xc_arr, yc_arr, zc_arr, t_arr = (
            all_x[time_mask],
            all_y[time_mask],
            all_z[time_mask],
            all_t[time_mask],
        )

        for xc, yc, zc, t in zip(xc_arr, yc_arr, zc_arr, t_arr):
            if not (np.isfinite(xc) and np.isfinite(yc)):
                continue
            t0_val = max(t, 0.1)
            t_diff = max(current_time - t0_val, 0.0)
            current_z = max(zc - (t_diff * V_SETTLE_KM_S), VENT_ELEV_KM)
            sigma = max(
                (4.0 * K_RISE * t0_val) + (1.6 * C_TURB * (t_diff**2.5)),
                (4.0 * K_RISE * t0_val) + (4.0 * K_DIFF * t_diff),
            )
            dx, dy = X_m - xc, Y_m - yc
            nakami_exp = -(dx**2 + dy**2) / sigma

            if np.max(nakami_exp) > -40:
                weight = np.where(
                    nakami_exp > -40,
                    (1.0 / (sigma * np.pi)) * np.exp(nakami_exp) * 1000.0,
                    0.0,
                )
                ash_grid += weight
                height_grid += weight * current_z
                weight_sum += weight

    masked_grid = np.where(ash_grid >= levels[0], ash_grid, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_z = np.where(weight_sum > 0, height_grid / weight_sum, np.nan)

    plume_z = np.where(np.isnan(masked_grid), np.nan, avg_z)
    return masked_grid, plume_z


def generate_eruption_column_mesh(max_height, plume_center_lon=None, plume_center_lat=None):
    # Anchor top center to dynamic plume position (fallback to vent)
    top_lon = plume_center_lon if plume_center_lon is not None else v_lon
    top_lat = plume_center_lat if plume_center_lat is not None else v_lat
    
    # Realistic base/top radii converted from meters to degrees (~30m base to ~300m top)
    r_base_deg = 0.0003  # ~33 meters
    r_top_deg  = 0.0030  # ~330 meters
    
    n_theta, n_z = 24, 20
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z_vals = np.linspace(VENT_ELEV_KM, max_height, n_z)
    
    theta_grid, z_grid = np.meshgrid(theta, z_vals)
    
    # Linear interpolation of radius and center position along Z-axis
    t_factor = (z_grid - VENT_ELEV_KM) / max(max_height - VENT_ELEV_KM, 0.01)
    r_grid = r_base_deg + (r_top_deg - r_base_deg) * t_factor
    
    cx_grid = v_lon + (top_lon - v_lon) * t_factor
    cy_grid = v_lat + (top_lat - v_lat) * t_factor
    
    col_x = cx_grid + r_grid * np.cos(theta_grid)
    col_y = cy_grid + r_grid * np.sin(theta_grid)
    
    return col_x, col_y, z_grid

plume_frames_data = [compute_atmospheric_plume(t) for t in time_steps]

# -------------------------------------------------------------------
# 5. BUILD COLUMN MESH + PLUME SURFACE SCENE
# -------------------------------------------------------------------
fig = go.Figure()

for trace in coast_traces:
    fig.add_trace(trace)

fig.add_trace(
    go.Surface(
        x=X_lon,
        y=Y_lat,
        z=terrain_z_km,
        colorscale="Earth",
        cmin=0.0,
        cmax=2.5,
        showscale=False,
        opacity=0.75,
        name="Terrain",
    )
)

initial_ash, initial_z = plume_frames_data[0]

fig.add_trace(
    go.Surface(
        x=X_lon,
        y=Y_lat,
        z=ground_footprint_z,
        surfacecolor=initial_ash,
        colorscale=plotly_colorscale,
        cmin=cmin,
        cmax=cmax,
        showscale=False,
        opacity=0.45,
        name="Ground Deposit Footprint",
    )
)

fig.add_trace(
    go.Surface(
        x=X_lon,
        y=Y_lat,
        z=initial_z,
        surfacecolor=initial_ash,
        colorscale=plotly_colorscale,
        cmin=cmin,
        cmax=cmax,
        opacity=0.92,
        colorbar=dict(title="Ash Concentration"),
        name="Umbrella Cloud Plume",
    )
)

col_x, col_y, col_z = generate_eruption_column_mesh(max_z_plume)
fig.add_trace(
    go.Surface(
        x=col_x,
        y=col_y,
        z=col_z,
        colorscale=[[0, "#FF0000"], [1, "#FFCC00"]],
        showscale=False,
        opacity=0.85,
        name="Eruption Column Mesh",
    )
)

plotly_frames = []
for idx, (t, (ash_data, z_plume)) in enumerate(
    zip(time_steps, plume_frames_data)
):
    frame_traces = list(coast_traces)
    frame_traces.append(
        go.Surface(
            x=X_lon,
            y=Y_lat,
            z=terrain_z_km,
            colorscale="Earth",
            cmin=0.0,
            cmax=2.5,
            showscale=False,
            opacity=0.75,
        )
    )
    frame_traces.append(
        go.Surface(
            x=X_lon,
            y=Y_lat,
            z=ground_footprint_z,
            surfacecolor=ash_data,
            colorscale=plotly_colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            opacity=0.45,
        )
    )
    frame_traces.append(
        go.Surface(
            x=X_lon,
            y=Y_lat,
            z=z_plume,
            surfacecolor=ash_data,
            colorscale=plotly_colorscale,
            cmin=cmin,
            cmax=cmax,
            opacity=0.92,
        )
    )
    # Compute dynamic top center from active plume data
    if not np.all(np.isnan(z_plume)):
        valid_mask = ~np.isnan(z_plume)
        c_lon = np.mean(X_lon[valid_mask])
        c_lat = np.mean(Y_lat[valid_mask])
        frame_max_z = np.nanmax(z_plume)
    else:
        c_lon, c_lat = v_lon, v_lat
        frame_max_z = VENT_ELEV_KM + 1.0

    fc_x, fc_y, fc_z = generate_eruption_column_mesh(frame_max_z, c_lon, c_lat)
    frame_traces.append(
        go.Surface(
            x=fc_x,
            y=fc_y,
            z=fc_z,
            colorscale=[[0, "#FF0000"], [1, "#FFCC00"]],
            showscale=False,
            opacity=0.85,
        )
    )

    plotly_frames.append(go.Frame(data=frame_traces, name=f"frame_{idx}"))

fig.frames = plotly_frames

fig.update_layout(
    title=f"3D Eruption Column Mesh & Atmospheric Plume — Time: {int(time_steps[0])}s",
    scene=dict(
        xaxis=dict(title="Longitude (°E)", range=[llcrnlon, urcrnrlon]),
        yaxis=dict(title="Latitude (°N)", range=[llrcnlat, urcrnrlat]),
        zaxis=dict(
            title="Altitude ASL (km)",
            range=[0.0, np.ceil(max_z_plume + 1.0)],
            dtick=2.0,
        ),
        aspectratio=dict(x=2.2, y=1.8, z=0.6),
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8)),
    ),
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "y": 0,
            "x": 0.1,
            "xanchor": "right",
            "yanchor": "top",
            "pad": {"t": 45, "r": 10},
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 200, "redraw": True},
                            "fromcurrent": True,
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                        },
                    ],
                },
            ],
        }
    ],
    sliders=[
        {
            "active": 0,
            "currentvalue": {"prefix": "Time: ", "suffix": "s"},
            "pad": {"t": 50},
            "steps": [
                {
                    "label": f"{int(t)}s",
                    "method": "animate",
                    "args": [
                        [f"frame_{k}"],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 200, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                }
                for k, t in enumerate(time_steps)
            ],
        }
    ],
)

html_file = outpath / "total_composite_ash_column_meshv2.html"
fig.write_html(html_file)
print(f"-> Saved 3D volcanic mesh plot: {html_file}")