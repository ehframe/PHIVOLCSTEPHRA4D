import io
import math
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from pyproj import Transformer
import requests
from scipy.interpolate import RegularGridInterpolator

# ===================================================================
# SECTION 1: SETUP & CONSTANTS
# ===================================================================
main_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(main_dir))

from variable import (
    date,
    date_clear,
    date_slash,
    edate,
    end_hour,
    file_hour,
    start_hour,
)

trajpath = (
    main_dir
    / f"{date_clear}_{start_hour:02d}0000_files_v1"
    / f"{date_clear}"
    / f"traj{date_clear}"
)
outpath = (
    main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / "Load in grids" / "3D"
)
outpath.mkdir(parents=True, exist_ok=True)

VENT_X = 514495.0
VENT_Y = 1150889.0

K_RISE = 300.0
K_DIFF = 100.0
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)

# Settling velocity spectrum (converted m/s to km/s)
VELOCITIES_M_S = np.array([
    0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
    1.1,  1.3,  1.5,  1.7,  1.9,  2.2,  2.6,  3.0,  3.4,  3.8,
    4.4,  5.2,  6.0,  6.8,  7.6,  8.8,  10.4, 12.0, 13.6, 15.2,
    17.6, 20.8
])
V_SETTLE_SPECTRUM_KM_S = VELOCITIES_M_S / 1000.0
WEIGHT_PER_BIN = 1.0 / len(V_SETTLE_SPECTRUM_KM_S)

latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
utm_to_latlon = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

v_lon, v_lat = utm_to_latlon.transform(VENT_X, VENT_Y)

llcrnlon, urcrnrlon = 122.800842, 124.105784
llcrnlat, urcrnrlat = 10.199636, 10.690102

# Grid resolution (280 x 200)
lon_utm = np.linspace(llcrnlon, urcrnrlon, 280)
lat_utm = np.linspace(llcrnlat, urcrnrlat, 200)
X_lon, Y_lat = np.meshgrid(lon_utm, lat_utm)
X_m, Y_m = latlon_to_utm.transform(X_lon, Y_lat)

levels = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
cmin, cmax = levels[0], levels[-1]

base_cmap = matplotlib.colormaps["YlOrRd"].resampled(len(levels) - 1)
colors_list = [mcolors.to_hex(base_cmap(i)) for i in range(len(levels) - 1)]

plotly_colorscale = []
for i in range(len(levels) - 1):
    val_start = (levels[i] - cmin) / (cmax - cmin)
    val_end = (levels[i + 1] - cmin) / (cmax - cmin)
    hex_col = colors_list[i]
    plotly_colorscale.append([val_start, hex_col])
    plotly_colorscale.append([val_end, hex_col])

csvs = list(trajpath.glob("*mms-1.csv"))
if not csvs:
    raise FileNotFoundError("No trajectory files found in directory.")

# ===================================================================
# SECTION 2: LOAD TRAJECTORY DATA
# ===================================================================
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
        z_pts = df["z0"].to_numpy() / 1000.0 if df["z0"].max() > 100 else df["z0"].to_numpy()
    elif "h" in df.columns:
        z_pts = df["h"].to_numpy() / 1000.0 if df["h"].max() > 100 else df["h"].to_numpy()
    elif "z" in df.columns:
        z_pts = df["z"].to_numpy() / 1000.0 if df["z"].max() > 100 else df["z"].to_numpy()
    else:
        z_pts = np.full_like(x_pts, 2.435)

    all_x_list.append(x_pts)
    all_y_list.append(y_pts)
    all_z_list.append(z_pts)
    all_t_list.append(df_time)

all_x = np.concatenate(all_x_list)
all_y = np.concatenate(all_y_list)
all_z = np.concatenate(all_z_list)
all_t = np.concatenate(all_t_list)

# Spatial bounding box pre-filtering in UTM meters
min_x_m, max_x_m = X_m.min() - 5000, X_m.max() + 5000
min_y_m, max_y_m = Y_m.min() - 5000, Y_m.max() + 5000

in_bounds = (all_x >= min_x_m) & (all_x <= max_x_m) & (all_y >= min_y_m) & (all_y <= max_y_m)
all_x, all_y, all_z, all_t = all_x[in_bounds], all_y[in_bounds], all_z[in_bounds], all_t[in_bounds]

max_time = min(all_t.max(), 7200.0)
min_time = all_t.min()
num_frames = 30
time_steps = np.linspace(min_time, max_time, num_frames)

# ===================================================================
# SECTION 3a: TERRAIN DEM WITH LOCAL DISK CACHING
# ===================================================================
cache_file = outpath / "terrain_dem_cache.npy"

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
    if cache_file.exists():
        print("Loading cached terrain DEM...")
        return np.load(cache_file)

    print("Fetching terrain DEM from web...")
    min_lon, max_lon = lon_grid.min(), lon_grid.max()
    min_lat, max_lat = lat_grid.min(), lat_grid.max()

    x_min, y_max = lonlat_to_tile(min_lon, min_lat, zoom)
    x_max, y_min = lonlat_to_tile(max_lon, max_lat, zoom)

    y_start, y_end = min(y_min, y_max), max(y_min, y_max)
    stitched_img = Image.new("RGB", ((x_max - x_min + 1) * 256, (y_end - y_start + 1) * 256))

    for i, x in enumerate(range(x_min, x_max + 1)):
        for j, y in enumerate(range(y_start, y_end + 1)):
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    stitched_img.paste(tile_img, (i * 256, j * 256))
            except Exception:
                pass

    img_arr = np.array(stitched_img, dtype=np.float32)
    elevation_m = (img_arr[:, :, 0] * 256.0 + img_arr[:, :, 1] + (img_arr[:, :, 2] / 256.0)) - 32768.0

    tile_min_lon, tile_max_lat = tile_to_lonlat(x_min, y_start, zoom)
    tile_max_lon, tile_min_lat = tile_to_lonlat(x_max + 1, y_end + 1, zoom)

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
    dem_km = np.clip(interp(pts).reshape(lat_grid.shape) / 1000.0, 0.0, None)
    np.save(cache_file, dem_km)
    return dem_km

terrain_z_km = fetch_terrain_grid_km(X_lon, Y_lat, zoom=10)
ground_footprint_z = terrain_z_km + 0.015

lat_axis_sorted = np.linspace(Y_lat.min(), Y_lat.max(), terrain_z_km.shape[0])
lon_axis_sorted = np.linspace(X_lon.min(), X_lon.max(), terrain_z_km.shape[1])

terrain_grid_correct = np.flipud(terrain_z_km) if Y_lat[0, 0] > Y_lat[-1, 0] else terrain_z_km
terrain_interp = RegularGridInterpolator(
    (lat_axis_sorted, lon_axis_sorted),
    terrain_grid_correct,
    bounds_error=False,
    fill_value=0.0,
)

VENT_ELEV_KM = float(terrain_interp([v_lat, v_lon])[0])
max_z_plume = max(float(all_z.max()), VENT_ELEV_KM + 1.0)

fig_temp, ax_temp = plt.subplots()
m = Basemap(
    resolution="h",
    projection="cyl",
    llcrnrlon=llcrnlon,
    llcrnrlat=llcrnlat,
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

# ===================================================================
# SECTION 4: HIGH-SPEED VECTORIZED PLUME CALCULATION
# ===================================================================
def compute_atmospheric_plume(current_time):
    time_mask = all_t <= current_time
    ash_grid = np.zeros_like(X_m)
    height_grid = np.zeros_like(X_m)
    weight_sum = np.zeros_like(X_m)

    if np.any(time_mask):
        xc_arr = all_x[time_mask]
        yc_arr = all_y[time_mask]
        zc_arr = all_z[time_mask]
        t_arr = all_t[time_mask]

        t0_val = np.maximum(t_arr, 0.1)
        t_diff = np.maximum(current_time - t0_val, 0.0)

        sigma = np.maximum(
            (4.0 * K_RISE * t0_val) + (1.6 * C_TURB * (t_diff**2.5)),
            (4.0 * K_RISE * t0_val) + (4.0 * K_DIFF * t_diff),
        )

        for xc, yc, zc, sig, td in zip(xc_arr, yc_arr, zc_arr, sigma, t_diff):
            if not (np.isfinite(xc) and np.isfinite(yc)):
                continue

            current_z_all = zc - (td * V_SETTLE_SPECTRUM_KM_S)
            valid_vt_mask = current_z_all >= 0.0
            if not np.any(valid_vt_mask):
                continue

            dx, dy = X_m - xc, Y_m - yc
            nakami_exp = -(dx**2 + dy**2) / sig

            if np.max(nakami_exp) > -60:
                base_weight = np.where(
                    nakami_exp > -60,
                    (1.0 / (sig * np.pi)) * np.exp(nakami_exp) * WEIGHT_PER_BIN,
                    0.0,
                )
                
                sum_valid_weight = np.sum(valid_vt_mask) * WEIGHT_PER_BIN
                ash_grid += base_weight * (sum_valid_weight / WEIGHT_PER_BIN)

                mean_z_valid = np.mean(current_z_all[valid_vt_mask])
                height_grid += base_weight * mean_z_valid * (sum_valid_weight / WEIGHT_PER_BIN)
                weight_sum += base_weight * (sum_valid_weight / WEIGHT_PER_BIN)

    ash_load_g_m2 = ash_grid * 1000.0
    masked_grid = np.where(ash_load_g_m2 >= levels[0], ash_load_g_m2, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_z = np.where(weight_sum > 0, height_grid / weight_sum, np.nan)

    plume_z = np.where(
        ~np.isnan(masked_grid),
        np.maximum(avg_z, ground_footprint_z + 0.05),
        np.nan,
    )
    return masked_grid, plume_z

def generate_eruption_column_mesh(max_height, plume_center_lon=None, plume_center_lat=None):
    top_lon = plume_center_lon if plume_center_lon is not None else v_lon
    top_lat = plume_center_lat if plume_center_lat is not None else v_lat

    r_base_deg = 0.0003
    r_top_deg = 0.0030

    n_theta, n_z = 24, 20
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z_vals = np.linspace(VENT_ELEV_KM, max_height, n_z)

    theta_grid, z_grid = np.meshgrid(theta, z_vals)

    t_factor = (z_grid - VENT_ELEV_KM) / max(max_height - VENT_ELEV_KM, 0.01)
    r_grid = r_base_deg + (r_top_deg - r_base_deg) * t_factor

    cx_grid = v_lon + (top_lon - v_lon) * t_factor
    cy_grid = v_lat + (top_lat - v_lat) * t_factor

    col_x = cx_grid + r_grid * np.cos(theta_grid)
    col_y = cy_grid + r_grid * np.sin(theta_grid)

    return col_x, col_y, z_grid

print("Computing plume animation frames...")
plume_frames_data = [compute_atmospheric_plume(t) for t in time_steps]

# ===================================================================
# SECTION 3b: SITE MARKERS & TOTAL ACCUMULATED PLUME ASH
# ===================================================================
final_load_frame, _ = plume_frames_data[-1]
total_ash_load_grid = np.nan_to_num(final_load_frame, nan=0.0)

dx = np.abs(X_m[0, 1] - X_m[0, 0])
dy = np.abs(Y_m[1, 0] - Y_m[0, 0])
cell_area_m2 = dx * dy

total_ash_grams = np.nansum(total_ash_load_grid) * cell_area_m2
total_ash_tonnes = total_ash_grams / 1e6
print(f"Total accumulated ash within expanded plume grid: {total_ash_tonnes:,.2f} metric tonnes")

load_interp = RegularGridInterpolator(
    (lat_axis_sorted, lon_axis_sorted),
    total_ash_load_grid,
    bounds_error=False,
    fill_value=0.0,
)

pts_x, pts_y, pts_z = [], [], []
pts_text, pts_hover, pts_color, pts_symbol, pts_size = [], [], [], [], []

vent_load = float(load_interp([v_lat, v_lon])[0])
pts_x.append(v_lon)
pts_y.append(v_lat)
pts_z.append(VENT_ELEV_KM + 0.25)
pts_text.append("VENT")
pts_hover.append(
    f"<b>Volcanic Vent</b><br>"
    f"Lat: {v_lat:.4f}°<br>"
    f"Lon: {v_lon:.4f}°<br>"
    f"Terrain Elev: {VENT_ELEV_KM:.3f} km<br>"
    f"Plume Ash Load: {vent_load:.2f} g/m²"
)
pts_color.append("#FF0000")
pts_symbol.append("diamond")
pts_size.append(12)

sites_csv = main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / "mapoutputs" / "totalload.csv"

if sites_csv.exists():
    df_sites = pd.read_csv(sites_csv)
    col_map = {str(c).strip().lower(): c for c in df_sites.columns}

    lat_col = next((col_map[c] for c in col_map if any(k in c for k in ["lat", "y_utm", "y"])), None)
    lon_col = next((col_map[c] for c in col_map if any(k in c for k in ["lon", "long", "x_utm", "x"])), None)
    name_col = next((col_map[c] for c in col_map if any(k in c for k in ["site", "name", "station", "loc"])), None)
    load_col = next((col_map[c] for c in col_map if any(k in c for k in ["load", "totalload", "ash"])), None)

    if lat_col and lon_col:
        for idx, row in df_sites.iterrows():
            raw_y = float(row[lat_col])
            raw_x = float(row[lon_col])
            s_name = str(row[name_col]) if name_col and pd.notna(row[name_col]) else f"Site_{idx+1}"
            csv_load_val = float(row[load_col]) if load_col and pd.notna(row[load_col]) else 0.0

            if abs(raw_x) > 1000 or abs(raw_y) > 1000:
                s_lon, s_lat = utm_to_latlon.transform(raw_x, raw_y)
            else:
                s_lat, s_lon = raw_y, raw_x

            if (llcrnlat <= s_lat <= urcrnrlat) and (llcrnlon <= s_lon <= urcrnrlon):
                site_ground_elev = float(terrain_interp([s_lat, s_lon])[0])
                model_site_load = float(load_interp([s_lat, s_lon])[0])
                csv_load_g = csv_load_val * 1000.0 if csv_load_val < 1000 else csv_load_val

                pts_x.append(s_lon)
                pts_y.append(s_lat)
                pts_z.append(site_ground_elev + 0.20)
                pts_text.append(s_name)
                pts_hover.append(
                    f"<b>Site: {s_name}</b><br>"
                    f"Lat: {s_lat:.4f}°<br>"
                    f"Lon: {s_lon:.4f}°<br>"
                    f"Terrain Elev: {site_ground_elev:.3f} km<br>"
                    f"<b>Plume Ash Load:</b> {model_site_load:.2f} g/m²<br>"
                    f"<b>CSV Total Load:</b> {csv_load_g:.2f} g/m² ({csv_load_val:.4f} kg/m²)"
                )
                pts_color.append("#00FFFF")
                pts_symbol.append("circle")
                pts_size.append(10)

overlay_points_trace = go.Scatter3d(
    x=pts_x,
    y=pts_y,
    z=pts_z,
    mode="markers+text",
    marker=dict(
        size=pts_size,
        color=pts_color,
        symbol=pts_symbol,
        line=dict(color="#000000", width=2),
    ),
    text=pts_text,
    textposition="top center",
    textfont=dict(size=11, color="#000000"),
    name="Monitoring Sites & Vent",
    hoverinfo="text",
    hovertext=pts_hover,
)

# ===================================================================
# SECTION 5: SCENE CONSTRUCTION & RENDER ANIMATION
# ===================================================================
fig = go.Figure()

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
        name="Terrain DEM",
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
        connectgaps=False,
        opacity=0.45,
        name="Ground Load Footprint",
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
        connectgaps=False,
        opacity=0.92,
        colorbar=dict(
            title="Ash Load (g/m²)",
            tickvals=levels,
            ticktext=[str(lvl) for lvl in levels],
        ),
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
        name="Eruption Column",
    )
)

for trace in coast_traces:
    fig.add_trace(trace)

fig.add_trace(overlay_points_trace)

plotly_frames = []
for idx, (t, (ash_data, z_plume)) in enumerate(zip(time_steps, plume_frames_data)):
    frame_traces = [
        go.Surface(
            x=X_lon,
            y=Y_lat,
            z=terrain_z_km,
            colorscale="Earth",
            cmin=0.0,
            cmax=2.5,
            showscale=False,
            opacity=0.75,
        ),
        go.Surface(
            x=X_lon,
            y=Y_lat,
            z=ground_footprint_z,
            surfacecolor=ash_data,
            colorscale=plotly_colorscale,
            cmin=cmin,
            cmax=cmax,
            connectgaps=False,
            showscale=False,
            opacity=0.45,
        ),
        go.Surface(
            x=X_lon,
            y=Y_lat,
            z=z_plume,
            surfacecolor=ash_data,
            colorscale=plotly_colorscale,
            cmin=cmin,
            cmax=cmax,
            connectgaps=False,
            opacity=0.92,
        ),
    ]

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

    for ct in coast_traces:
        frame_traces.append(ct)

    frame_traces.append(overlay_points_trace)
    plotly_frames.append(go.Frame(data=frame_traces, name=f"frame_{idx}"))

fig.frames = plotly_frames

fig.update_layout(
    title=f"3D Volcanic Column Mesh, Atmospheric Ash Load (g/m²) & Ground Load — Time: {int(time_steps[0])}s",
    scene=dict(
        xaxis=dict(title="Longitude (°E)", range=[llcrnlon, urcrnrlon]),
        yaxis=dict(title="Latitude (°N)", range=[llcrnlat, urcrnrlat]),
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

html_file = outpath / "ash_composite_v1.html"
fig.write_html(html_file)
print(f"-> Saved optimized 3D plot to: {html_file}")