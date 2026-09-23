from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")  # Non-GUI backend for Basemap processing
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import Transformer

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
outpath = main_dir / f"{date_clear}_{start_hour:02d}0000_files" / "3D Load in a grid v2"
outpath.mkdir(parents=True, exist_ok=True)

VENT_X = 514495.0
VENT_Y = 1150889.0
VENT_ELEV_KM = 2.435  # Kanlaon summit elevation in km ASL
VENT_ELEV_M = VENT_ELEV_KM * 1000.0
MAX_PLUME_TOP_KM = 12.0  # Altitude ceiling in km ASL

# Tephra4D Dispersion Constants
K_RISE = 300.0  # K_rise (m^2/s)
K_DIFF = 100.0  # Horizontal diffusion K (m^2/s)
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)  # C factor
VERT_SCALE = 0.1  # Anisotropic scale factor for vertical dispersion (prevents huge vertical blobs)

latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
utm_to_latlon = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

v_lon, v_lat = utm_to_latlon.transform(VENT_X, VENT_Y)

# Map Extent
llcrnlon, urcrnrlon = 122.800842, 124.105784
llrcnlat, urcrnrlat = 10.199636, 10.690102

# -------------------------------------------------------------------
# 3D VOLUMETRIC GRID SETUP (X, Y, Z)
# -------------------------------------------------------------------
lon_vec = np.linspace(llcrnlon, urcrnrlon, 50)
lat_vec = np.linspace(llrcnlat, urcrnrlat, 40)
z_km_vec = np.linspace(0.0, MAX_PLUME_TOP_KM, 30)

# Create 3D Meshgrids for Lat/Lon, UTM Coordinates, and Altitude
X_lon_3d, Y_lat_3d, Z_km_3d = np.meshgrid(lon_vec, lat_vec, z_km_vec, indexing="ij")
X_m_3d, Y_m_3d = latlon_to_utm.transform(X_lon_3d, Y_lat_3d)
Z_m_3d = Z_km_3d * 1000.0  # Altitude in meters ASL

# Flat 2D meshgrid for ocean base surface plane
X_lon_2d, Y_lat_2d = np.meshgrid(lon_vec, lat_vec)

# Discrete Concentration Levels
levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
cmin, cmax = levels[0], levels[-1]

norm_levels = [(lvl - cmin) / (cmax - cmin) for lvl in levels]
hex_colors = [
    "#FFFFCC",
    "#FFEDA0",
    "#FED976",
    "#FEB24C",
    "#FD8D3C",
    "#FC4E2A",
    "#E31A1C",
    "#BD0026",
    "#800026",
]

plotly_colorscale = []
for n_val, hex_c in zip(norm_levels, hex_colors):
    plotly_colorscale.append([n_val, hex_c])

csvs = list(trajpath.glob("*mms-1.csv"))
if not csvs:
    raise FileNotFoundError("No trajectory files found in directory.")

print("Extracting geographical coastlines from Basemap...")
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
paths = coastlines.get_paths()

coast_traces = []
for path in paths:
    verts = path.vertices
    coast_traces.append(
        go.Scatter3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=np.zeros(len(verts)),
            mode="lines",
            line=dict(color="#333333", width=2.5),
            showlegend=False,
            hoverinfo="none",
        )
    )

plt.close(fig_temp)

# -------------------------------------------------------------------
# 2. LOAD & COMPOSITE TRAJECTORIES
# -------------------------------------------------------------------
for file_path in csvs:
    d_name = file_path.stem.replace("mms-1", "").capitalize()
    print(f"Processing trajectory: {d_name}")
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

    # Convert Z coordinates to meters ASL
    if raw_z.max() < 1000.0:
        z_pts_m = (raw_z * 1000.0) + VENT_ELEV_M if raw_z.min() < VENT_ELEV_KM else raw_z * 1000.0
    else:
        z_pts_m = raw_z + VENT_ELEV_M if raw_z.min() < VENT_ELEV_M else raw_z

    max_time = min(df_time.max(), 7200.0)
    min_time = df_time.min()
    num_frames = 30
    time_steps = np.linspace(min_time, max_time, num_frames)

    def compute_3d_ash_volume(current_time):
        time_mask = df_time <= current_time
        ash_volume = np.zeros_like(X_m_3d)

        if np.any(time_mask):
            xc_arr = x_pts[time_mask]
            yc_arr = y_pts[time_mask]
            zc_arr = z_pts_m[time_mask]
            t_arr = df_time[time_mask]

            for xc, yc, zc, t in zip(xc_arr, yc_arr, zc_arr, t_arr):
                if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(zc)):
                    continue

                t0_val = max(t, 0.1)
                t_diff = max(current_time - t0_val, 0.0)

                sigma1 = (4.0 * K_RISE * t0_val) + (1.6 * C_TURB * (t_diff**2.5))
                sigma2 = (4.0 * K_RISE * t0_val) + (4.0 * K_DIFF * t_diff)
                sigma = max(sigma1, sigma2)

                # Scaled 3D Spatial Distance (horizontal vs vertical factor)
                dx = X_m_3d - xc
                dy = Y_m_3d - yc
                dz = (Z_m_3d - zc) / VERT_SCALE

                nakami_exp = -(dx**2 + dy**2 + dz**2) / sigma

                if np.max(nakami_exp) > -40:
                    weight = np.where(
                        nakami_exp > -40,
                        (1.0 / (sigma * np.pi)) * np.exp(nakami_exp) * 1000.0,
                        0.0,
                    )
                    ash_volume += weight

        return np.where(ash_volume >= levels[0], ash_volume, 0.0)

    volumes = [compute_3d_ash_volume(t) for t in time_steps]

    fig = go.Figure()

    # Add Coastlines
    for trace in coast_traces:
        fig.add_trace(trace)

    # Ocean Base Plane (Z = 0 km ASL)
    fig.add_trace(
        go.Surface(
            x=X_lon_2d,
            y=Y_lat_2d,
            z=np.zeros_like(X_lon_2d),
            colorscale=[[0, "#A6CAE0"], [1, "#A6CAE0"]],
            showscale=False,
            opacity=0.4,
            hoverinfo="none",
            name="Ocean Base (0.0 km)",
        )
    )

    # Vent Marker Line
    fig.add_trace(
        go.Scatter3d(
            x=[v_lon, v_lon],
            y=[v_lat, v_lat],
            z=[0.0, VENT_ELEV_KM],
            mode="lines+markers",
            line=dict(color="red", width=6),
            marker=dict(size=[0, 8], color="red", symbol="diamond"),
            name="Kanlaon Vent (2.435 km)",
        )
    )

    # Initial 3D Ash Isosurface Volume
    initial_vol = volumes[0]
    fig.add_trace(
        go.Isosurface(
            x=X_lon_3d.flatten(),
            y=Y_lat_3d.flatten(),
            z=Z_km_3d.flatten(),
            value=initial_vol.flatten(),
            isomin=cmin,
            isomax=cmax,
            surface_count=5,
            colorscale=plotly_colorscale,
            caps=dict(x_show=False, y_show=False, z_show=False),
            lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.2, specular=0.2),
            colorbar=dict(
                title="Total Active Ash Concentration",
                tickvals=levels,
                ticks="outside",
            ),
            name="Ash Cloud Volume",
        )
    )

    # Animation Frames
    plotly_frames = []
    for idx, (t, vol_data) in enumerate(zip(time_steps, volumes)):
        frame_traces = list(coast_traces)
        frame_traces.append(
            go.Surface(
                x=X_lon_2d,
                y=Y_lat_2d,
                z=np.zeros_like(X_lon_2d),
                colorscale=[[0, "#A6CAE0"], [1, "#A6CAE0"]],
                showscale=False,
                opacity=0.4,
                hoverinfo="none",
            )
        )
        frame_traces.append(
            go.Scatter3d(
                x=[v_lon, v_lon],
                y=[v_lat, v_lat],
                z=[0.0, VENT_ELEV_KM],
                mode="lines+markers",
                line=dict(color="red", width=6),
                marker=dict(size=[0, 8], color="red", symbol="diamond"),
            )
        )
        frame_traces.append(
            go.Isosurface(
                x=X_lon_3d.flatten(),
                y=Y_lat_3d.flatten(),
                z=Z_km_3d.flatten(),
                value=vol_data.flatten(),
                isomin=cmin,
                isomax=cmax,
                surface_count=5,
                colorscale=plotly_colorscale,
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )

        plotly_frames.append(go.Frame(data=frame_traces, name=f"frame_{idx}"))

    fig.frames = plotly_frames

    # Controls & Layout
    sliders = [
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
    ]

    fig.update_layout(
        title=f"Total Ash Airborne Propagation Composite 3D — Time: {int(time_steps[0])}s",
        scene=dict(
            xaxis=dict(title="Longitude (°E)", range=[llcrnlon, urcrnrlon]),
            yaxis=dict(title="Latitude (°N)", range=[llrcnlat, urcrnrlat]),
            zaxis=dict(
                title="Altitude ASL (km)",
                range=[0.0, MAX_PLUME_TOP_KM],
                dtick=2.0,
            ),
            # Controlled visual aspect ratio (lowered Z scale)
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.0, z=0.35),
            camera=dict(eye=dict(x=-1.3, y=-1.3, z=0.8)),
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
        sliders=sliders,
    )

    html_file = outpath / f"{d_name}_ash_propagation_3d.html"
    fig.write_html(html_file)
    print(f"-> Saved plot: {html_file}")

print("all traj donezo")