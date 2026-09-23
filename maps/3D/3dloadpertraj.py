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
outpath = main_dir / f"{date_clear}_{start_hour:02d}0000_files" / "3D Load in a grid"
outpath.mkdir(parents=True, exist_ok=True)

VENT_X = 514495.0
VENT_Y = 1150889.0
VENT_ELEV_KM = 2.435  # Kanlaon summit elevation in km ASL
MAX_PLUME_TOP_KM = 12.0  # Top plume altitude ceiling in km ASL

# Tephra4D Dispersion Constants
K_RISE = 300.0  # K_rise (m^2/s)
K_DIFF = 100.0  # Horizontal diffusion K (m^2/s)
C_TURB = 2.5 * K_DIFF / (3600.0**1.5)  # C factor
TIME_WINDOW = 60.0  # Active particle window (seconds)

latlon_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
utm_to_latlon = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

v_lon, v_lat = utm_to_latlon.transform(VENT_X, VENT_Y)

# Map Extent
llcrnlon, urcrnrlon = 122.800842, 124.105784
llrcnlat, urcrnrlat = 10.199636, 10.690102

# Grid Mesh Setup
lon_utm = np.linspace(llcrnlon, urcrnrlon, 100)
lat_utm = np.linspace(llrcnlat, urcrnrlat, 80)
X_lon, Y_lat = np.meshgrid(lon_utm, lat_utm)
X_m, Y_m = latlon_to_utm.transform(X_lon, Y_lat)

# Discrete Concentration Levels
levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
cmin, cmax = levels[0], levels[-1]

# Properly Normalized Plotly Colorscale [0.0 to 1.0]
norm_levels = [(lvl - cmin) / (cmax - cmin) for lvl in levels]
hex_colors = [
    "#FFFFCC",  # 0.01 - Light Yellow
    "#FFEDA0",  # 0.05
    "#FED976",  # 0.10
    "#FEB24C",  # 0.25
    "#FD8D3C",  # 0.50 - Bright Orange
    "#FC4E2A",  # 1.00
    "#E31A1C",  # 2.50 - Red
    "#BD0026",  # 5.00
    "#800026",  # 10.0 - Dark Red/Purple
]

plotly_colorscale = [
    [0.0, "rgba(0,0,0,0)"]
]  # Completely transparent for zero concentrations
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


#load traj and run for 3d

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

    if "x0" in df.columns and "y0" in df.columns:
        x_pts = df["x0"].to_numpy()
        y_pts = df["y0"].to_numpy()
    else:
        x_pts = df["x"].to_numpy()
        y_pts = df["y"].to_numpy()

    max_time = min(df_time.max(), 7200.0)
    min_time = df_time.min()
    num_frames = 30
    time_steps = np.linspace(min_time, max_time, num_frames)

    def compute_ash_grid(current_time):
            time_mask = df_time <= current_time
            ash_grid = np.zeros_like(X_m)

            if np.any(time_mask):
                xc_arr = x_pts[time_mask]
                yc_arr = y_pts[time_mask]
                t_arr = df_time[time_mask]

                for xc, yc, t in zip(xc_arr, yc_arr, t_arr):
                    if not (np.isfinite(xc) and np.isfinite(yc)):
                        continue

                    t0_val = max(t, 0.1)
                    t_diff = max(current_time - t0_val, 0.0)

                    sigma1 = (4.0 * K_RISE * t0_val) + (1.6 * C_TURB * (t_diff**2.5))
                    sigma2 = (4.0 * K_RISE * t0_val) + (4.0 * K_DIFF * t_diff)
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
                        ash_grid += weight

            masked_grid = np.where(ash_grid >= levels[0], ash_grid, 0.0)

            z_height_km = np.where(
                masked_grid > 0,
                VENT_ELEV_KM
                + (np.log10(masked_grid + 1.0) / np.log10(11.0))
                * (MAX_PLUME_TOP_KM - VENT_ELEV_KM),
                np.nan,
            )

            return masked_grid, z_height_km


    grids_and_heights = [compute_ash_grid(t) for t in time_steps]

#plotly smth bs
    fig = go.Figure()

    # Add Coastlines from Basemap to Sea Level Floor
    for trace in coast_traces:
        fig.add_trace(trace)

    # Ocean/Base Plane (Sea Level Z=0.0 km)
    fig.add_trace(
        go.Surface(
            x=X_lon,
            y=Y_lat,
            z=np.zeros_like(X_lon),
            colorscale=[[0, "#A6CAE0"], [1, "#A6CAE0"]],
            showscale=False,
            opacity=0.4,
            hoverinfo="none",
            name="Ocean Base (0.0 km)",
        )
    )

    # Vent Marker Line: Sea Level (0.0 km) -> Vent Summit (2.435 km)
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

    # Initial Ash Surface
    initial_color, initial_z = grids_and_heights[0]

    fig.add_trace(
        go.Surface(
            x=X_lon,
            y=Y_lat,
            z=initial_z,
            surfacecolor=initial_color,
            colorscale=plotly_colorscale,
            cmin=cmin,
            cmax=cmax,
            lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.2, specular=0.2),
            colorbar=dict(
                title="Total Active Ash Concentration",
                tickvals=levels,
                ticks="outside",
            ),
            name="Ash Cloud",
        )
    )

    # Generate Animation Frames
    plotly_frames = []
    for idx, (t, (color_data, z_data)) in enumerate(zip(time_steps, grids_and_heights)):
        # Build frame data traces: Coastlines + Base + Vent + Plume Surface
        frame_traces = list(coast_traces)
        frame_traces.append(
            go.Surface(x=X_lon, y=Y_lat, z=np.zeros_like(X_lon))
        )  # Base Ocean
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
            go.Surface(x=X_lon, y=Y_lat, z=z_data, surfacecolor=color_data)
        )

        plotly_frames.append(go.Frame(data=frame_traces, name=f"frame_{idx}"))

    fig.frames = plotly_frames

    # Interactive Sliders and Play Controls
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
            aspectratio=dict(x=1.2, y=1.0, z=0.5),
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

        # Output Interactive Web Page
    html_file = outpath / f"{d_name}_ash_propagation_3d.html"
    fig.write_html(html_file)
    print(f"-> Saved plot: {html_file}")
print('all traj donezo')