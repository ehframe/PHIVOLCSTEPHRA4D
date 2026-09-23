# importing all needed libraries
import sys
import os
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from pathlib import Path
from pyproj import Transformer
from mpl_toolkits.basemap import Basemap

main_dir = Path(__file__).resolve().parents[1]

sys.path.append(str(main_dir))

from variable import date, date_slash, date_clear, file_hour, start_hour, end_hour, edate

sitecsv = main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / 'mapoutputs' / 'totalload.csv'
trajpath = main_dir / f'{date_clear}_{start_hour:02d}0000_files_v1' / f'{date_clear}' / f'traj{date_clear}'
outputdir = main_dir / f"{date_clear}_{start_hour:02d}0000_files_v1" / 'Load and Trajectory (Overlay)'
outputdir.mkdir(parents=True, exist_ok=True)

# Vent Coords in UTM Zone 51N (Kanlaon)
VENT_X = 514495.0
VENT_Y = 1150889.0

# Coordinate Converter
utm_to_latlon = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
v_lon, v_lat = utm_to_latlon.transform(VENT_X, VENT_Y)

# --- COLOR PALETTES ---
# 1. Site Load Palette (Keep original custom palette)
colorload = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]
ld_level = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]
ld_marker = colors.ListedColormap(colorload[:len(ld_level) - 1])
ld_norm = colors.BoundaryNorm(ld_level, ld_marker.N)

# 2. Trajectory Release Altitude Palette (Changed to 'Viridis' to contrast with Site Load)
at_level = [0, 1, 2, 3, 4, 5, 6, 7]
at_cmap = plt.cm.get_cmap('viridis', len(at_level) - 1)
at_norm = colors.BoundaryNorm(at_level, at_cmap.N)

# READ SITE LOAD DATA & CONVERT TO LAT/LON
if not sitecsv.exists():
    raise FileNotFoundError(f"Input file not found at path: {sitecsv}")

load = pd.read_csv(sitecsv)

if 'Long' in load.columns and 'Lat' in load.columns:
    if load['Long'].max() > 180:
        ld_lon, ld_lat = utm_to_latlon.transform(load['Long'].values, load['Lat'].values)
    else:
        ld_lon, ld_lat = load['Long'].values, load['Lat'].values
else:
    raise KeyError("Site load CSV must contain 'Long' and 'Lat' columns.")

ld_z = load['total_load'].values

# Define Basemap Bounds
llcrnlon, urcrnrlon = 122.800842, 123.801148
llcrnlat, urcrnrlat = 10.199636, 10.594028

# ASH TRAJECTORY DATA
csvs = list(trajpath.glob('*mms-1.csv'))

### PLOTTING ###
if not csvs:
    print('heehee beat it (walang -mms)')
else:
    all_trajs = []

    for file_path in csvs:
        file_name = os.path.basename(file_path)
        print(f"Processing trajectory: {file_name}")
        d_name = file_name.replace('mms-1.csv', '').capitalize()
        traj = pd.read_csv(file_path)
        
        all_trajs.append(traj)

        # Convert Trajectory UTM coordinates to Lat/Lon
        traj_lon, traj_lat = utm_to_latlon.transform(traj['x0'].values, traj['y0'].values)
        traj_z = traj['z0'].values / 1000.0  # Convert altitude to km

        fig, ax = plt.subplots(figsize=(10, 8))

        # 1. Initialize Basemap
        m = Basemap(
            resolution="h",
            projection="lcc",
            lat_0=10.4120,
            lon_0=123.2500,
            llcrnrlon=llcrnlon,
            llcrnrlat=llcrnlat,
            urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat,
            ax=ax,
        )

        # Base Geography Layers
        m.drawmapboundary(fill_color="#EBF4FA", zorder=0)
        m.fillcontinents(color="#E6E6E6", lake_color="#EBF4FA", zorder=1)
        m.drawcoastlines(linewidth=0.8, color="#222222", zorder=2)

        site_map_x, site_map_y = m(ld_lon, ld_lat)
        v_map_x, v_map_y = m(v_lon, v_lat)
        traj_map_x, traj_map_y = m(traj_lon, traj_lat)

        # Separate points into airborne (> 0 km) and ground-reaching (<= 0 km)
        mask_ground = traj['z0'] <= 0.0
        mask_air = ~mask_ground

        # 2a. Plot Airborne Trajectory Points (No outline, low opacity)
        m.scatter(
            traj_map_x[mask_air], traj_map_y[mask_air],
            c=traj_z[mask_air], cmap=at_cmap, norm=at_norm,
            s=12, marker='o', edgecolors='none',
            alpha=0.25, zorder=3
        )

        # 2b. Plot Ground-Reaching Points (0 km) (Solid black border, full opacity)
        at = m.scatter(
            traj_map_x[mask_ground], traj_map_y[mask_ground],
            c=traj_z[mask_ground], cmap=at_cmap, norm=at_norm,
            s=25, marker='o', edgecolors='black', linewidths=0.6,
            alpha=1.0, label='Trajectory Ground Touch (0 km)', zorder=4
        )

        # 3. Individual Site Observations as Solid Dots
        site_dots = m.scatter(
            site_map_x, site_map_y,
            c=ld_z, cmap=ld_marker, norm=ld_norm,
            s=35, marker='o', edgecolors='black', linewidths=0.8,
            alpha=1.0, label='Site Load', zorder=5
        )

        # 4. Mark Crater Vent Position
        m.plot(v_map_x, v_map_y, marker='^', color='red', markersize=10, markeredgecolor='black', label='VENT', zorder=6)

        # Grid Parallels & Meridians
        m.drawparallels(np.arange(10.20, 10.60, 0.1), labels=[1, 0, 0, 0], fontsize=9, linewidth=0.3, zorder=7)
        m.drawmeridians(np.arange(122.80, 123.90, 0.2), labels=[0, 0, 0, 1], fontsize=9, linewidth=0.3, zorder=7)

        plt.title(f'TRAJECTORY AND SITE LOAD ({d_name})', fontweight='bold', pad=14, fontsize=12)

        # Colorbars
        cbar_at = fig.colorbar(at, ax=ax, boundaries=at_level, shrink=0.85, pad=0.03)
        cbar_at.set_label('z0 Altitude (km)', rotation=270, labelpad=15, fontweight='bold')

        cbar_ld = fig.colorbar(site_dots, ax=ax, boundaries=ld_level, shrink=0.85, pad=0.08)
        cbar_ld.set_label('Total Load (kg/m^2)', rotation=270, labelpad=15, fontweight='bold')

        ax.legend(loc='upper left', framealpha=0.9)

        fig.canvas.draw()
        save_path = outputdir / f'{d_name}_combined_overlay.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # =========================================================================
    # COMPOSITE PLOT (Deduplicated Trajectories)
    # =========================================================================
    print("Generating composite trajectory overlay...")
    composite_df = pd.concat(all_trajs, ignore_index=True)

    # Group by unique coordinates and find the lowest altitude reached
    dedup_traj = composite_df.groupby(['x0', 'y0'], as_index=False)['z0'].min()

    comp_lon, comp_lat = utm_to_latlon.transform(dedup_traj['x0'].values, dedup_traj['y0'].values)
    comp_z = dedup_traj['z0'].values / 1000.0

    fig, ax = plt.subplots(figsize=(10, 8))

    m = Basemap(
        resolution="h",
        projection="lcc",
        lat_0=10.4120,
        lon_0=123.2500,
        llcrnrlon=llcrnlon,
        llcrnrlat=llcrnlat,
        urcrnrlon=urcrnrlon,
        urcrnrlat=urcrnrlat,
        ax=ax,
    )

    m.drawmapboundary(fill_color="#EBF4FA", zorder=0)
    m.fillcontinents(color="#E6E6E6", lake_color="#EBF4FA", zorder=1)
    m.drawcoastlines(linewidth=0.8, color="#222222", zorder=2)

    site_map_x, site_map_y = m(ld_lon, ld_lat)
    v_map_x, v_map_y = m(v_lon, v_lat)
    comp_map_x, comp_map_y = m(comp_lon, comp_lat)

    comp_ground = dedup_traj['z0'] <= 0.0
    comp_air = ~comp_ground

    # Plot Airborne Points (No borders, transparent)
    m.scatter(
        comp_map_x[comp_air], comp_map_y[comp_air],
        c=comp_z[comp_air], cmap=at_cmap, norm=at_norm,
        s=12, marker='o', edgecolors='none',
        alpha=0.25, zorder=3
    )

    # Plot Ground Touch Points (0 km) (Black border, full opacity)
    at_comp = m.scatter(
        comp_map_x[comp_ground], comp_map_y[comp_ground],
        c=comp_z[comp_ground], cmap=at_cmap, norm=at_norm,
        s=25, marker='o', edgecolors='black', linewidths=0.6,
        alpha=1.0, label='Composite Ground Touch (0 km)', zorder=4
    )

    # Site Load Observations
    site_dots = m.scatter(
        site_map_x, site_map_y,
        c=ld_z, cmap=ld_marker, norm=ld_norm,
        s=35, marker='o', edgecolors='black', linewidths=0.8,
        alpha=1.0, label='Site Load', zorder=5
    )

    # Crater Vent
    m.plot(v_map_x, v_map_y, marker='^', color='red', markersize=10, markeredgecolor='black', label='VENT', zorder=6)

    # Grid Lines
    m.drawparallels(np.arange(10.20, 10.60, 0.1), labels=[1, 0, 0, 0], fontsize=9, linewidth=0.3, zorder=7)
    m.drawmeridians(np.arange(122.80, 123.90, 0.2), labels=[0, 0, 0, 1], fontsize=9, linewidth=0.3, zorder=7)

    plt.title('COMPOSITE TRAJECTORIES AND SITE LOAD OVERLAY', fontweight='bold', pad=14, fontsize=12)

    # Dual Colorbars Layout
    cbar_at = fig.colorbar(at_comp, ax=ax, boundaries=at_level, shrink=0.85, pad=0.03)
    cbar_at.set_label('z0 Altitude (km)', rotation=270, labelpad=15, fontweight='bold')

    cbar_ld = fig.colorbar(site_dots, ax=ax, boundaries=ld_level, shrink=0.85, pad=0.08)
    cbar_ld.set_label('Total Load (kg/m^2)', rotation=270, labelpad=15, fontweight='bold')

    ax.legend(loc='upper left', framealpha=0.9)

    fig.canvas.draw()
    comp_save_path = outputdir / 'Composite_trajectory_overlay.png'
    plt.savefig(comp_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('omke na')