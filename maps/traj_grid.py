import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import rioxarray
from xrspatial import hillshade

# 1. PATHS AND SITE PARAMETERS
main_dir = Path(__file__).resolve().parents[1]
dem_path = main_dir / "KANLAON" / 'kanlaonDEM' / "kanlaon_DEM.tif"
trajpath = main_dir / '2026709_files' /'2026709' / 'traj2026709' 
outpath = main_dir / '2026709_files' / 'plumegridv1'
outpath.mkdir(parents=True, exist_ok=True)

VENT_X = 514495.0
VENT_Y = 1150889.0

# 2. EXACT TRAJ-FALL PARAMETERS
K_RISE = 500.0   # Horizontal eddy diffusion coefficient (m^2/s)
C_FALL = 0.04    # Atmospheric boundary layer turbulence constant
FTT = 3600.0     # Falling Time Threshold in seconds (1 hour cutoff)

csvs = list(trajpath.glob('*mms-1.csv'))

# LOAD DEM AND INITIALIZE TERRAIN LAYOUT
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
Y_km = (Y_m - VENT_Y) / 1000.0

# 3. TRAJ-FALL PHYSICS CONFIGURATION
if not csvs:
    print('No trajectory files found (*mms-1.csv)')
else:
    for file_path in csvs:
        file_name = os.path.basename(file_path)
        print(f"Generating cell-grid block simulation for: {file_name}")
        d_name = file_name.replace('mms-1.csv', '').capitalize()
        traj = pd.read_csv(file_path)

        x_centers = traj['x0'].values
        y_centers = traj['y0'].values

        if 'time' in traj.columns:
            t_travel = traj['time'].values
        elif 't' in traj.columns:
            t_travel = traj['t'].values
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
                sigma = (4.0 * K_RISE * FTT) + (1.6 * C_FALL * (t_calc ** 2.5))
            
            dx = X_m - xc
            dy = Y_m - yc
            nakami_exp = -(dx**2 + dy**2) / sigma
            
            if np.max(nakami_exp) > -40:
                weight_distribution = np.where(
                    nakami_exp > -40, 
                    1.0 / (sigma * np.pi) * np.exp(nakami_exp) * 1000.0, 
                    0.0
                )
                ash_accumulation_grid += weight_distribution

        # 4. RENDER CELL BLOCK STRUCTURE MAPS (OPTIMIZED & VECTORIZED)
        fig, ax = plt.subplots(figsize=(10, 8))

        # Show terrain hillshade base map
        ax.imshow(hillshade_img, cmap='gray', extent=dem_extent, origin='upper', alpha=0.7)

        max_load = ash_accumulation_grid.max()
        if max_load > 0:
            # Mask out background/noise grid values (replaces the nested loop threshold check)
            masked_grid = np.where(ash_accumulation_grid > max_load * 0.02, ash_accumulation_grid, np.nan)
            
            # Render the entire block layout grid instantly
            mesh = ax.pcolormesh(
                X_km, Y_km, masked_grid,
                cmap='YlOrRd',
                alpha=0.75,
                edgecolor='black',
                linewidth=0.2,
                shading='nearest'
            )
            
            # Map the colorbar explicitly to the output mesh layout
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.04)
            cbar.set_label('Ash Accumulation Weight Intensity', rotation=270, labelpad=15)

        # Mark the Kanlaon crater vent position at local origin (0,0)
        ax.plot(0, 0, '^r', markersize=11, mew=2, label='Kanlaon Vent')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

        ax.set_xlabel('Distance from Vent East/West (km)')
        ax.set_ylabel('Distance from Vent North/South (km)')
        ax.set_title(f'Trajectory Load ({d_name})')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.4)

        plt.savefig(outpath / f'{d_name}_block_plume_model.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

print("rock w u")