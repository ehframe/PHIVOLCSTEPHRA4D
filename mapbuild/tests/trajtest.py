import glob
from pathlib import Path
import matplotlib.colors as colors
import matplotlib.font_manager as fm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import earthpy.spatial as es

# --- 1. SETUP PATHS & ANCHOR GEOMETRY ---
main_dir = Path(__file__).resolve().parents[2]

VENT_X = 514495.0
VENT_Y = 1150889.0
dem_borders = 5.0

# Define custom color palette and discrete bounds
pallet = ["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"]
levels = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]

marker_colored = colors.ListedColormap(pallet[:len(levels)-1])
custom_norm = colors.BoundaryNorm(levels, marker_colored.N)

# Set up text fonts
fp = fm.FontProperties(size=12, weight='bold')
fp2 = fm.FontProperties(size=10)

# --- 2. LOAD DEM & COMPUTE HILLSHADE ---
dem_path = main_dir / "KANLAON" / "kanlaonDEM" / "kanlaon_DEM.tif"
with rasterio.open(dem_path) as dem:
    dem_data = dem.read(1)
    bounds = dem.bounds
    dem_data[dem_data < 0] = np.nan

# Scale bounding box to kilometers relative to Vent position
x_min = (bounds.left - VENT_X) / 1000.0
x_max = (bounds.right - VENT_X) / 1000.0
y_min = (bounds.bottom - VENT_Y) / 1000.0
y_max = (bounds.top - VENT_Y) / 1000.0
dem_extent = (x_min, x_max, y_min, y_max)

plot_xlim = (x_min - dem_borders, x_max + dem_borders)
plot_ylim = (y_min - dem_borders, y_max + dem_borders)

# Compute terrain relief structure
hillshade = es.hillshade(dem_data, azimuth=360, altitude=45)

# --- 3. SEPARATE IMAGE GENERATOR PER TIME STEP ---
def draw_separate_time_plots(erno_str):
    """
    Generates isolated image files for every individual time step interval.
    Each image contains a vertical column layout matching the 4 velocity bins.
    """
    # Load site metadata and coordinates
    bg = pd.read_csv(main_dir / "KANLAON" / "brgycindem_table.csv")
    bg['x_km'] = (bg['Long'] - VENT_X) / 1000.0
    bg['y_km'] = (bg['Lat'] - VENT_Y) / 1000.0
    
    velocity_bins = [0.5, 1.5, 3.0, 6.0] 
    time_intervals = [0, 20, 40, 60, 80, 100]  # Each will get its own PNG file

    # Loop over every time interval to build separate plots
    for t_min in time_intervals:
        print(f"Generating isolated plot layer for: {t_min} minutes...")
        
        # Setup figure layout optimized for a single vertical column stack
        fig = plt.figure(figsize=(6, 12), dpi=300)
        g_fig = gs.GridSpec(9, 1, hspace=0.4) # 9 rows high, but only 1 column wide
        
        # Find trajectory path files for this specific time block
        traj_pattern = str(main_dir / "mapbuild" / f"*_er{erno_str}_{t_min}min*.csv")
        traj_files = glob.glob(traj_pattern)

        # Loop to create the velocity comparison rows inside this specific time frame
        for row_idx, v_bin in enumerate(velocity_bins):
            # Form grid axis block spans
            ax = fig.add_subplot(g_fig[row_idx*2 : row_idx*2 + 2, 0])
            
            # Layer 1: Base Topography Map
            ax.imshow(hillshade, cmap='gray', extent=dem_extent, origin="upper", alpha=0.6, zorder=0)
            
            # Layer 2: Ground Observation Sites
            sc = ax.scatter(bg['x_km'], bg['y_km'], 
                            c=bg.get('total_load', np.zeros(len(bg))), 
                            cmap=marker_colored, norm=custom_norm,
                            s=25, edgecolors='black', linewidth=0.4, zorder=3)
            
            # Layer 3: Overlay Simulation Trajectory Lines
            if traj_files:
                for tf in traj_files[:3]:
                    try:
                        traj_df = pd.read_csv(tf)
                        sub_traj = traj_df[traj_df['vel'] == v_bin] if 'vel' in traj_df.columns else traj_df
                        
                        t_x = (sub_traj.iloc[:, 2] - VENT_X) / 1000.0
                        t_y = (sub_traj.iloc[:, 3] - VENT_Y) / 1000.0
                        ax.plot(t_x, t_y, color='red', linewidth=0.7, linestyle='dashed', alpha=0.8, zorder=2)
                    except Exception:
                        pass

            # Layer 4: Volcanic Vent Icon
            ax.plot(0, 0, marker='^', color='red', markersize=10, markeredgecolor='black', zorder=4)
            
            # Subplot Framing adjustments
            ax.set_aspect('equal')
            ax.set_xlim(*plot_xlim)
            ax.set_ylim(*plot_ylim)
            
            # Add descriptive text to the left side and inside headers
            ax.set_ylabel('Distance from Vent (km)', fontproperties=fp2)
            ax.set_title(f"Terminal Velocity: {v_bin} m/s", fontproperties=fp2, loc='left')
            
            if row_idx == 3:
                ax.set_xlabel('Distance from Vent (km)', fontproperties=fp2)

        # --- COLOR BAR AXIS (Placed in the final row slice) ---
        ax_bar = fig.add_subplot(g_fig[-1, 0])
        cbar = fig.colorbar(sc, cax=ax_bar, orientation='horizontal', boundaries=levels)
        cbar.set_label('Total Ground Accumulation Load ($kg/m^2$)', fontproperties=fp2)
        cbar.ax.tick_params(labelsize=8)

        # Individual Title Heading
        fig.suptitle(f"Eruption Profile Analysis\nEvent ID: {erno_str} | Time: +{t_min} min", fontproperties=fp, y=0.96)
        
        # Save this step out as its own separate filename image asset
        output_path = main_dir / 'mapbuild' / f'kanlaon_{erno_str}_{t_min}min.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    print(f"\nDone! All single time interval charts exported directly to: {main_dir / 'mapbuild'}/")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    draw_separate_time_plots("2026315")