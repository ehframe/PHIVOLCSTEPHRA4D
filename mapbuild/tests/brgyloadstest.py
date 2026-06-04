import numpy as np
import matplotlib.pyplot as plt
import rasterio
import earthpy.spatial as es
from pathlib import Path

# --- Configuration & Paths ---
main_dir = Path(__file__).resolve().parents[1]  # Adjust this to your main directory path
dem_path = main_dir / "KANLAON" / "kanlaonDEM" / "kanlaon_DEM.tif"

# Vent coordinates matching your project's reference system (meters)
VENT_X = 514495.0
VENT_Y = 1150889.0

# Mock scatter data (Replace with your actual x, y, z variables)
scatter_x = np.array([514495.0, 520000.0, 510000.0, 518000.0])
scatter_y = np.array([1150889.0, 1155000.0, 1145000.0, 1151000.0])
scatter_z = np.array([100, 50, 25, 80])

# --- 1. Load DEM and Extract Spatial Metadata Bounds ---
with rasterio.open(dem_path) as src:
    dem_data = src.read(1)
    bounds = src.bounds  # Restoring automatic GeoTIFF bounds mapping
    dem_data = np.where(dem_data < 0, np.nan, dem_data)

# --- 2. Convert Bounds to Kilometers Relative to Vent ---
x_min = (bounds.left - VENT_X) / 1000.0
x_max = (bounds.right - VENT_X) / 1000.0
y_min = (bounds.bottom - VENT_Y) / 1000.0
y_max = (bounds.top - VENT_Y) / 1000.0

dem_extent = (x_min, x_max, y_min, y_max)

# --- 3. Synchronize Scatter Data Coordinates ---
scatter_x_km = (scatter_x - VENT_X) / 1000.0
scatter_y_km = (scatter_y - VENT_Y) / 1000.0

# --- 4. Generate Hillshade ---
hillshade = es.hillshade(dem_data, azimuth=315, altitude=45)

# --- 5. CALCULATE EXPANDED PLOT BORDERS ---
# To make the DEM look smaller than the window, we extend the outer plot limits.
# This adds a clean, balanced padding buffer (e.g., 5 kilometers) around the image edges.
BORDER_PADDING_KM = 5.0  

plot_xlim = (x_min - BORDER_PADDING_KM, x_max + BORDER_PADDING_KM)
plot_ylim = (y_min - BORDER_PADDING_KM, y_max + BORDER_PADDING_KM)

# --- 6. Plot Everything ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("white")  # Ensures the surrounding border area is clean white

# Draw Gray DEM (Locks exactly onto dem_extent)
ax.imshow(
    hillshade, 
    cmap='gray', 
    extent=dem_extent, 
    origin="upper", 
    alpha=0.8,
    zorder=0
)

# Overlay Viridis Scatter Array
sc = ax.scatter(
    scatter_x_km, 
    scatter_y_km, 
    c=scatter_z, 
    cmap="viridis", 
    s=70, 
    edgecolors='black', 
    linewidths=0.9,
    zorder=2
)

# Mark the Vent Origin
ax.plot(0, 0, marker='^', color='red', markersize=11, label='Vent Origin', zorder=3)

# Apply the Expanded Window Framework
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(*plot_xlim)  # Axis limits are now wider than the DEM extent
ax.set_ylim(*plot_ylim)

# Styling and Grid
ax.set_xlabel("km (relative to vent)")
ax.set_ylabel("km (relative to vent)")
ax.set_title("Viridis Scatter with Framed DEM Background")
ax.grid(color="#cccccc", alpha=0.4, linestyle="--")
ax.legend(loc="upper right")

cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
cbar.set_label('Data Value (z)')

plt.show()