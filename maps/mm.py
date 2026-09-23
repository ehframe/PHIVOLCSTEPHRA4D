import pandas as pd
from pathlib import Path


main_dir = Path(__file__).resolve().parents[1]
df = pd.read_csv(main_dir / '20260709_070000_files_v1'/ 'mapoutputs' / 'totalload.csv')

# Set ash bulk density (kg/m^3)
rho = 1200  # adjust to 1000, 1200, or 1500 as needed

# Calculate thickness in mm
df['ash_mm'] = (df['total_load'] / rho) * 1000

df.to_csv( main_dir / '20260709_070000_files_v1'/ 'mapoutputs' / 'totalload_ash_mm.csv', index=False)