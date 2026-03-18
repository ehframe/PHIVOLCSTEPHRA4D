# Volume Computation in `volumeintegrationoftephra.py`

This document explains, in detail, how `volumeintegrationoftephra.py` computes tephra mass and volume for each trajectory panel.

## 1. Purpose

For each input trajectory CSV (panel), the program:

1. Builds a spatial deposition/load field over a 2D grid.
2. Integrates that field over area to get total deposited mass.
3. Converts mass to volume using a bulk density.

The script now supports two volume methods:

- `grid` (original cell-by-cell integration)
- `weibull` (Weibull fit to radial decay profile, then analytic volume integral)

Outputs are written per panel to:

- `volume_summary.csv`
- `volume_integration_per_panel.xlsx`
- optional PNG maps

## 2. Inputs That Control the Computation

Main arguments affecting volume:

- `--traj-dir`: directory with trajectory files (`*mms-1.csv` by default)
- `--sites-csv`: site coordinate table (used to define map extent)
- `--vent-x`, `--vent-y`: vent location (UTM)
- `--coords-mode`: `relative` or `absolute` plotting/reference coordinates
- `--sigma-km`: Gaussian spread parameter for synthetic load field
- `--nx`, `--ny`: grid resolution in x and y
- `--impact-threshold`: minimum field value retained (below becomes zero)
- `--bulk-density-kgm3`: bulk density used for mass -> volume conversion
- `--volume-method`: `grid` or `weibull`
- `--weibull-min-load-kgm2`: minimum load included in Weibull fit
- `--weibull-bins`: number of radial bins for Weibull profile fitting

## 3. Coordinate Preparation

Site coordinates (`Long`, `Lat`) are read from `sites-csv`.

- If `coords-mode=relative`:
  - `x = (Long - vent_x) / 1000`
  - `y = (Lat - vent_y) / 1000`
- If `coords-mode=absolute`:
  - `x = Long / 1000`
  - `y = Lat / 1000`

So the integration grid coordinates are in kilometers either way.

## 4. Grid Construction

The code sets domain limits from site extents plus fixed margins, then builds:

- `x_vec = linspace(x_min, x_max, nx)`
- `y_vec = linspace(y_min, y_max, ny)`
- `xg, yg = meshgrid(x_vec, y_vec)`

Each `(i,j)` grid cell holds one value of the synthetic load field.

## 5. Synthetic Field Generation (`make_field`)

For each panel trajectory:

1. Read trajectory points `(x0, y0)` and convert to km as above.
2. Initialize `field = 0` over the full grid.
3. For each trajectory point `(px, py)` with weight `w`, add a Gaussian term:

   `field += w * exp(-((xg-px)^2 + (yg-py)^2) / (2*sigma_km^2))`

4. Weights are linearly spaced from `1.0` down to `0.25` along the trajectory.
5. Normalize by the maximum value:

   `field = field / field.max()` (if max > 0)

Important: this makes the field dimensionless in practice (peak = 1), even though downstream labels use `kg/m^2`.

## 6. Thresholding

After building the field for a panel:

- any cell with value `< impact_threshold` is set to `0.0`.

This removes low tails before integration.

## 7. Cell Area Computation

Integration uses constant rectangular cell area from first grid spacing:

- `dx_m = abs(x_vec[1] - x_vec[0]) * 1000`
- `dy_m = abs(y_vec[1] - y_vec[0]) * 1000`
- `cell_area_m2 = dx_m * dy_m`

Because `x_vec` and `y_vec` are in km, multiplying by `1000` converts spacing to meters.

## 8. Mass Integration (`grid` method)

Total mass is computed as:

`total_mass_kg = nansum(field) * cell_area_m2`

Interpretation:

- Sum of per-cell load values over all cells
- multiplied by common cell area

This is equivalent to a Riemann-sum area integral on a regular grid.

## 9. Volume Conversion (`grid` method)

Volume is:

`total_volume_m3 = total_mass_kg / bulk_density_kg_m3` (if density > 0)

Then:

`integrated_volume_km3 = total_volume_m3 / 1e9`

## 10. Weibull Volume Method (`weibull`)

When `--volume-method weibull` is selected, the program computes volume using a Weibull radial-decay fit and analytic integral.

### 10.1 Radial profile preparation

1. Use grid cells where `field > weibull-min-load-kgm2`.
2. Compute radial distance from vent for each selected cell:
   - `r_km = sqrt((x-vent_x)^2 + (y-vent_y)^2)`
3. Convert load to thickness:
   - `t_m = field_kg_m2 / bulk_density_kg_m3`
4. Bin `(r_km, t_m)` into radial bins (`--weibull-bins`) and compute mean thickness per bin.

### 10.2 Weibull model fitted

The implemented form is:

`t(r) = theta * exp(-(r/lambda)^n)`

with:

- `theta` in meters
- `lambda` in kilometers
- `n` dimensionless

The fit is done by grid-searching `n` and `lambda`, and solving `theta` by least squares for each pair.

### 10.3 Analytic volume integral

For this model, the areal integral gives:

`V = (2 * theta * lambda^2 / n) * Gamma(2/n)`

Because `theta` is in meters and `lambda` is in km, this yields `km^3` in the equation above, then converted to `m^3` by multiplying by `1e9`.

Mass is then:

`M = V * bulk_density`

### 10.4 Outputs added for Weibull

Summary rows include:

- `volume_method`
- `weibull_status`
- `weibull_theta_m`
- `weibull_lambda_km`
- `weibull_n`
- `weibull_sse`
- `weibull_profile_bins`

If Weibull fitting fails, the code automatically falls back to `grid` integration for that panel.

## 11. Per-Panel Summary Fields

For each panel, the summary includes:

- `panel`
- `trajectory_file`
- `velocity_m_s` (parsed from filename like `1500mms-1.csv -> 1.5 m/s`)
- `sigma_km`
- `impact_threshold_kg_m2`
- `bulk_density_kg_m3`
- `max_s_obs_kg_m2` (max field after threshold)
- `affected_area_km2` (count of nonzero cells * cell area)
- `integrated_mass_kg`
- `integrated_volume_m3`
- `integrated_volume_km3`

## 12. Ranking Table Logic

The barangay ranking sheet uses `make_field` evaluated at site points (`xs, ys`) instead of grid cells:

- `site_scores = make_field(tx, ty, xs, ys, sigma_km)`
- threshold applied
- top `N` rows sorted descending

This ranking is shape-based and tied to the same normalized synthetic field.

## 13. Assumptions and Limitations

1. The field amplitude is normalized to 1.0, so volume scale is model-relative unless externally calibrated.
2. Grid-based integration assumes uniform spacing and rectangular cells.
3. `sigma_km` strongly controls spread and therefore integrated mass/volume.
4. `impact_threshold` can materially change totals by truncating tails.
5. Bulk density linearly scales final volume (`V ~ 1/rho`).

## 14. Practical Interpretation

Within this script version, computed volume is best interpreted as:

- consistent comparative metric across panels under the same settings,
- not an absolute physical volume unless field amplitude has an external physical calibration.

## 15. Reproducibility Checklist

To reproduce results exactly, keep fixed:

- same trajectory files and order
- same `sites-csv`
- same `coords-mode`, `vent-x`, `vent-y`
- same `sigma-km`, `nx`, `ny`
- same `impact-threshold`
- same `bulk-density-kgm3`

Any change in these inputs will change integrated mass/volume.
