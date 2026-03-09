import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
try:
    from pyproj import Transformer
except Exception:
    Transformer = None
try:
    import rasterio
    _RASTERIO_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    rasterio = None
    _RASTERIO_IMPORT_ERROR = exc

Basemap: Any = None
_BASEMAP_IMPORT_ATTEMPTED = False

# change as you like
erno = 20254
volc = "KANLAON"
DEM_MAX_PLOT_DIM = 2200


def resolve_input_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        in_base = base_dir / candidate
        if in_base.exists():
            return in_base
    matches = list(base_dir.rglob(candidate.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find input file: {raw_path}")


def velocity_from_name(path: Path) -> float | None:
    stem = path.stem
    if not stem.endswith("mms-1"):
        return None
    raw = stem.replace("mms-1", "")
    try:
        return float(raw) / 1000.0
    except ValueError:
        return None


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _latlon_to_utm(lon: float, lat: float, zone: int = 51) -> tuple[float, float]:
    a = 6378137.0
    f = 1 / 298.257223563
    e_sq = f * (2 - f)
    e_prime_sq = e_sq / (1 - e_sq)
    k0 = 0.9996

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    n = a / math.sqrt(1 - e_sq * (math.sin(lat_rad) ** 2))
    t = math.tan(lat_rad) ** 2
    c = e_prime_sq * (math.cos(lat_rad) ** 2)
    aa = math.cos(lat_rad) * (lon_rad - lon0)
    m = a * (
        (1 - e_sq / 4 - 3 * e_sq ** 2 / 64 - 5 * e_sq ** 3 / 256) * lat_rad
        - (3 * e_sq / 8 + 3 * e_sq ** 2 / 32 + 45 * e_sq ** 3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e_sq ** 2 / 256 + 45 * e_sq ** 3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e_sq ** 3 / 3072) * math.sin(6 * lat_rad)
    )
    easting = k0 * n * (
        aa + (1 - t + c) * aa ** 3 / 6 + (5 - 18 * t + t ** 2 + 72 * c - 58 * e_prime_sq) * aa ** 5 / 120
    ) + 500000.0
    northing = k0 * (
        m + n * math.tan(lat_rad) * (aa ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * aa ** 4 / 24 +
                                      (61 - 58 * t + t ** 2 + 600 * c - 330 * e_prime_sq) * aa ** 6 / 720)
    )
    return easting, northing


def _extract_site_xy_z(sites: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_col = _pick_col(sites, ["Long", "LONG", "lon", "Lon", "longitude", "Longitude", "X", "x", "Easting"])
    y_col = _pick_col(sites, ["Lat", "LAT", "lat", "latitude", "Latitude", "Y", "y", "Northing"])
    z_col = _pick_col(sites, ["Z", "z", "ELEV", "Elev", "elev", "Elevation", "elevation"])
    if x_col is None or y_col is None:
        raise KeyError(f"Site CSV must contain coordinate columns. Found: {list(sites.columns)}")

    x = pd.to_numeric(sites[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sites[y_col], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(sites[z_col], errors="coerce").to_numpy(dtype=float) if z_col else np.zeros(len(sites), dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    if np.any(finite):
        x95 = np.nanpercentile(np.abs(x[finite]), 95)
        y95 = np.nanpercentile(np.abs(y[finite]), 95)
        if x95 <= 180 and y95 <= 90:
            xy = np.array([_latlon_to_utm(float(lon), float(lat)) if np.isfinite(lon) and np.isfinite(lat) and lon != 0 and lat != 0
                           else (np.nan, np.nan) for lon, lat in zip(x, y)], dtype=float)
            x = xy[:, 0]
            y = xy[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    # Drop placeholder rows where coordinates/elevation are all zero.
    valid &= ~((np.abs(x) < 1e-9) & (np.abs(y) < 1e-9) & (np.abs(z) < 1e-9))
    return x, y, z, valid


def make_field(
    x_line: np.ndarray,
    y_line: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    sigma_km: float,
    target_max_s_obs: float = 1.0,
) -> np.ndarray:
    field = np.zeros_like(xg, dtype=float)
    if len(x_line) == 0:
        return field
    weights = np.linspace(1.0, 0.25, len(x_line))
    for px, py, w in zip(x_line, y_line, weights):
        d2 = (xg - px) ** 2 + (yg - py) ** 2
        field += w * np.exp(-d2 / (2.0 * sigma_km**2))
    if field.max() > 0:
        field = (field / field.max()) * float(max(target_max_s_obs, 0.0))
    return field


def _clean_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\[\]\:\*\?/\\]", "_", name).strip()
    return cleaned if cleaned else "Sheet"


def _unique_sheet_name(base: str, used: set[str]) -> str:
    candidate = _clean_sheet_name(base)[:31]
    if candidate not in used:
        used.add(candidate)
        return candidate
    i = 2
    while True:
        suffix = f"_{i}"
        trimmed = candidate[: 31 - len(suffix)] + suffix
        if trimmed not in used:
            used.add(trimmed)
            return trimmed
        i += 1


def _sparse_ticks(vmin: float, vmax: float, n: int = 6) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([])
    if np.isclose(vmin, vmax):
        return np.array([vmin])
    return np.linspace(vmin, vmax, n)


def draw_half_markers(ax: plt.Axes, x: np.ndarray, y: np.ndarray, r: float = 0.33) -> None:
    # Wedge patches can crash on some Windows/matplotlib builds; use a stable marker fallback.
    marker_area = max(12.0, (r * 18.0) ** 2)
    ax.scatter(x, y, s=marker_area, facecolors="white", edgecolors="black", linewidths=0.9, zorder=5)


def utm_to_lonlat(easting: np.ndarray, northing: np.ndarray, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    if Transformer is None:
        raise RuntimeError("pyproj is required for map background conversion.")
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return np.asarray(lon), np.asarray(lat)


def build_basemap(ll_lon: float, ll_lat: float, ur_lon: float, ur_lat: float):
    global Basemap, _BASEMAP_IMPORT_ATTEMPTED
    if not _BASEMAP_IMPORT_ATTEMPTED:
        _BASEMAP_IMPORT_ATTEMPTED = True
        try:
            from mpl_toolkits.basemap import Basemap as _Basemap
            Basemap = _Basemap
        except Exception:
            Basemap = None
    if Basemap is None:
        return None
    return Basemap(
        projection="merc",
        llcrnrlon=ll_lon,
        llcrnrlat=ll_lat,
        urcrnrlon=ur_lon,
        urcrnrlat=ur_lat,
        resolution="i",
    )


def load_dem_layer(
    dem_path: Path,
    coords_mode: str,
    vent_x_m: float,
    vent_y_m: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]] | None:
    if rasterio is None:
        import_hint = ""
        if _RASTERIO_IMPORT_ERROR is not None:
            import_hint = f" Import error: {_RASTERIO_IMPORT_ERROR}"
        print(
            "DEM load failed: rasterio is not available in this Python environment "
            f"({sys.executable}). Install rasterio and rerun.{import_hint}"
        )
        return None
    try:
        # Follows the same basis as your reference snippet: read band 1 and mask NoData.
        with rasterio.open(dem_path) as src:
            dem_array = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                dem_array[dem_array == nodata] = np.nan
            dem_array, decimate_step = decimate_for_plot(dem_array, max_dim=DEM_MAX_PLOT_DIM)
            hillshade = compute_hillshade(
                dem_array,
                x_res_m=float(abs(src.res[0])) * decimate_step,
                y_res_m=float(abs(src.res[1])) * decimate_step,
            )

            left_m = float(src.bounds.left)
            right_m = float(src.bounds.right)
            bottom_m = float(src.bounds.bottom)
            top_m = float(src.bounds.top)

            if coords_mode == "relative":
                x_min = (left_m - vent_x_m) / 1000.0
                x_max = (right_m - vent_x_m) / 1000.0
                y_min = (bottom_m - vent_y_m) / 1000.0
                y_max = (top_m - vent_y_m) / 1000.0
            else:
                x_min = left_m / 1000.0
                x_max = right_m / 1000.0
                y_min = bottom_m / 1000.0
                y_max = top_m / 1000.0

            return dem_array, hillshade, (min(x_min, x_max), max(x_min, x_max), min(y_min, y_max), max(y_min, y_max))
    except Exception as exc:
        print(f"DEM load failed: {exc}")
        return None


def decimate_for_plot(array: np.ndarray, max_dim: int = 2200) -> tuple[np.ndarray, int]:
    if array.ndim != 2:
        return array, 1
    h, w = array.shape
    largest = max(h, w)
    if largest <= max_dim:
        return array, 1
    step = int(math.ceil(largest / float(max_dim)))
    return array[::step, ::step], step


def compute_hillshade(
    dem_array: np.ndarray,
    x_res_m: float,
    y_res_m: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    if dem_array.size == 0:
        return np.zeros_like(dem_array, dtype=np.uint8)

    filled = dem_array.astype("float64", copy=True)
    finite = np.isfinite(filled)
    if not np.any(finite):
        return np.zeros_like(dem_array, dtype=np.uint8)
    fill_value = float(np.nanmedian(filled[finite]))
    filled[~finite] = fill_value

    dy, dx = np.gradient(filled, max(y_res_m, 1e-6), max(x_res_m, 1e-6))
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)

    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0.0, 1.0)
    shade[~finite] = 0.0
    return (shade * 255.0).astype(np.uint8)


def draw_dem_background(
    ax: plt.Axes,
    dem_layer: tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]] | None,
    alpha: float,
    cmap: str = "gray",
) -> bool:
    if dem_layer is None:
        return False
    dem_array, hillshade, (x_min, x_max, y_min, y_max) = dem_layer
    alpha_val = float(max(0.0, min(alpha, 1.0)))
    ax.imshow(
        hillshade,
        extent=(x_min, x_max, y_min, y_max),
        origin="upper",
        cmap="gray",
        alpha=alpha_val,
        zorder=0,
        interpolation="bilinear",
    )
    ax.imshow(
        dem_array,
        extent=(x_min, x_max, y_min, y_max),
        origin="upper",
        cmap=cmap,
        alpha=min(0.45, alpha_val * 0.5),
        zorder=1,
        interpolation="bilinear",
    )
    return True


def integrate_mass_and_volume(
    field_kg_m2: np.ndarray,
    x_vec_km: np.ndarray,
    y_vec_km: np.ndarray,
    bulk_density_kg_m3: float,
) -> tuple[float, float]:
    if len(x_vec_km) < 2 or len(y_vec_km) < 2:
        raise ValueError("Need at least 2 grid points in both x and y to integrate.")
    dx_m = float(abs(x_vec_km[1] - x_vec_km[0]) * 1000.0)
    dy_m = float(abs(y_vec_km[1] - y_vec_km[0]) * 1000.0)
    cell_area_m2 = dx_m * dy_m
    total_mass_kg = float(np.nansum(field_kg_m2) * cell_area_m2)
    total_volume_m3 = total_mass_kg / bulk_density_kg_m3 if bulk_density_kg_m3 > 0 else np.nan
    return total_mass_kg, total_volume_m3


def integrate_volume_weibull(
    field_kg_m2: np.ndarray,
    xg_km: np.ndarray,
    yg_km: np.ndarray,
    vent_x_km: float,
    vent_y_km: float,
    bulk_density_kg_m3: float,
    min_load_kg_m2: float,
    n_bins: int = 160,
) -> tuple[float, float, dict]:
    if bulk_density_kg_m3 <= 0:
        return np.nan, np.nan, {"status": "invalid_density"}

    mask = np.isfinite(field_kg_m2) & (field_kg_m2 > min_load_kg_m2)
    if np.count_nonzero(mask) < 20:
        return np.nan, np.nan, {"status": "insufficient_cells"}

    r_km = np.sqrt((xg_km - vent_x_km) ** 2 + (yg_km - vent_y_km) ** 2)[mask]
    t_m = (field_kg_m2[mask] / bulk_density_kg_m3).astype(float)

    good = np.isfinite(r_km) & np.isfinite(t_m) & (r_km >= 0.0) & (t_m > 0.0)
    r_km = r_km[good]
    t_m = t_m[good]
    if len(r_km) < 20:
        return np.nan, np.nan, {"status": "insufficient_valid_points"}

    r_max = float(np.nanmax(r_km))
    if not np.isfinite(r_max) or r_max <= 0.0:
        return np.nan, np.nan, {"status": "invalid_radius"}

    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(r_km, edges) - 1
    valid_idx = (idx >= 0) & (idx < n_bins)
    idx = idx[valid_idx]
    t_use = t_m[valid_idx]
    if len(idx) == 0:
        return np.nan, np.nan, {"status": "no_binned_points"}

    sum_t = np.bincount(idx, weights=t_use, minlength=n_bins)
    count_t = np.bincount(idx, minlength=n_bins).astype(float)
    mean_t = np.divide(sum_t, count_t, out=np.full_like(sum_t, np.nan, dtype=float), where=count_t > 0)

    bin_mask = np.isfinite(mean_t) & (mean_t > 0.0)
    x = centers[bin_mask]
    y = mean_t[bin_mask]
    if len(x) < 12:
        return np.nan, np.nan, {"status": "insufficient_profile_bins"}

    x_for_fit = np.maximum(x, 1e-6)
    y_for_fit = y
    lam_lo = max(np.nanpercentile(x_for_fit, 15), 0.05)
    lam_hi = max(np.nanpercentile(x_for_fit, 98), lam_lo * 1.2)

    n_grid = np.linspace(0.4, 2.4, 100)
    lam_grid = np.geomspace(lam_lo, lam_hi, 120)

    best_sse = np.inf
    best_theta = np.nan
    best_lam = np.nan
    best_n = np.nan

    for n_val in n_grid:
        pow_term = np.power(x_for_fit[:, None] / lam_grid[None, :], n_val)
        g = np.exp(-pow_term)
        denom = np.sum(g * g, axis=0)
        numer = np.sum(y_for_fit[:, None] * g, axis=0)
        theta = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        pred = g * theta[None, :]
        sse = np.sum((pred - y_for_fit[:, None]) ** 2, axis=0)
        j = int(np.argmin(sse))
        if sse[j] < best_sse:
            best_sse = float(sse[j])
            best_theta = float(theta[j])
            best_lam = float(lam_grid[j])
            best_n = float(n_val)

    if not np.isfinite(best_theta) or best_theta <= 0 or not np.isfinite(best_lam) or best_lam <= 0 or not np.isfinite(best_n) or best_n <= 0:
        return np.nan, np.nan, {"status": "fit_failed"}

    volume_km3 = (2.0 * best_theta * (best_lam**2) / best_n) * math.gamma(2.0 / best_n)
    total_volume_m3 = float(volume_km3 * 1_000_000_000.0)
    total_mass_kg = float(total_volume_m3 * bulk_density_kg_m3)

    return total_mass_kg, total_volume_m3, {
        "status": "ok",
        "weibull_theta_m": best_theta,
        "weibull_lambda_km": best_lam,
        "weibull_n": best_n,
        "weibull_sse": best_sse,
        "weibull_profile_bins": int(len(x_for_fit)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate tephra field to mass/volume and export panel sheets.")
    parser.add_argument("--traj-dir", default=f"{erno}/traj{erno}")
    parser.add_argument("--traj-pattern", default="*mms-1.csv")
    parser.add_argument("--sites-csv", default=f"{volc}/brgycindem_table.csv")
    parser.add_argument("--vent-x", type=float, default=514495.0)
    parser.add_argument("--vent-y", type=float, default=1150889.0)
    parser.add_argument(
        "--max-panels",
        type=int,
        default=0,
        help="Maximum number of trajectory panels to process. Use 0 or negative for all.",
    )
    parser.add_argument("--sigma-km", type=float, default=0.55)
    parser.add_argument("--coords-mode", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--nx", type=int, default=260)
    parser.add_argument("--ny", type=int, default=220)
    parser.add_argument("--bulk-density-kgm3", type=float, default=1000.0)
    parser.add_argument("--impact-threshold", type=float, default=1e-6)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-dir", default=f"{erno}/maps/volume")
    parser.add_argument("--excel-out", default=None)
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--data-root", default=None, help="Root folder for resolving all relative paths.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--volume-method", choices=["grid", "weibull"], default="grid")
    parser.add_argument("--weibull-min-load-kgm2", type=float, default=1e-6)
    parser.add_argument("--weibull-bins", type=int, default=160)
    parser.add_argument("--target-max-s-obs-kgm2", type=float, default=1.0)
    parser.add_argument("--background-mode", choices=["map", "contour", "dem", "none"], default="map")
    parser.add_argument("--site-epsg", type=int, default=32651)
    parser.add_argument("--basemap-radius-km", type=float, default=50.0)
    parser.add_argument("--basemap-bounds", default="117,5,127,19")
    parser.add_argument("--no-etopo", action="store_true", default=True)
    parser.add_argument("--dem-path", default=None, help="Path to DEM GeoTIFF for background.")
    parser.add_argument("--dem-alpha", type=float, default=0.72)
    parser.add_argument("--zoom-factor", type=float, default=1.0, help=">1 zooms in, <1 zooms out.")
    parser.add_argument("--view-shift-x-km", type=float, default=0.0, help="Negative moves view left, positive moves right.")
    args = parser.parse_args()

    # Keep all relative paths inside the FEB262026 dataset folder.
    project_root = Path(__file__).resolve().parents[1]
    if args.data_root:
        base_dir = Path(args.data_root).resolve()
    elif args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = project_root

    traj_dir = resolve_input_path(args.traj_dir, base_dir)
    traj_files = sorted(traj_dir.glob(args.traj_pattern))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {traj_dir} matching {args.traj_pattern}")
    if args.max_panels > 0:
        traj_files = traj_files[: args.max_panels]

    sites_path = resolve_input_path(args.sites_csv, base_dir)
    sites = pd.read_csv(sites_path)
    x_m, y_m, z_m, valid = _extract_site_xy_z(sites)
    sites = sites.loc[valid].reset_index(drop=True)
    x_m = x_m[valid]
    y_m = y_m[valid]
    z_m = z_m[valid]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_out = Path(args.excel_out) if args.excel_out else (out_dir / "volume_integration_per_panel.xlsx")
    if not excel_out.is_absolute():
        excel_out = base_dir / excel_out
    excel_out.parent.mkdir(parents=True, exist_ok=True)

    if args.coords_mode == "relative":
        xs = (x_m - args.vent_x) / 1000.0
        ys = (y_m - args.vent_y) / 1000.0
        vent_plot_x = 0.0
        vent_plot_y = 0.0
        x_label = "km (relative to vent)"
        y_label = "km (relative to vent)"
    else:
        xs = x_m / 1000.0
        ys = y_m / 1000.0
        vent_plot_x = args.vent_x / 1000.0
        vent_plot_y = args.vent_y / 1000.0
        x_label = "UTM X (km)"
        y_label = "UTM Y (km)"
    zs = z_m if len(z_m) else np.zeros_like(xs)

    barangay_col = next((c for c in ["Barangay", "BRGY", "Brgy", "barangay", "name", "Name"] if c in sites.columns), None)
    barangay_names = (
        sites[barangay_col].astype(str).fillna("").str.strip().replace("", "Unknown")
        if barangay_col
        else pd.Series([f"Site_{i+1:03d}" for i in range(len(sites))], dtype=str)
    )

    x_min_full = min(xs.min(), -7.0) - 0.5
    x_max_full = max(xs.max(), 5.0) + 0.5
    y_min_full = min(ys.min(), -4.0) - 0.5
    y_max_full = max(ys.max(), 5.0) + 0.5
    x_min_plot, x_max_plot = x_min_full, x_max_full
    y_min_plot, y_max_plot = y_min_full, y_max_full
    if args.zoom_factor > 0 and not np.isclose(args.zoom_factor, 1.0):
        cx = float(vent_plot_x) + float(args.view_shift_x_km)
        cy = float(vent_plot_y)
        half_w = 0.5 * (x_max_full - x_min_full) / float(args.zoom_factor)
        half_h = 0.5 * (y_max_full - y_min_full) / float(args.zoom_factor)
        x_min_plot, x_max_plot = cx - half_w, cx + half_w
        y_min_plot, y_max_plot = cy - half_h, cy + half_h
    elif not np.isclose(args.view_shift_x_km, 0.0):
        x_min_plot += float(args.view_shift_x_km)
        x_max_plot += float(args.view_shift_x_km)

    # Keep full-domain grid for generation/integration; zoom is plotting only.
    x_vec = np.linspace(x_min_full, x_max_full, args.nx)
    y_vec = np.linspace(y_min_full, y_max_full, args.ny)
    xg, yg = np.meshgrid(x_vec, y_vec)
    triang = mtri.Triangulation(xs, ys)

    top_level = float(max(args.target_max_s_obs_kgm2, 1.0))
    levels = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, top_level]
    colors = ["#b3b3b3", "#0072b2", "#00a9e0", "#9ad9e6", "#ece8c9", "#f3bb8f", "#ff9f68", "#e63946"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)
    saved_plot_paths: list[Path] = []
    dem_layer = None
    if args.dem_path and not args.no_plots and args.background_mode == "dem":
        try:
            dem_layer = load_dem_layer(
                dem_path=resolve_input_path(args.dem_path, base_dir),
                coords_mode=args.coords_mode,
                vent_x_m=float(args.vent_x),
                vent_y_m=float(args.vent_y),
            )
        except FileNotFoundError:
            print(f"DEM not found ({args.dem_path}). Continuing without DEM background.")
            dem_layer = None

    summary_rows: list[dict] = []
    panel_rankings: list[pd.DataFrame] = []

    for i, traj_file in enumerate(traj_files, start=1):
        traj = pd.read_csv(traj_file)
        if {"x0", "y0"}.issubset(traj.columns):
            if args.coords_mode == "relative":
                tx = (traj["x0"].to_numpy(dtype=float) - args.vent_x) / 1000.0
                ty = (traj["y0"].to_numpy(dtype=float) - args.vent_y) / 1000.0
            else:
                tx = traj["x0"].to_numpy(dtype=float) / 1000.0
                ty = traj["y0"].to_numpy(dtype=float) / 1000.0
        else:
            tx = np.array([])
            ty = np.array([])

        field = make_field(tx, ty, xg, yg, args.sigma_km, args.target_max_s_obs_kgm2)
        max_before_threshold = float(np.nanmax(field)) if field.size else 0.0
        field[field < args.impact_threshold] = 0.0
        max_after_threshold = float(np.nanmax(field)) if field.size else 0.0

        weibull_meta = {}
        if args.volume_method == "weibull":
            total_mass_kg, total_volume_m3, weibull_meta = integrate_volume_weibull(
                field,
                xg,
                yg,
                vent_plot_x,
                vent_plot_y,
                args.bulk_density_kgm3,
                args.weibull_min_load_kgm2,
                args.weibull_bins,
            )
            if not np.isfinite(total_volume_m3):
                total_mass_kg, total_volume_m3 = integrate_mass_and_volume(
                    field, x_vec, y_vec, args.bulk_density_kgm3
                )
                weibull_meta["status"] = "fallback_to_grid"
        else:
            total_mass_kg, total_volume_m3 = integrate_mass_and_volume(
                field, x_vec, y_vec, args.bulk_density_kgm3
            )
        cell_area_m2 = abs((x_vec[1] - x_vec[0]) * 1000.0) * abs((y_vec[1] - y_vec[0]) * 1000.0)
        affected_cells = int(np.count_nonzero(field > 0.0))
        affected_area_km2 = (affected_cells * cell_area_m2) / 1_000_000.0
        vt = velocity_from_name(traj_file)

        summary_rows.append(
            {
                "panel": i,
                "trajectory_file": traj_file.name,
                "velocity_m_s": np.nan if vt is None else vt,
                "velocity_mms_1": np.nan if vt is None else int(round(vt * 1000.0)),
                "sigma_km": args.sigma_km,
                "impact_threshold_kg_m2": args.impact_threshold,
                "bulk_density_kg_m3": args.bulk_density_kgm3,
                "volume_method": args.volume_method,
                "weibull_status": weibull_meta.get("status", ""),
                "weibull_theta_m": weibull_meta.get("weibull_theta_m", np.nan),
                "weibull_lambda_km": weibull_meta.get("weibull_lambda_km", np.nan),
                "weibull_n": weibull_meta.get("weibull_n", np.nan),
                "weibull_sse": weibull_meta.get("weibull_sse", np.nan),
                "weibull_profile_bins": weibull_meta.get("weibull_profile_bins", np.nan),
                "max_s_obs_kg_m2": float(np.nanmax(field)) if field.size else 0.0,
                "affected_area_km2": affected_area_km2,
                "integrated_mass_kg": total_mass_kg,
                "integrated_volume_m3": total_volume_m3,
                "integrated_volume_km3": total_volume_m3 / 1e9 if np.isfinite(total_volume_m3) else np.nan,
            }
        )
        print(
            f"Panel {i:02d} | target_max={args.target_max_s_obs_kgm2:g} kg/m^2 | "
            f"max_before_threshold={max_before_threshold:.6g} | "
            f"max_after_threshold={max_after_threshold:.6g}"
        )

        site_scores = make_field(tx, ty, xs, ys, args.sigma_km, args.target_max_s_obs_kgm2)
        site_scores[site_scores < args.impact_threshold] = 0.0
        ranking_df = pd.DataFrame(
            {
                "rank": np.arange(1, len(site_scores) + 1),
                "barangay": barangay_names.to_numpy(),
                "s_obs_kg_m2": site_scores,
            }
        ).sort_values("s_obs_kg_m2", ascending=False, kind="mergesort")
        ranking_df = ranking_df.head(max(1, args.top_n)).reset_index(drop=True)
        ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)
        ranking_df.insert(1, "panel", i)
        ranking_df.insert(2, "trajectory_file", traj_file.name)
        ranking_df.insert(3, "velocity_m_s", np.nan if vt is None else vt)
        ranking_df.insert(4, "velocity_mms_1", np.nan if vt is None else int(round(vt * 1000.0)))
        panel_rankings.append(ranking_df)

        if args.save_plots and not args.no_plots:
            fig, ax = plt.subplots(figsize=(8.0, 6.8))
            map_used = False
            contourf_handle = None
            field_plot = np.ma.masked_where(field <= args.impact_threshold, field)

            if args.background_mode == "map":
                try:
                    zoom_for_map = float(args.zoom_factor) if args.zoom_factor > 0 else 1.0
                    radius_m = (float(args.basemap_radius_km) / zoom_for_map) * 1000.0
                    center_x_utm = float(args.vent_x) + float(args.view_shift_x_km) * 1000.0
                    center_y_utm = float(args.vent_y)
                    ll_lon = ll_lat = ur_lon = ur_lat = None
                    try:
                        lon_c, lat_c = utm_to_lonlat(
                            np.array([center_x_utm - radius_m, center_x_utm + radius_m]),
                            np.array([center_y_utm - radius_m, center_y_utm + radius_m]),
                            args.site_epsg,
                        )
                        ll_lon, ur_lon = float(np.min(lon_c)), float(np.max(lon_c))
                        ll_lat, ur_lat = float(np.min(lat_c)), float(np.max(lat_c))
                    except Exception:
                        parts = [p.strip() for p in args.basemap_bounds.split(",") if p.strip()]
                        if len(parts) == 4:
                            ll_lon, ll_lat, ur_lon, ur_lat = (float(p) for p in parts)
                    if ll_lon is not None:
                        bm = build_basemap(ll_lon, ll_lat, ur_lon, ur_lat)
                    else:
                        bm = None
                    if bm is not None:
                        if args.no_etopo:
                            bm.drawmapboundary(fill_color="#e6f2ff")
                            bm.fillcontinents(color="#f2efe9", lake_color="#e6f2ff", zorder=0)
                        else:
                            bm.bluemarble(ax=ax, zorder=0)
                        bm.drawcoastlines(linewidth=0.5, color="#666666", zorder=1)
                        bm.drawcountries(linewidth=0.5, color="#666666", zorder=1)

                        if args.coords_mode == "relative":
                            xg_utm = xg * 1000.0 + args.vent_x
                            yg_utm = yg * 1000.0 + args.vent_y
                            sx_utm = xs * 1000.0 + args.vent_x
                            sy_utm = ys * 1000.0 + args.vent_y
                        else:
                            xg_utm = xg * 1000.0
                            yg_utm = yg * 1000.0
                            sx_utm = xs * 1000.0
                            sy_utm = ys * 1000.0

                        glon, glat = utm_to_lonlat(xg_utm.ravel(), yg_utm.ravel(), args.site_epsg)
                        slon, slat = utm_to_lonlat(sx_utm, sy_utm, args.site_epsg)
                        vlon, vlat = utm_to_lonlat(np.array([args.vent_x]), np.array([args.vent_y]), args.site_epsg)
                        gx, gy = bm(glon.reshape(xg.shape), glat.reshape(yg.shape))
                        sxm, sym = bm(slon, slat)
                        vxm, vym = bm(vlon, vlat)

                        contourf_handle = ax.contourf(gx, gy, field_plot, levels=levels, cmap=cmap, norm=norm, zorder=2)
                        draw_half_markers(ax, sxm, sym, r=330.0)
                        ax.plot(vxm, vym, marker="o", markersize=4, color="#6a6a6a", zorder=6)
                        ax.set_xlabel("Longitude", fontsize=13, labelpad=3)
                        ax.set_ylabel("Latitude", fontsize=13, labelpad=3)
                        map_used = True
                except Exception:
                    map_used = False

            if not map_used:
                if args.background_mode == "dem" or args.dem_path:
                    dem_ok = draw_dem_background(ax=ax, dem_layer=dem_layer, alpha=float(args.dem_alpha))
                    if not dem_ok and args.background_mode == "dem":
                        print("DEM background requested but could not be loaded; proceeding without DEM.")
                contourf_handle = ax.imshow(
                    field_plot,
                    extent=(x_vec.min(), x_vec.max(), y_vec.min(), y_vec.max()),
                    origin="lower",
                    cmap=cmap,
                    norm=norm,
                    zorder=2,
                    interpolation="nearest",
                )
                if args.background_mode == "contour":
                    try:
                        z_levels = np.linspace(np.nanpercentile(zs, 10), np.nanpercentile(zs, 95), 9)
                        ax.tricontour(triang, zs, levels=z_levels, colors="#9a9a9a", linewidths=0.85, zorder=2)
                    except Exception:
                        pass
                draw_half_markers(ax, xs, ys, r=0.33)
                ax.plot([vent_plot_x], [vent_plot_y], marker="o", markersize=4, color="#6a6a6a", zorder=6)
            title = (
                f"Panel {i:02d} | {traj_file.stem}\nV={total_volume_m3:.3e} m^3"
                if vt is None
                else f"Panel {i:02d} | v_t:{vt:.1f} m/s\nV={total_volume_m3:.3e} m^3"
            )
            ax.set_title(title, fontsize=14, pad=8)

            if not map_used:
                ax.set_xlim(x_min_plot, x_max_plot)
                ax.set_ylim(y_min_plot, y_max_plot)
                ax.set_xticks(_sparse_ticks(x_min_plot, x_max_plot, 6))
                ax.set_yticks(_sparse_ticks(y_min_plot, y_max_plot, 6))
                ax.grid(color="#bdbdbd", alpha=0.35, linewidth=0.7)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)
                    spine.set_color("#444444")
                ax.set_xlabel(x_label, fontsize=13, labelpad=3)
                ax.set_ylabel(y_label, fontsize=13, labelpad=3)

            cbar = fig.colorbar(
                contourf_handle, ax=ax, orientation="horizontal", fraction=0.06, pad=0.10, ticks=levels
            )
            cbar_labels = ["0", "10$^{-6}$", "10$^{-5}$", "10$^{-4}$", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "1", f"{top_level:g}"]
            cbar.ax.set_xticklabels(cbar_labels)
            cbar.set_label("S_obs (kg/m^2)", fontsize=12)

            out_name = f"volume_map_{i:02d}_{traj_file.stem}.png"
            out_path = out_dir / out_name
            fig.savefig(out_path, dpi=220)
            plt.close(fig)
            saved_plot_paths.append(out_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "volume_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    used_sheet_names: set[str] = set()
    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=_unique_sheet_name("Summary", used_sheet_names), index=False)
        for i, ranking_df in enumerate(panel_rankings, start=1):
            ranking_df.to_excel(
                writer,
                sheet_name=_unique_sheet_name(f"Panel_{i:02d}", used_sheet_names),
                index=False,
            )

    print(f"Saved volume summary CSV: {summary_csv.resolve()}")
    print(f"Saved volume workbook: {excel_out.resolve()}")
    if args.save_plots and not args.no_plots:
        print(f"Saved {len(saved_plot_paths)} volume maps in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
