import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter

# Reuse core helpers from volumeintegration to keep behavior aligned.
PROGRAM_ROOT = Path(__file__).resolve().parents[1]
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))
from volumeintegrationoftephra import load_dem_layer, resolve_input_path  # noqa: E402


ERNO = 20254
VOLC = "KANLAON"


def velocity_from_scenario_name(name: str) -> tuple[float | None, int | None, str]:
    """
    Parse scenario folder names like:
      1.300mps_1300mms-1
      1300mms-1
    Returns (velocity_m_s, velocity_mms_1, trajectory_file_name_guess).
    """
    vel_ms = None
    vel_mms = None
    m_ms = re.search(r"([0-9]+(?:\.[0-9]+)?)mps", name)
    if m_ms:
        try:
            vel_ms = float(m_ms.group(1))
        except ValueError:
            vel_ms = None
    m_mms = re.search(r"([0-9]+)mms-1", name)
    if m_mms:
        try:
            vel_mms = int(m_mms.group(1))
        except ValueError:
            vel_mms = None
    if vel_ms is None and vel_mms is not None:
        vel_ms = vel_mms / 1000.0
    traj_guess = f"{vel_mms}mms-1.csv" if vel_mms is not None else ""
    return vel_ms, vel_mms, traj_guess


def parse_ascii_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = [f.readline() for _ in range(6)]
        tok = {}
        for line in header:
            p = line.split()
            if p:
                tok[p[0].upper()] = p[1:]
        ncols = int(float(tok["NCOLS"][0]))
        nrows = int(float(tok["NROWS"][0]))
        xll = float(tok["XLLCORNER"][0])
        yll = float(tok["YLLCORNER"][0])
        cellsize = float(tok["CELLSIZE"][0])
        nodata = float(tok["NODATA_VALUE"][0])
        vals = []
        for line in f:
            s = line.strip()
            if s:
                vals.extend(float(v) for v in s.split())
    arr = np.array(vals, dtype=float).reshape((nrows, ncols))
    arr[arr == nodata] = np.nan
    arr = np.flipud(arr)
    x_m = xll + (np.arange(ncols) + 0.5) * cellsize
    y_m = yll + (np.arange(nrows) + 0.5) * cellsize
    xg_m, yg_m = np.meshgrid(x_m, y_m)
    return xg_m, yg_m, arr


def grid_is_local_meters(xg_m: np.ndarray, yg_m: np.ndarray) -> bool:
    x_abs = float(np.nanmax(np.abs(xg_m)))
    y_abs = float(np.nanmax(np.abs(yg_m)))
    return x_abs < 200000.0 and y_abs < 200000.0


def sample_field_nearest(
    x_sites: np.ndarray, y_sites: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, field: np.ndarray
) -> np.ndarray:
    x1 = x_grid[0, :]
    y1 = y_grid[:, 0]
    ix = np.abs(x_sites[:, None] - x1[None, :]).argmin(axis=1)
    iy = np.abs(y_sites[:, None] - y1[None, :]).argmin(axis=1)
    return field[iy, ix]


def sparse_ticks(vmin: float, vmax: float, n: int = 6) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([])
    if np.isclose(vmin, vmax):
        return np.array([vmin])
    return np.linspace(vmin, vmax, n)


def utm_to_lonlat(easting: float, northing: float, zone: int = 51, hemi: str = "N") -> tuple[float, float]:
    a = 6378137.0
    e_sq = 0.0066943799901413165
    e_prime_sq = e_sq / (1.0 - e_sq)
    k0 = 0.9996

    x = easting - 500000.0
    y = northing
    if hemi.upper().startswith("S"):
        y -= 10000000.0

    m = y / k0
    mu = m / (a * (1 - e_sq / 4 - 3 * e_sq * e_sq / 64 - 5 * e_sq**3 / 256))

    e1 = (1 - math.sqrt(1 - e_sq)) / (1 + math.sqrt(1 - e_sq))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512

    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)
    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)

    c1 = e_prime_sq * cos_fp**2
    t1 = tan_fp**2
    r1 = a * (1 - e_sq) / ((1 - e_sq * sin_fp**2) ** 1.5)
    n1 = a / math.sqrt(1 - e_sq * sin_fp**2)
    d = x / (n1 * k0)

    q1 = n1 * tan_fp / r1
    q2 = d**2 / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e_prime_sq) * d**4 / 24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * e_prime_sq - 3 * c1**2) * d**6 / 720
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * d**3 / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e_prime_sq + 24 * t1**2) * d**5 / 120
    lon = (zone - 1) * 6 - 180 + 3 + math.degrees((q5 - q6 + q7) / cos_fp)
    lat = math.degrees(lat)
    return lon, lat


def parse_measured_thickness_mm(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if not text or text in {"-", "na", "n/a", "none"}:
        return np.nan

    nums = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)]
    if not nums:
        return np.nan

    if ("-" in text or " to " in text) and len(nums) >= 2:
        base = 0.5 * (nums[0] + nums[1])
    else:
        base = nums[0]

    if "cm" in text:
        return base * 10.0
    if re.search(r"(?<!m)m(?!m)", text):
        return base * 1000.0
    if "mm" in text or "millimeter" in text:
        return base
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-scenario DEM map with site colors from scenario S_obs and per-velocity CSV output.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--sites-csv", default=f"{VOLC}/Tephra_Reports_elev.csv")
    parser.add_argument("--scenario-root", default=f"{ERNO}/maps/full maps/shapefiles")
    parser.add_argument("--grid-name", default="tephra_s_obs.asc")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="Maximum number of scenario folders to process. Use 0 or negative for all.",
    )
    parser.add_argument("--dem-path", default="kanlaonDEM/kanlaon_DEM.tif")
    parser.add_argument("--dem-alpha", type=float, default=0.72)
    parser.add_argument("--target-max-s-obs-kgm2", type=float, default=3.0)
    parser.add_argument("--min-site-s-obs-kgm2", type=float, default=1e-5)
    parser.add_argument("--vent-x", type=float, default=514495.0)
    parser.add_argument("--vent-y", type=float, default=1150889.0)
    parser.add_argument("--coords-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--degree-labels", action="store_true", default=True)
    parser.add_argument("--contour", dest="contour", action="store_true", help="Enable elevation contour lines.")
    parser.add_argument("--no-contour", dest="contour", action="store_false", help="Disable elevation contour lines.")
    parser.add_argument("--measured-thickness-col", default="Measured_T")
    parser.add_argument("--out-dir", default=f"{ERNO}/maps/full maps/site_thickness")
    parser.add_argument("--out-csv", default="sites_s_obs_summary.csv")
    parser.add_argument("--per-velocity-csv-prefix", default="sites_s_obs_by_velocity")
    parser.set_defaults(contour=False)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if args.data_root:
        base_dir = Path(args.data_root).resolve()
    elif args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = project_root

    sites_path = resolve_input_path(args.sites_csv, base_dir)
    scenario_root = resolve_input_path(args.scenario_root, base_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(sites_path).copy()
    x_col = "LONG" if "LONG" in sites.columns else ("Long" if "Long" in sites.columns else None)
    y_col = "LAT" if "LAT" in sites.columns else ("Lat" if "Lat" in sites.columns else None)
    z_col = "Z" if "Z" in sites.columns else None
    if x_col is None or y_col is None:
        raise KeyError(f"Expected LONG/LAT (or Long/Lat) columns, got: {list(sites.columns)}")

    sites[x_col] = pd.to_numeric(sites[x_col], errors="coerce")
    sites[y_col] = pd.to_numeric(sites[y_col], errors="coerce")
    sites["Z_NUM"] = pd.to_numeric(sites[z_col], errors="coerce") if z_col else 0.0

    if args.measured_thickness_col in sites.columns:
        sites["measured_thickness_mm"] = sites[args.measured_thickness_col].map(parse_measured_thickness_mm)
    else:
        sites["measured_thickness_mm"] = np.nan

    # Ignore placeholder rows with LONG=LAT=Z=0.
    valid = np.isfinite(sites[x_col]) & np.isfinite(sites[y_col])
    valid &= ~((sites[x_col].abs() < 1e-9) & (sites[y_col].abs() < 1e-9) & (sites["Z_NUM"].abs() < 1e-9))
    sites = sites.loc[valid].copy()
    if sites.empty:
        raise ValueError("No valid rows after filtering (check LONG/LAT/Z).")

    x_m = sites[x_col].to_numpy(dtype=float)
    y_m = sites[y_col].to_numpy(dtype=float)
    z = sites["Z_NUM"].to_numpy(dtype=float)
    if args.coords_mode == "relative":
        xs = (x_m - args.vent_x) / 1000.0
        ys = (y_m - args.vent_y) / 1000.0
        vent_plot_x, vent_plot_y = 0.0, 0.0
        x_label = "km (relative to vent)"
        y_label = "km (relative to vent)"
    else:
        xs, ys = x_m / 1000.0, y_m / 1000.0
        vent_plot_x, vent_plot_y = args.vent_x / 1000.0, args.vent_y / 1000.0
        x_label = "UTM X (km)"
        y_label = "UTM Y (km)"

    # Scenario S_obs palette copied from volumeintegration style.
    top_level = float(max(args.target_max_s_obs_kgm2, 1.0))
    sobs_levels = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, top_level]
    sobs_colors = ["#b3b3b3", "#0072b2", "#00a9e0", "#9ad9e6", "#ece8c9", "#ff9f68", "#e63946"]
    sobs_cmap = ListedColormap(sobs_colors)
    sobs_norm = BoundaryNorm(sobs_levels, sobs_cmap.N)

    dem_layer = None
    try:
        dem_layer = load_dem_layer(resolve_input_path(args.dem_path, base_dir), args.coords_mode, args.vent_x, args.vent_y)
    except Exception:
        dem_layer = None

    scenarios = sorted([d for d in scenario_root.iterdir() if d.is_dir() and (d / args.grid_name).exists()])
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]
    if not scenarios:
        raise FileNotFoundError(f"No scenario folders with {args.grid_name} in {scenario_root}")

    scenario_tables: list[pd.DataFrame] = []
    for sc_dir in scenarios:
        grid_path = sc_dir / args.grid_name
        xg_m, yg_m, field = parse_ascii_grid(grid_path)
        local_grid = grid_is_local_meters(xg_m, yg_m)
        if args.coords_mode == "relative":
            if local_grid:
                xg = xg_m / 1000.0
                yg = yg_m / 1000.0
            else:
                xg = (xg_m - args.vent_x) / 1000.0
                yg = (yg_m - args.vent_y) / 1000.0
        else:
            if local_grid:
                xg = (xg_m + args.vent_x) / 1000.0
                yg = (yg_m + args.vent_y) / 1000.0
            else:
                xg = xg_m / 1000.0
                yg = yg_m / 1000.0

        # Compute S_obs at site locations from this scenario grid.
        sx = xs if args.coords_mode == "relative" else (x_m / 1000.0)
        sy = ys if args.coords_mode == "relative" else (y_m / 1000.0)
        site_sobs = sample_field_nearest(sx, sy, xg, yg, field)
        floor = float(max(args.min_site_s_obs_kgm2, 0.0))
        if floor > 0:
            site_sobs = np.where((site_sobs > 0) & (site_sobs < floor), floor, site_sobs)

        x_min = min(xs.min(), np.nanmin(xg), -7.0) - 0.5
        x_max = max(xs.max(), np.nanmax(xg), 5.0) + 0.5
        y_min = min(ys.min(), np.nanmin(yg), -4.0) - 0.5
        y_max = max(ys.max(), np.nanmax(yg), 5.0) + 0.5

        fig, ax = plt.subplots(figsize=(9.2, 6.8))

        if dem_layer is not None:
            dem, hs, (dx0, dx1, dy0, dy1) = dem_layer
            alpha = float(max(0.0, min(args.dem_alpha, 1.0)))
            ax.imshow(hs, extent=(dx0, dx1, dy0, dy1), origin="upper", cmap="gray", alpha=alpha, zorder=0)
            ax.imshow(dem, extent=(dx0, dx1, dy0, dy1), origin="upper", cmap="gray", alpha=min(0.45, alpha * 0.5), zorder=0.5)

        fm = np.ma.masked_where(~np.isfinite(field) | (field <= 0), field)
        if np.any(np.isfinite(fm)):
            ax.contourf(xg, yg, fm, levels=sobs_levels, cmap=sobs_cmap, norm=sobs_norm, alpha=0.35, zorder=1)

        if args.contour:
            try:
                tri = mtri.Triangulation(xs, ys)
                z_levels = np.linspace(np.nanpercentile(z, 10), np.nanpercentile(z, 95), 9)
                ax.tricontour(tri, z, levels=z_levels, colors="#9a9a9a", linewidths=0.85, zorder=2)
            except Exception:
                pass

        sc = ax.scatter(xs, ys, c=site_sobs, cmap=sobs_cmap, norm=sobs_norm, s=55, edgecolors="black", linewidths=0.5, zorder=3)
        ax.plot([vent_plot_x], [vent_plot_y], marker="o", markersize=4, color="#6a6a6a", zorder=4)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(sparse_ticks(x_min, x_max, 6))
        ax.set_yticks(sparse_ticks(y_min, y_max, 6))
        ax.grid(color="#bdbdbd", alpha=0.35, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#444444")
        if args.degree_labels:

            def _fmt_lon(xk, _pos):
                if args.coords_mode == "relative":
                    e = xk * 1000.0 + args.vent_x
                    n = vent_plot_y * 1000.0 + args.vent_y
                else:
                    e = xk * 1000.0
                    n = vent_plot_y * 1000.0
                lon, _ = utm_to_lonlat(e, n, zone=51, hemi="N")
                return f"{lon:.4f}"

            def _fmt_lat(yk, _pos):
                if args.coords_mode == "relative":
                    e = vent_plot_x * 1000.0 + args.vent_x
                    n = yk * 1000.0 + args.vent_y
                else:
                    e = vent_plot_x * 1000.0
                    n = yk * 1000.0
                _, lat = utm_to_lonlat(e, n, zone=51, hemi="N")
                return f"{lat:.4f}"

            ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
            ax.set_xlabel("Longitude (decimal degrees)", fontsize=13, labelpad=3)
            ax.set_ylabel("Latitude (decimal degrees)", fontsize=13, labelpad=3)
        else:
            ax.set_xlabel(x_label, fontsize=13, labelpad=3)
            ax.set_ylabel(y_label, fontsize=13, labelpad=3)
        ax.set_title(f"S_obs at Sites (kg/m^2) | {sc_dir.name}", fontsize=14, pad=8)

        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.06, pad=0.10, ticks=sobs_levels)
        cbar.ax.set_xticklabels(["0", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e0", f"{top_level:g}"])
        cbar.set_label("S_obs (kg/m^2)", fontsize=12)

        out_png = out_dir / f"sites_measured_thickness_{sc_dir.name}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        print(f"Saved map: {out_png.resolve()}")

        vel_ms, vel_mms, traj_guess = velocity_from_scenario_name(sc_dir.name)
        tdf = sites.copy()
        tdf["scenario"] = sc_dir.name
        tdf["trajectory_file"] = traj_guess
        tdf["velocity_m_s"] = np.nan if vel_ms is None else vel_ms
        tdf["velocity_mms_1"] = np.nan if vel_mms is None else vel_mms
        tdf["S_obs_kg_m2"] = site_sobs
        scenario_tables.append(tdf)

    out_csv = out_dir / args.out_csv
    if scenario_tables:
        all_sites = pd.concat(scenario_tables, ignore_index=True)
        keep_cols = [
            c
            for c in [
                "scenario",
                "trajectory_file",
                "velocity_m_s",
                "velocity_mms_1",
                "Municipali",
                "Barangay",
                x_col,
                y_col,
                "Z_NUM",
                args.measured_thickness_col,
                "measured_thickness_mm",
                "S_obs_kg_m2",
                "SOURCE",
            ]
            if c in all_sites.columns
        ]
        all_sites[keep_cols].to_csv(out_csv, index=False)
        print(f"Saved parsed csv: {out_csv.resolve()}")

        grouped = all_sites.groupby(["velocity_m_s", "velocity_mms_1"], dropna=False, sort=True)
        for (vel_ms, vel_mms), gdf in grouped:
            if np.isfinite(vel_ms):
                vel_tag = f"{float(vel_ms):.3f}mps"
            elif np.isfinite(vel_mms):
                vel_tag = f"{int(vel_mms)}mms-1"
            else:
                vel_tag = "unknown_velocity"
            per_vel_csv = out_dir / f"{args.per_velocity_csv_prefix}_{vel_tag}.csv"
            gdf[keep_cols].to_csv(per_vel_csv, index=False)
            print(f"Saved velocity csv: {per_vel_csv.resolve()} ({len(gdf)} rows)")
    else:
        pd.DataFrame().to_csv(out_csv, index=False)
        print(f"Saved parsed csv: {out_csv.resolve()}")
    print(f"Scenarios processed: {len(scenarios)} | Points plotted: {len(sites)}")


if __name__ == "__main__":
    main()
