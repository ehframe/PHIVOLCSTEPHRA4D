import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Wedge

#change as you like
erno = 20254
volc = "KANLAON"

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

    # If coordinates look like lon/lat degrees, convert to UTM meters (zone 51N for Kanlaon).
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


def draw_half_markers(ax: plt.Axes, x: np.ndarray, y: np.ndarray, r: float = 0.33) -> None:
    for px, py in zip(x, y):
        ax.add_patch(Wedge((px, py), r, 90, 270, facecolor="white", edgecolor="black", linewidth=1.0, zorder=5))
        ax.add_patch(Wedge((px, py), r, -90, 90, facecolor="black", edgecolor="black", linewidth=1.0, zorder=5))


def _clean_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\[\]\:\*\?/\\]", "_", name).strip()
    return cleaned if cleaned else "Sheet"


def _unique_sheet_name(base: str, used: set[str]) -> str:
    # Excel worksheet names are max 31 chars and must be unique.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create contour maps")
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
    parser.add_argument("--target-max-s-obs-kgm2", type=float, default=1.0)
    parser.add_argument("--out-dir", default=f"{erno}/maps/contour")
    parser.add_argument(
        "--excel-out",
        default=f"{erno}/maps/contour/most_affected_barangays_per_panel.xlsx",
        help="Path to Excel file that will contain ranked most-affected barangays per panel.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV summary output. Defaults to same folder/name as --excel-out but .csv.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top N barangays to keep in each panel sheet.")
    parser.add_argument("--coords-mode", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--data-root", default=None, help="Root folder for resolving all relative paths.")
    parser.add_argument("--base-dir", default=None)
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
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(sites_path)
    x_m, y_m, z_m, valid = _extract_site_xy_z(sites)
    sites = sites.loc[valid].reset_index(drop=True)
    x_m = x_m[valid]
    y_m = y_m[valid]
    z_m = z_m[valid]
    barangay_col = next((c for c in ["Barangay", "BRGY", "Brgy", "barangay", "name", "Name"] if c in sites.columns), None)
    barangay_names = (
        sites[barangay_col].astype(str).fillna("").str.strip().replace("", "Unknown")
        if barangay_col
        else pd.Series([f"Site_{i+1:03d}" for i in range(len(sites))], dtype=str)
    )

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

    x_min = min(xs.min(), -7.0) - 0.5
    x_max = max(xs.max(), 5.0) + 0.5
    y_min = min(ys.min(), -4.0) - 0.5
    y_max = max(ys.max(), 5.0) + 0.5

    nx, ny = 260, 220
    x_vec = np.linspace(x_min, x_max, nx)
    y_vec = np.linspace(y_min, y_max, ny)
    xg, yg = np.meshgrid(x_vec, y_vec)

    levels = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    colors = ["#b3b3b3", "#0072b2", "#00a9e0", "#9ad9e6", "#ece8c9", "#f3bb8f", "#ff7d79"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)

    triang = mtri.Triangulation(xs, ys)
    saved_paths: list[Path] = []
    panel_rankings: list[pd.DataFrame] = []
    for i, traj_file in enumerate(traj_files):
        fig, ax = plt.subplots(figsize=(8.0, 6.8))

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

        field = make_field(tx, ty, xg, yg, args.sigma_km)
        field[field < 1e-6] = 0.0
        contourf_handle = ax.contourf(xg, yg, field, levels=levels, cmap=cmap, norm=norm, zorder=1)

        site_scores = make_field(tx, ty, xs, ys, args.sigma_km)
        site_scores[site_scores < 1e-6] = 0.0

        try:
            z_levels = np.linspace(np.nanpercentile(zs, 10), np.nanpercentile(zs, 95), 9)
            ax.tricontour(triang, zs, levels=z_levels, colors="#9a9a9a", linewidths=0.85, zorder=2)
        except Exception:
            pass

        draw_half_markers(ax, xs, ys, r=0.33)
        ax.plot([vent_plot_x], [vent_plot_y], marker="o", markersize=4, color="#6a6a6a", zorder=6)

        vt = velocity_from_name(traj_file)
        title = traj_file.stem if vt is None else f"v$_t$:{vt:.1f}m/s"
        ax.set_title(title, fontsize=16, pad=8)

        ranking_df = pd.DataFrame(
            {
                "barangay": barangay_names.to_numpy(),
                "s_obs_kg_m2": site_scores,
            }
        ).sort_values("s_obs_kg_m2", ascending=False, kind="mergesort")
        ranking_df = ranking_df.head(max(1, args.top_n)).reset_index(drop=True)
        ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
        ranking_df.insert(1, "panel", i + 1)
        ranking_df.insert(2, "trajectory_file", traj_file.name)
        ranking_df.insert(3, "velocity_m_s", np.nan if vt is None else vt)
        ranking_df.insert(4, "velocity_mms_1", np.nan if vt is None else int(round(vt * 1000.0)))
        panel_rankings.append(ranking_df)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(_sparse_ticks(x_min, x_max, 6))
        ax.set_yticks(_sparse_ticks(y_min, y_max, 6))
        ax.grid(color="#bdbdbd", alpha=0.35, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#444444")
        ax.set_xlabel(x_label, fontsize=13, labelpad=3)
        ax.set_ylabel(y_label, fontsize=13, labelpad=3)

        cbar = fig.colorbar(contourf_handle, ax=ax, orientation="horizontal", fraction=0.06, pad=0.10, ticks=levels)
        cbar.ax.set_xticklabels(
            ["0", "10$^{-6}$", "10$^{-5}$", "10$^{-4}$", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "10$^{0}$"])
        cbar.set_label("S$_{obs}$ (kg/m$^2$)", fontsize=12)

        out_name = f"map_reference_style_{i + 1:02d}_{traj_file.stem}.png"
        out_path = out_dir / out_name
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        saved_paths.append(out_path)

    excel_out = Path(args.excel_out)
    if not excel_out.is_absolute():
        excel_out = base_dir / excel_out
    csv_out = Path(args.csv_out) if args.csv_out else excel_out.with_suffix(".csv")
    if not csv_out.is_absolute():
        csv_out = base_dir / csv_out
    excel_out.parent.mkdir(parents=True, exist_ok=True)

    used_sheet_names: set[str] = set()
    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        if panel_rankings:
            summary_df = pd.concat(panel_rankings, ignore_index=True)
            summary_df.to_csv(csv_out, index=False)
            summary_df.to_excel(writer, sheet_name=_unique_sheet_name("Summary", used_sheet_names), index=False)
            for i, ranking_df in enumerate(panel_rankings, start=1):
                panel_base = f"Panel_{i:02d}"
                ranking_df.to_excel(writer, sheet_name=_unique_sheet_name(panel_base, used_sheet_names), index=False)

    print(f"Saved {len(saved_paths)} maps in: {out_dir.resolve()}")
    print(f"Saved ranked barangay sheets in: {excel_out.resolve()}")
    if panel_rankings:
        print(f"Saved ranked barangay CSV summary in: {csv_out.resolve()}")


if __name__ == "__main__":
    main()
