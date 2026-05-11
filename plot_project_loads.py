import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MultipleLocator
from scipy.spatial import cKDTree

try:
    from PIL import Image
except Exception:
    Image = None
else:
    Image.MAX_IMAGE_PIXELS = None


PROGRAM_FOR_MAPS = Path(__file__).resolve().parent / "program_for_maps"
if str(PROGRAM_FOR_MAPS) not in sys.path:
    sys.path.insert(0, str(PROGRAM_FOR_MAPS))

from volumeintegrationoftephra import integrate_mass_and_volume, load_dem_layer, resolve_input_path  # noqa: E402


VOLC = "KANLAON"
DEFAULT_VENT_X = 514495.0
DEFAULT_VENT_Y = 1150889.0
MAP_COLORS = [
    "#b3b3b3",
    "#0a7cc2",
    "#14b8e5",
    "#9ad9e6",
    "#ece8c9",
    "#ffd0a0",
    "#ff9d86",
    "#ff6b77",
    "#f04a57",
]
VT_COLORS = [
    "#0d3b66",
    "#145ea8",
    "#1f78b4",
    "#29a0c3",
    "#3fc1b5",
    "#78d19a",
    "#b5df7a",
    "#f1dd63",
    "#f6b14f",
    "#ef7d42",
    "#e85d3f",
    "#b73a3a",
]
PHI_BY_VT = {
    12.535: -2,
    8.694: -1,
    5.693: 0,
    3.5: 1,
    1.962: 2,
    0.864: 3,
    0.267: 4,
    0.069: 5,
    0.017: 6,
    0.004: 7,
}


def phi_label_for_vt(vt: float) -> str:
    phi_items = np.array(list(PHI_BY_VT.items()), dtype=float)
    vt_values = phi_items[:, 0]
    phi_values = phi_items[:, 1]
    idx = int(np.argmin(np.abs(vt_values - float(vt))))
    matched_vt = float(vt_values[idx])
    matched_phi = int(phi_values[idx])
    if np.isclose(float(vt), matched_vt, atol=1e-6):
        return f"phi {matched_phi}"
    return f"phi {matched_phi} (~{vt:g} m/s)"


def prepare_sites(sites: pd.DataFrame) -> pd.DataFrame:
    cleaned = sites.copy()
    cleaned["Lat"] = pd.to_numeric(cleaned["Lat"], errors="coerce")
    cleaned["Long"] = pd.to_numeric(cleaned["Long"], errors="coerce")
    cleaned["Z"] = pd.to_numeric(cleaned.get("Z", 0.0), errors="coerce")
    valid = np.isfinite(cleaned["Lat"]) & np.isfinite(cleaned["Long"])
    valid &= ~(
        (cleaned["Lat"].abs() < 1e-9)
        & (cleaned["Long"].abs() < 1e-9)
        & (cleaned["Z"].fillna(0.0).abs() < 1e-9)
    )
    cleaned = cleaned.loc[valid].reset_index(drop=True)
    if "site" not in cleaned.columns:
        cleaned["site"] = np.arange(len(cleaned), dtype=int)
    else:
        cleaned["site"] = pd.to_numeric(cleaned["site"], errors="coerce").astype("Int64")
    return cleaned


def parse_time_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if str(col).isdigit()]


def aggregate_time_series(frame: pd.DataFrame, time_cols: list[str], value_col: str) -> pd.DataFrame:
    work = frame.copy()
    for col in time_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    work["site"] = pd.to_numeric(work["site"], errors="coerce")
    values = work[time_cols].sum(axis=1, min_count=1)
    aggregated = pd.DataFrame({"site": work["site"], value_col: values})
    return aggregated.groupby("site", as_index=False)[value_col].sum(min_count=1)


def build_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    nx: int,
    ny: int,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if xlim is None:
        pad_x = max(1.0, 0.04 * float(np.nanmax(xs) - np.nanmin(xs)))
        x_min = float(np.nanmin(xs)) - pad_x
        x_max = float(np.nanmax(xs)) + pad_x
    else:
        x_min, x_max = xlim
    if ylim is None:
        pad_y = max(1.0, 0.04 * float(np.nanmax(ys) - np.nanmin(ys)))
        y_min = float(np.nanmin(ys)) - pad_y
        y_max = float(np.nanmax(ys)) + pad_y
    else:
        y_min, y_max = ylim
    return np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny)


def estimate_sigma_km(xs: np.ndarray, ys: np.ndarray) -> float:
    points = np.column_stack([xs, ys])
    if len(points) < 2:
        return 2.5
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(2, len(points)))
    if distances.ndim == 1:
        return 2.5
    neighbor_distances = distances[:, 1]
    finite = neighbor_distances[np.isfinite(neighbor_distances) & (neighbor_distances > 0.0)]
    if finite.size == 0:
        return 2.5
    return float(np.clip(np.median(finite) * 1.35, 1.2, 4.5))


def expand_limits(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    zoom_out_factor: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    factor = float(max(zoom_out_factor, 1.0))
    x_center = 0.5 * (x_limits[0] + x_limits[1])
    y_center = 0.5 * (y_limits[0] + y_limits[1])
    half_width = 0.5 * (x_limits[1] - x_limits[0]) * factor
    half_height = 0.5 * (y_limits[1] - y_limits[0]) * factor
    return (x_center - half_width, x_center + half_width), (y_center - half_height, y_center + half_height)


def build_sigma_field(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_km = estimate_sigma_km(xs, ys)
    field = np.zeros_like(xg, dtype=float)
    support = np.zeros_like(xg, dtype=float)
    for x0, y0, value in zip(xs, ys, values, strict=False):
        if not np.isfinite(value) or value <= 0.0:
            continue
        dist2 = (xg - x0) ** 2 + (yg - y0) ** 2
        blob = np.exp(-dist2 / (2.0 * sigma_km ** 2))
        field += value * blob
        support = np.maximum(support, blob)
    return field, support


def contour_levels(target_max: float, min_visible_level: float) -> list[float]:
    levels = [0.0, float(max(min_visible_level, 1e-12))]
    for value in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 25.0, 50.0]:
        if value > levels[-1]:
            levels.append(value)
    if target_max > levels[-1]:
        levels.append(float(target_max))
    return levels


def draw_dem(ax: plt.Axes, dem_layer, alpha: float) -> None:
    if dem_layer is None:
        return
    dem_array, hillshade, (x_min, x_max, y_min, y_max) = dem_layer
    alpha_val = float(max(0.0, min(alpha, 1.0)))
    ax.imshow(
        hillshade,
        extent=(x_min, x_max, y_min, y_max),
        origin="upper",
        cmap="gray",
        alpha=0.28 + 0.28 * alpha_val,
        zorder=0,
        interpolation="bilinear",
    )
    ax.imshow(
        dem_array,
        extent=(x_min, x_max, y_min, y_max),
        origin="upper",
        cmap="terrain",
        alpha=0.10 + 0.12 * alpha_val,
        zorder=0.5,
        interpolation="bilinear",
    )


def draw_dem_contours(ax: plt.Axes, dem_layer, alpha: float) -> None:
    if dem_layer is None:
        return
    dem_array, _, (x_min, x_max, y_min, y_max) = dem_layer
    ax.contour(
        np.linspace(x_min, x_max, dem_array.shape[1]),
        np.linspace(y_min, y_max, dem_array.shape[0]),
        dem_array,
        levels=12,
        colors="#a0a0a0",
        linewidths=0.65,
        alpha=0.65 * float(max(0.0, min(alpha, 1.0))),
        zorder=1,
    )


def load_dem_layer_fallback(dem_path: Path, vent_x: float, vent_y: float):
    if Image is None:
        return None
    tfw_path = dem_path.with_suffix(".tfw")
    if not tfw_path.exists():
        return None
    transform = [float(line.strip()) for line in tfw_path.read_text().splitlines()[:6]]
    if len(transform) < 6:
        return None
    pixel_x, _, _, pixel_y, x0, y0 = transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = Image.open(dem_path)
    max_dim = max(img.size)
    step = max(1, int(np.ceil(max_dim / 2200)))
    if step > 1:
        img = img.resize((img.size[0] // step, img.size[1] // step), resample=Image.BILINEAR)
    dem_array = np.asarray(img, dtype=np.float32)
    dem_array = np.where(np.isfinite(dem_array), dem_array, np.nan)

    filled = np.nan_to_num(
        dem_array,
        nan=np.nanmedian(dem_array[np.isfinite(dem_array)]) if np.any(np.isfinite(dem_array)) else 0.0,
        copy=False,
    )
    dy, dx = np.gradient(filled)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy) * 0.9)
    aspect = np.arctan2(-dx, dy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(42.0)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    hillshade = np.clip(shaded, 0.0, 1.0)

    full_width = dem_array.shape[1] * pixel_x * step
    full_height = dem_array.shape[0] * abs(pixel_y) * step
    x_min = (x0 - vent_x) / 1000.0
    x_max = (x0 + full_width - vent_x) / 1000.0
    y_max = (y0 - vent_y) / 1000.0
    y_min = (y0 - full_height - vent_y) / 1000.0
    return dem_array, hillshade, (x_min, x_max, y_min, y_max)


def load_total_time_series(load_ts_path: Path) -> pd.DataFrame:
    load_df = pd.read_csv(load_ts_path)
    first_col = str(load_df.columns[0]).strip()
    if first_col in {"", "Unnamed: 0"}:
        load_df = load_df.rename(columns={load_df.columns[0]: "site"})
    if "site" not in load_df.columns:
        raise KeyError(f"Expected a site column in {load_ts_path}")
    load_df["site"] = pd.to_numeric(load_df["site"], errors="coerce")
    time_cols = parse_time_columns(load_df)
    if not time_cols:
        raise ValueError(f"No hourly columns found in {load_ts_path}")
    for col in time_cols:
        load_df[col] = pd.to_numeric(load_df[col], errors="coerce").fillna(0.0)
    return load_df[["site", *time_cols]]


def load_velocity_time_series(load_ts_vt_path: Path) -> tuple[pd.DataFrame, list[str]]:
    load_vt = pd.read_csv(load_ts_vt_path)
    if "site" not in load_vt.columns:
        first_col = str(load_vt.columns[0]).strip()
        if first_col in {"", "Unnamed: 0"}:
            load_vt = load_vt.rename(columns={load_vt.columns[0]: "site"})
    if "site" not in load_vt.columns or "v_t" not in load_vt.columns:
        raise KeyError(f"Expected site and v_t columns in {load_ts_vt_path}")
    load_vt["site"] = pd.to_numeric(load_vt["site"], errors="coerce")
    load_vt["v_t"] = pd.to_numeric(load_vt["v_t"], errors="coerce")
    time_cols = parse_time_columns(load_vt)
    if not time_cols:
        raise ValueError(f"No hourly columns found in {load_ts_vt_path}")
    for col in time_cols:
        load_vt[col] = pd.to_numeric(load_vt[col], errors="coerce").fillna(0.0)
    return load_vt, time_cols


def plot_composite_map(
    sites: pd.DataFrame,
    total_ts: pd.DataFrame,
    out_path: Path,
    dem_layer,
    show_dem: bool,
    vent_x: float,
    vent_y: float,
    nx: int,
    ny: int,
    bulk_density_kgm3: float,
    impact_threshold: float,
    zoom_out_factor: float,
    grid_margin_km: float,
    support_decay: float,
) -> Path:
    aggregated = aggregate_time_series(total_ts, parse_time_columns(total_ts), "total_load")
    merged = sites.merge(aggregated, on="site", how="left")
    merged["total_load"] = pd.to_numeric(merged["total_load"], errors="coerce").fillna(0.0)
    merged["Lat"] = pd.to_numeric(merged["Lat"], errors="coerce")
    merged["Long"] = pd.to_numeric(merged["Long"], errors="coerce")
    merged = merged.dropna(subset=["Lat", "Long"]).sort_values("total_load", ascending=False).reset_index(drop=True)

    xs = (merged["Long"].to_numpy(dtype=float) - vent_x) / 1000.0
    ys = (merged["Lat"].to_numpy(dtype=float) - vent_y) / 1000.0
    values = merged["total_load"].to_numpy(dtype=float)
    sigma_km = estimate_sigma_km(xs, ys)
    grid_margin_km = max(float(grid_margin_km), sigma_km * 4.5)
    x_vec, y_vec = build_grid(
        xs,
        ys,
        nx,
        ny,
        (float(np.nanmin(xs)) - grid_margin_km, float(np.nanmax(xs)) + grid_margin_km),
        (float(np.nanmin(ys)) - grid_margin_km, float(np.nanmax(ys)) + grid_margin_km),
    )
    xg, yg = np.meshgrid(x_vec, y_vec)
    field, support = build_sigma_field(xs, ys, values, xg, yg)
    levels = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]
    cmap = ListedColormap(["#25a7d9", "#6cd7ff", "#efeab7", "#f7b3a0", "#ef6b73", "#cf3d57"])
    norm = BoundaryNorm(levels, cmap.N)
    support_cutoff = np.exp(-float(max(support_decay, 0.1)))
    field_plot = np.ma.masked_where((field < levels[0]) | (support < support_cutoff), field)
    field_volume = np.where(np.isfinite(field), field, 0.0)
    field_volume[(support < support_cutoff) | (field_volume < max(impact_threshold, 0.0))] = 0.0
    total_mass_kg, total_volume_m3 = integrate_mass_and_volume(field_volume, x_vec, y_vec, bulk_density_kgm3)
    valid_mask = np.isfinite(field_plot.filled(np.nan))
    if np.any(valid_mask):
        plume_x = xg[valid_mask]
        plume_y = yg[valid_mask]
        x_pad = max(1.0, 0.08 * float(plume_x.max() - plume_x.min()))
        y_pad = max(1.0, 0.08 * float(plume_y.max() - plume_y.min()))
        x_limits = (float(plume_x.min()) - x_pad, float(plume_x.max()) + x_pad)
        y_limits = (float(plume_y.min()) - y_pad, float(plume_y.max()) + y_pad)
    else:
        x_limits = (float(x_vec.min()), float(x_vec.max()))
        y_limits = (float(y_vec.min()), float(y_vec.max()))

    x_limits, y_limits = expand_limits(x_limits, y_limits, zoom_out_factor)

    if dem_layer is not None:
        _, _, (dem_x_min, dem_x_max, dem_y_min, dem_y_max) = dem_layer
        if x_limits[0] >= x_limits[1] or y_limits[0] >= y_limits[1]:
            x_limits = (dem_x_min, dem_x_max)
            y_limits = (dem_y_min, dem_y_max)

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    ax.set_facecolor("white")
    if show_dem:
        draw_dem(ax, dem_layer, 1.0)
    contour = ax.contourf(
        xg,
        yg,
        field_plot,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
        zorder=2,
        alpha=0.92,
    )
    if show_dem:
        draw_dem_contours(ax, dem_layer, 0.7)
    ax.scatter(
        xs,
        ys,
        s=38,
        facecolors="white",
        edgecolors="#222222",
        linewidths=0.9,
        zorder=4,
    )
    ax.plot([0.0], [0.0], marker="o", markersize=5, color="#4a4a4a", zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.grid(color="#c9c9c9", alpha=0.45, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#333333")
    ax.set_xlabel("km (relative to vent)", fontsize=12, labelpad=2)
    ax.set_ylabel("km (relative to vent)", fontsize=12, labelpad=2)
    ax.set_title(f"Composite Load | All Settling Velocities\nV={total_volume_m3:.3e} m^3", fontsize=16, pad=8)

    cbar = fig.colorbar(contour, ax=ax, orientation="horizontal", fraction=0.06, pad=0.10, ticks=levels)
    cbar.ax.set_xticklabels([f"{level:g}" for level in levels])
    cbar.set_label("load", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_path.with_name(f"{out_path.stem}_summary.csv")
    merged["integrated_mass_kg"] = total_mass_kg
    merged["integrated_volume_m3"] = total_volume_m3
    merged["integrated_volume_km3"] = total_volume_m3 / 1e9 if np.isfinite(total_volume_m3) else np.nan
    merged.to_csv(summary_path, index=False)
    return summary_path


def plot_site_timeseries(
    sites: pd.DataFrame,
    total_ts: pd.DataFrame,
    load_vt: pd.DataFrame,
    vt_time_cols: list[str],
    out_dir: Path,
    site_ids: list[int],
) -> list[Path]:
    outputs: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    total_time_cols = parse_time_columns(total_ts)
    total_lookup = total_ts.set_index("site")
    site_lookup = sites.set_index("site")
    velocity_values = sorted(v for v in load_vt["v_t"].dropna().unique())
    color_map = {vt: VT_COLORS[idx % len(VT_COLORS)] for idx, vt in enumerate(velocity_values)}

    for site_id in site_ids:
        if site_id not in total_lookup.index or site_id not in site_lookup.index:
            continue

        total_series = total_lookup.loc[site_id, total_time_cols]
        total_values = np.asarray(total_series, dtype=float)
        vt_subset = load_vt.loc[load_vt["site"] == site_id].copy()
        if vt_subset.empty:
            continue
        vt_subset = vt_subset.sort_values("v_t").reset_index(drop=True)
        bottom = np.zeros(len(vt_time_cols), dtype=float)
        hours = np.arange(len(vt_time_cols), dtype=int)

        fig, ax = plt.subplots(figsize=(11.0, 5.8))
        for _, row in vt_subset.iterrows():
            vt = float(row["v_t"])
            values = row[vt_time_cols].to_numpy(dtype=float)
            if not np.any(values > 0.0):
                continue
            ax.bar(
                hours,
                values,
                bottom=bottom,
                width=0.92,
                color=color_map[vt],
                edgecolor="none",
                label=phi_label_for_vt(vt),
            )
            bottom += values

        cumulative = np.cumsum(total_values)
        ax2 = ax.twinx()
        ax2.plot(hours, cumulative, color="#111111", linewidth=1.8)

        site_row = site_lookup.loc[site_id]
        barangay = str(site_row.get("Barangay", f"site {site_id}"))
        lgu = str(site_row.get("LGU", ""))
        ax.set_title(f"{barangay} | {lgu}".strip(" |"))
        ax.set_xlabel("Hour")
        ax.set_ylabel("Hourly load")
        ax2.set_ylabel("Cumulative load")
        ax.set_xlim(-0.5, len(hours) - 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(6))
        ax.grid(axis="y", color="#cfcfcf", alpha=0.45, linewidth=0.6)
        ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")

        filename = f"site_{site_id}_{barangay.replace('/', '-').replace(' ', '_')}.png"
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a composite Kanlaon load map and site time-series plots.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--sites-csv", default=f"{VOLC}/NEGROS_AFFECTED.csv")
    parser.add_argument("--load-ts-csv", default="4_load_ts.csv")
    parser.add_argument("--load-ts-vt-csv", default="4_load_ts_vt.csv")
    parser.add_argument("--dem-path", default=f"{VOLC}/kanlaonDEM/kanlaon_DEM.tif")
    parser.add_argument("--out-dir", default="generated_plots")
    parser.add_argument("--map-name", default="composite_load_ts_sum.png")
    parser.add_argument("--vent-x", type=float, default=DEFAULT_VENT_X)
    parser.add_argument("--vent-y", type=float, default=DEFAULT_VENT_Y)
    parser.add_argument("--nx", type=int, default=700)
    parser.add_argument("--ny", type=int, default=600)
    parser.add_argument("--top-sites", type=int, default=6)
    parser.add_argument(
        "--all-sites",
        action="store_true",
        help="Generate site time-series plots for every site instead of only the top-load subset.",
    )
    parser.add_argument("--bulk-density-kgm3", type=float, default=1000.0)
    parser.add_argument("--impact-threshold", type=float, default=0.1)
    parser.add_argument("--zoom-out-factor", type=float, default=1.5, help=">1 expands the plot view outward.")
    parser.add_argument("--grid-margin-km", type=float, default=10.0, help="Minimum extra computation margin around stations.")
    parser.add_argument("--support-decay", type=float, default=10.0, help="Larger values keep more of the outer sigma tail.")
    parser.add_argument("--no-dem", action="store_false", dest="show_dem", help="Disable DEM hillshade and contours on the map.")
    parser.set_defaults(show_dem=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent

    sites_path = resolve_input_path(args.sites_csv, base_dir)
    load_ts_path = resolve_input_path(args.load_ts_csv, base_dir)
    load_ts_vt_path = resolve_input_path(args.load_ts_vt_csv, base_dir)
    dem_path = resolve_input_path(args.dem_path, base_dir)

    sites = prepare_sites(pd.read_csv(sites_path))
    total_ts = load_total_time_series(load_ts_path)
    load_vt, vt_time_cols = load_velocity_time_series(load_ts_vt_path)
    dem_layer = None
    if args.show_dem:
        dem_layer = load_dem_layer_fallback(dem_path, args.vent_x, args.vent_y)
        if dem_layer is None:
            try:
                dem_layer = load_dem_layer(dem_path, "relative", args.vent_x, args.vent_y)
            except Exception:
                dem_layer = None

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    map_path = out_dir / args.map_name
    summary_path = plot_composite_map(
        sites=sites,
        total_ts=total_ts,
        out_path=map_path,
        dem_layer=dem_layer,
        show_dem=args.show_dem,
        vent_x=args.vent_x,
        vent_y=args.vent_y,
        nx=args.nx,
        ny=args.ny,
        bulk_density_kgm3=args.bulk_density_kgm3,
        impact_threshold=args.impact_threshold,
        zoom_out_factor=args.zoom_out_factor,
        grid_margin_km=args.grid_margin_km,
        support_decay=args.support_decay,
    )

    totals = aggregate_time_series(total_ts, parse_time_columns(total_ts), "total_load")
    if args.all_sites:
        top_site_ids = (
            sites["site"]
            .dropna()
            .astype(int)
            .tolist()
        )
    else:
        top_site_ids = (
            totals.sort_values("total_load", ascending=False)["site"]
            .dropna()
            .astype(int)
            .head(max(args.top_sites, 0))
            .tolist()
        )
    site_outputs = plot_site_timeseries(
        sites=sites,
        total_ts=total_ts,
        load_vt=load_vt,
        vt_time_cols=vt_time_cols,
        out_dir=out_dir / "site_timeseries",
        site_ids=top_site_ids,
    )

    print(f"Saved map: {map_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved {len(site_outputs)} site time-series plots to: {out_dir / 'site_timeseries'}")


if __name__ == "__main__":
    main()
