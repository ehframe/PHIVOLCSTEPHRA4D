import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


# change as you like
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


def make_field(
    x_line: np.ndarray,
    y_line: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    sigma_km: float,
    target_max_s_obs: float,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 3D tephra maps from program_for_maps inputs.")
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
    parser.add_argument("--bulk-density-kgm3", type=float, default=1000.0)
    parser.add_argument("--z-exaggeration", type=float, default=550.0)
    parser.add_argument("--coords-mode", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--out-dir", default=f"{erno}/maps/three_d")
    parser.add_argument("--base-dir", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    base_dir = Path(args.base_dir).resolve() if args.base_dir else project_root

    traj_dir = resolve_input_path(args.traj_dir, base_dir)
    traj_files = sorted(traj_dir.glob(args.traj_pattern))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {traj_dir} matching {args.traj_pattern}")
    if args.max_panels > 0:
        traj_files = traj_files[: args.max_panels]

    sites_path = resolve_input_path(args.sites_csv, base_dir)
    sites = pd.read_csv(sites_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.coords_mode == "relative":
        xs = (sites["Long"].to_numpy(dtype=float) - args.vent_x) / 1000.0
        ys = (sites["Lat"].to_numpy(dtype=float) - args.vent_y) / 1000.0
        x_label = "km (relative to vent)"
        y_label = "km (relative to vent)"
        vent_plot_x = 0.0
        vent_plot_y = 0.0
    else:
        xs = sites["Long"].to_numpy(dtype=float) / 1000.0
        ys = sites["Lat"].to_numpy(dtype=float) / 1000.0
        x_label = "UTM X (km)"
        y_label = "UTM Y (km)"
        vent_plot_x = args.vent_x / 1000.0
        vent_plot_y = args.vent_y / 1000.0

    zs = sites["Z"].to_numpy(dtype=float) if "Z" in sites.columns else np.zeros_like(xs)
    triang = mtri.Triangulation(xs, ys)
    vent_z = float(np.nanmedian(zs)) if np.isfinite(zs).any() else 0.0

    summary_rows: list[dict[str, float | str]] = []
    saved_paths: list[Path] = []
    for idx, traj_file in enumerate(traj_files, start=1):
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

        tz = traj["z0"].to_numpy(dtype=float) if "z0" in traj.columns else np.array([])
        if len(tz) == 0 and len(tx) > 0:
            tz = np.full_like(tx, vent_z + 200.0, dtype=float)

        s_obs_sites = make_field(tx, ty, xs, ys, args.sigma_km, args.target_max_s_obs_kgm2)
        s_obs_sites[s_obs_sites < 1e-6] = 0.0

        thickness_m = (
            s_obs_sites / args.bulk_density_kgm3
            if args.bulk_density_kgm3 > 0
            else np.zeros_like(s_obs_sites)
        )
        tephra_z = zs + thickness_m * args.z_exaggeration

        fig = plt.figure(figsize=(10.8, 7.6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_trisurf(
            triang,
            zs,
            cmap="Greys",
            linewidth=0.05,
            antialiased=True,
            alpha=0.52,
            shade=True,
        )
        tephra_surf = ax.plot_trisurf(
            triang,
            tephra_z,
            cmap="inferno",
            linewidth=0.0,
            antialiased=True,
            alpha=0.93,
            shade=True,
        )

        if len(tx) > 1 and len(tz) == len(tx):
            ax.plot(tx, ty, tz, color="#2b6cb0", linewidth=1.2, alpha=0.95, label="Trajectory")

        ax.scatter(
            [vent_plot_x],
            [vent_plot_y],
            [vent_z],
            color="#ffd23f",
            edgecolors="black",
            s=36,
            marker="o",
            label="Vent",
            depthshade=False,
        )

        vt = velocity_from_name(traj_file)
        title = traj_file.stem if vt is None else f"3D tephra map (v_t={vt:.1f} m/s)"
        ax.set_title(title, pad=12)
        ax.set_xlabel(x_label, labelpad=8)
        ax.set_ylabel(y_label, labelpad=8)
        ax.set_zlabel("Elevation / relative deposit (m)", labelpad=8)
        ax.view_init(elev=30, azim=-60)

        cbar = fig.colorbar(tephra_surf, ax=ax, fraction=0.026, pad=0.06)
        cbar.set_label("Relative tephra intensity (elevated by S_obs / density)")

        out_name = f"tephra_3d_{idx:02d}_{traj_file.stem}.png"
        out_path = out_dir / out_name
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        saved_paths.append(out_path)

        summary_rows.append(
            {
                "panel": idx,
                "trajectory_file": traj_file.name,
                "velocity_m_s": np.nan if vt is None else vt,
                "max_s_obs_kg_m2": float(np.nanmax(s_obs_sites)) if len(s_obs_sites) else 0.0,
                "max_thickness_m": float(np.nanmax(thickness_m)) if len(thickness_m) else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "tephra_3d_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved {len(saved_paths)} 3D maps in: {out_dir.resolve()}")
    print(f"Saved summary CSV: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
