import argparse
import subprocess
import sys
from pathlib import Path


# change as you like
erno = 20254
volc = "KANLAON"


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run contour + volume programs and place outputs in a single 'full maps' folder."
    )
    parser.add_argument("--data-root", default=None, help="Root folder for resolving all relative paths.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--traj-dir", default=f"{erno}/traj{erno}")
    parser.add_argument("--traj-pattern", default="*mms-1.csv")
    parser.add_argument("--sites-csv", default=f"{volc}/Tephra_Reports_elev.csv")
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
    parser.add_argument("--top-n", type=int, default=20)

    parser.add_argument("--nx", type=int, default=260)
    parser.add_argument("--ny", type=int, default=220)
    parser.add_argument("--bulk-density-kgm3", type=float, default=1000.0)
    parser.add_argument("--impact-threshold", type=float, default=1e-6)
    parser.add_argument("--volume-method", choices=["grid", "weibull"], default="grid")
    parser.add_argument("--weibull-min-load-kgm2", type=float, default=1e-6)
    parser.add_argument("--weibull-bins", type=int, default=160)
    parser.add_argument("--target-max-s-obs-kgm2", type=float, default=3.0)
    parser.add_argument("--three-d-target-max-s-obs-kgm2", type=float, default=1.0)
    parser.add_argument("--three-d-bulk-density-kgm3", type=float, default=1000.0)
    parser.add_argument("--three-d-z-exaggeration", type=float, default=550.0)
    parser.add_argument("--background-mode", choices=["map", "contour", "dem", "none"], default="dem")
    parser.add_argument("--dem-path", default="kanlaonDEM/kanlaon_DEM.tif")
    parser.add_argument("--dem-alpha", type=float, default=0.72)
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=1.0,
        help=">1 zooms in the volume panel outputs.",
    )
    parser.add_argument("--view-shift-x-km", type=float, default=0.0, help="Negative moves volume map view left.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--volume-only", action="store_true", help="Run only the volume step.")
    parser.add_argument("--barangay-only", action="store_true", help="Run only the barangay step.")
    parser.add_argument("--with-3d", action="store_true", help="Also run 3D tephra map generation.")
    parser.add_argument("--out-root", default=f"{erno}/maps/full maps")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    contour_script = script_dir / "weightdistributionoftephra.py"
    volume_script = script_dir / "volumeintegrationoftephra.py"
    three_d_script = script_dir / "threedmapsoftephra.py"
    if not contour_script.exists():
        raise FileNotFoundError(f"Missing script: {contour_script}")
    if not volume_script.exists():
        raise FileNotFoundError(f"Missing script: {volume_script}")
    if args.with_3d and not three_d_script.exists():
        raise FileNotFoundError(f"Missing script: {three_d_script}")

    # Keep all relative paths inside the FEB262026 dataset folder.
    project_root = Path(__file__).resolve().parents[1]
    if args.data_root:
        base_dir = Path(args.data_root).resolve()
    elif args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = project_root

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = base_dir / out_root
    contour_out = out_root / "contour"
    contour_xlsx = contour_out / "most_affected_barangays_per_panel.xlsx"
    volume_out = out_root / "volume"
    volume_xlsx = volume_out / "volume_integration_per_panel.xlsx"
    three_d_out = out_root / "three_d"
    out_root.mkdir(parents=True, exist_ok=True)

    shared = [
        "--data-root",
        str(base_dir),
        "--base-dir",
        str(base_dir),
        "--traj-dir",
        args.traj_dir,
        "--traj-pattern",
        args.traj_pattern,
        "--sites-csv",
        args.sites_csv,
        "--vent-x",
        str(args.vent_x),
        "--vent-y",
        str(args.vent_y),
        "--max-panels",
        str(args.max_panels),
        "--sigma-km",
        str(args.sigma_km),
        "--coords-mode",
        args.coords_mode,
        "--top-n",
        str(args.top_n),
    ]

    contour_cmd = [
        sys.executable,
        str(contour_script),
        *shared,
        "--out-dir",
        str(contour_out),
        "--excel-out",
        str(contour_xlsx),
    ]

    volume_cmd = [
        sys.executable,
        str(volume_script),
        *shared,
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--bulk-density-kgm3",
        str(args.bulk_density_kgm3),
        "--impact-threshold",
        str(args.impact_threshold),
        "--volume-method",
        args.volume_method,
        "--weibull-min-load-kgm2",
        str(args.weibull_min_load_kgm2),
        "--weibull-bins",
        str(args.weibull_bins),
        "--target-max-s-obs-kgm2",
        str(args.target_max_s_obs_kgm2),
        "--background-mode",
        args.background_mode,
        "--dem-path",
        str(args.dem_path),
        "--dem-alpha",
        str(args.dem_alpha),
        "--zoom-factor",
        str(args.zoom_factor),
        "--view-shift-x-km",
        str(args.view_shift_x_km),
        "--out-dir",
        str(volume_out),
        "--excel-out",
        str(volume_xlsx),
    ]
    if args.no_plots:
        volume_cmd.append("--no-plots")
    three_d_cmd = [
        sys.executable,
        str(three_d_script),
        *shared,
        "--target-max-s-obs-kgm2",
        str(args.three_d_target_max_s_obs_kgm2),
        "--bulk-density-kgm3",
        str(args.three_d_bulk_density_kgm3),
        "--z-exaggeration",
        str(args.three_d_z_exaggeration),
        "--out-dir",
        str(three_d_out),
    ]

    if not args.volume_only:
        _run(contour_cmd)
    if not args.barangay_only:
        try:
            _run(volume_cmd)
        except subprocess.CalledProcessError:
            if "--no-plots" in volume_cmd:
                raise
            fallback_cmd = [*volume_cmd, "--no-plots"]
            print("Volume plotting failed; retrying without plots to complete CSV/XLSX outputs.")
            _run(fallback_cmd)
    if args.with_3d:
        _run(three_d_cmd)
    print(f"Saved combined outputs in: {out_root.resolve()}")


if __name__ == "__main__":
    main()
