import argparse
import subprocess
import sys
from pathlib import Path


def find_template_kmz(data_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Template KMZ not found: {explicit}")

    # Prefer template in data root first.
    local = sorted(data_root.glob("*.kmz"))
    for p in local:
        n = p.name.lower()
        if "thickness" in n and not n.startswith("generated_"):
            return p.resolve()
    if local:
        return local[0].resolve()

    # Fallback: search one level up (e.g., D:\\TEPHRA4DCLEAN).
    parent = data_root.parent
    parent_kmzs = sorted(parent.glob("*.kmz"))
    for p in parent_kmzs:
        n = p.name.lower()
        if "tephra4d" in n and "thickness" in n:
            return p.resolve()
    for p in parent_kmzs:
        n = p.name.lower()
        if "thickness" in n and not n.startswith("generated_"):
            return p.resolve()
    if parent_kmzs:
        return parent_kmzs[0].resolve()

    raise FileNotFoundError(
        "No template KMZ found. Provide --template-kmz explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate KMZ files from full-maps shapefile scenario folders."
    )
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root dataset folder (default: FEB262026).",
    )
    parser.add_argument("--event", default="20254", help="Event folder name.")
    parser.add_argument(
        "--template-kmz",
        default=None,
        help="Template KMZ path. If omitted, auto-detected from data root or parent.",
    )
    parser.add_argument(
        "--batch-root",
        default=None,
        help="Scenario root folder containing tephra_s_obs.asc files. "
        "Default: <data-root>/<event>/maps/full maps/shapefiles",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output KMZ directory. Default: <data-root>/<event>/maps/full maps/kmz",
    )
    parser.add_argument("--grid-name", default="tephra_s_obs.asc")
    parser.add_argument("--mode", default="thickness", choices=["auto", "thickness", "arrival"])
    parser.add_argument("--min-value", type=float, default=0.001)
    parser.add_argument(
        "--input-crs",
        default="utm",
        choices=["auto", "lonlat", "utm"],
        help="Use UTM by default so local-meter grids are anchored correctly.",
    )
    parser.add_argument("--utm-zone", type=int, default=51)
    parser.add_argument("--utm-hemi", choices=["N", "S"], default="N")
    parser.add_argument(
        "--keep-usgs-tagging",
        action="store_true",
        help="Keep USGS overlays/branding. By default they are removed.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    event = str(args.event)
    kmzgen = data_root / "kmzgenerator.py"
    if not kmzgen.exists():
        raise FileNotFoundError(f"kmzgenerator.py not found: {kmzgen}")

    template = find_template_kmz(data_root, args.template_kmz)
    batch_root = (
        Path(args.batch_root).resolve()
        if args.batch_root
        else (data_root / event / "maps" / "full maps" / "shapefiles")
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (data_root / event / "maps" / "full maps" / "kmz")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(kmzgen),
        "--template-kmz",
        str(template),
        "--batch-root",
        str(batch_root),
        "--output-dir",
        str(output_dir),
        "--grid-name",
        args.grid_name,
        "--mode",
        args.mode,
        "--min-value",
        str(args.min_value),
        "--input-crs",
        args.input_crs,
    ]
    if args.utm_zone is not None:
        cmd.extend(["--utm-zone", str(args.utm_zone)])
    if args.utm_hemi is not None:
        cmd.extend(["--utm-hemi", args.utm_hemi])
    if not args.keep_usgs_tagging:
        cmd.append("--remove-usgs-tagging")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done. KMZ files are in: {output_dir}")


if __name__ == "__main__":
    main()
