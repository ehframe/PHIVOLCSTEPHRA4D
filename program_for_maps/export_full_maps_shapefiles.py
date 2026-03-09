import argparse
import csv
import datetime as dt
import math
import struct
import xml.sax.saxutils as sx
import zipfile
from pathlib import Path

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


def velocity_from_name(path: Path) -> float | None:
    stem = path.stem
    if not stem.endswith("mms-1"):
        return None
    raw = stem.replace("mms-1", "")
    try:
        return float(raw) / 1000.0
    except ValueError:
        return None


def _truncate_field_name(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch == "_")
    if not cleaned:
        cleaned = "field"
    return cleaned[:10]


def _write_dbf(path: Path, fields: list[tuple[str, str, int, int]], rows: list[dict]) -> None:
    nrec = len(rows)
    nfields = len(fields)
    header_len = 32 + 32 * nfields + 1
    rec_len = 1 + sum(f[2] for f in fields)
    today = dt.date.today()

    with path.open("wb") as f:
        f.write(struct.pack("<BBBBIHH20x", 3, today.year - 1900, today.month, today.day, nrec, header_len, rec_len))
        for name, ftype, size, dec in fields:
            name_b = _truncate_field_name(name).encode("ascii", errors="ignore")[:10]
            name_b = name_b + b"\x00" * (11 - len(name_b))
            f.write(struct.pack("<11sc4xBB14x", name_b, ftype.encode("ascii"), size, dec))
        f.write(b"\x0D")

        for row in rows:
            f.write(b" ")
            for name, ftype, size, dec in fields:
                v = row.get(name, "")
                if ftype == "C":
                    s = str(v if v is not None else "")[:size].ljust(size)
                elif ftype == "N":
                    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        s = "".rjust(size)
                    else:
                        if dec > 0:
                            s = f"{float(v):>{size}.{dec}f}"
                        else:
                            s = f"{int(round(float(v))):>{size}d}"
                else:
                    s = "".ljust(size)
                f.write(s.encode("ascii", errors="ignore"))
        f.write(b"\x1A")


def _write_point_shapefile(base: Path, points: list[tuple[float, float, dict]], fields: list[tuple[str, str, int, int]]) -> None:
    shp_path = base.with_suffix(".shp")
    shx_path = base.with_suffix(".shx")
    dbf_path = base.with_suffix(".dbf")

    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    else:
        xmin = ymin = xmax = ymax = 0.0

    shp_records = []
    shx_records = []
    offset_words = 50
    for i, (x, y, attrs) in enumerate(points, start=1):
        content = struct.pack("<idd", 1, float(x), float(y))
        content_len_words = len(content) // 2
        rec_header = struct.pack(">ii", i, content_len_words)
        shp_records.append(rec_header + content)
        shx_records.append(struct.pack(">ii", offset_words, content_len_words))
        offset_words += 4 + content_len_words

    shp_file_len_words = offset_words
    shx_file_len_words = 50 + 4 * len(points)

    with shp_path.open("wb") as shp:
        shp.write(struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, shp_file_len_words))
        shp.write(struct.pack("<ii4d4d", 1000, 1, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))
        for rec in shp_records:
            shp.write(rec)

    with shx_path.open("wb") as shx:
        shx.write(struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, shx_file_len_words))
        shx.write(struct.pack("<ii4d4d", 1000, 1, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))
        for rec in shx_records:
            shx.write(rec)

    _write_dbf(dbf_path, fields, [p[2] for p in points])


def _write_polyline_shapefile(
    base: Path,
    lines: list[tuple[list[tuple[float, float]], dict]],
    fields: list[tuple[str, str, int, int]],
) -> None:
    shp_path = base.with_suffix(".shp")
    shx_path = base.with_suffix(".shx")
    dbf_path = base.with_suffix(".dbf")

    all_x = [x for line, _ in lines for x, _ in line]
    all_y = [y for line, _ in lines for _, y in line]
    if all_x and all_y:
        xmin, ymin, xmax, ymax = min(all_x), min(all_y), max(all_x), max(all_y)
    else:
        xmin = ymin = xmax = ymax = 0.0

    shp_records = []
    shx_records = []
    offset_words = 50
    for i, (line, attrs) in enumerate(lines, start=1):
        if len(line) < 2:
            continue
        xvals = [p[0] for p in line]
        yvals = [p[1] for p in line]
        lxmin, lymin, lxmax, lymax = min(xvals), min(yvals), max(xvals), max(yvals)
        num_parts = 1
        num_points = len(line)
        parts = struct.pack("<i", 0)
        pts = b"".join(struct.pack("<2d", float(x), float(y)) for x, y in line)
        content = struct.pack("<i4d2i", 3, lxmin, lymin, lxmax, lymax, num_parts, num_points) + parts + pts
        content_len_words = len(content) // 2
        rec_header = struct.pack(">ii", i, content_len_words)
        shp_records.append(rec_header + content)
        shx_records.append(struct.pack(">ii", offset_words, content_len_words))
        offset_words += 4 + content_len_words

    shp_file_len_words = offset_words
    shx_file_len_words = 50 + 4 * len(shx_records)

    with shp_path.open("wb") as shp:
        shp.write(struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, shp_file_len_words))
        shp.write(struct.pack("<ii4d4d", 1000, 3, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))
        for rec in shp_records:
            shp.write(rec)

    with shx_path.open("wb") as shx:
        shx.write(struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, shx_file_len_words))
        shx.write(struct.pack("<ii4d4d", 1000, 3, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))
        for rec in shx_records:
            shx.write(rec)

    _write_dbf(dbf_path, fields, [attrs for _, attrs in lines if len(_) >= 2])


def _zip_folder(folder: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(folder))


def _utm_zone_epsg(easting: np.ndarray, northing: np.ndarray) -> int:
    e0 = float(np.nanmean(easting))
    n0 = float(np.nanmean(northing))
    zone = int(math.floor((e0 / 1_000_000.0) * 60.0) + 1)
    zone = max(1, min(60, zone))
    north = n0 >= 0
    return (32600 if north else 32700) + zone


def _to_lonlat_utm(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyproj  # type: ignore
    except Exception:
        return np.array([]), np.array([])
    epsg = _utm_zone_epsg(x, y)
    tr = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def _write_tephra_kml(path: Path, points: list[tuple[float, float, dict]]) -> None:
    if not points:
        return
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    lon, lat = _to_lonlat_utm(x, y)
    if len(lon) != len(points):
        return

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "  <Document>",
        "    <name>tephra_cells</name>",
        '    <Style id="tephraStyle"><IconStyle><scale>0.5</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>',
    ]
    for i, ((_, _, attrs), lo, la) in enumerate(zip(points, lon, lat), start=1):
        panel = attrs.get("panel", "")
        sobs = attrs.get("s_obs", "")
        traj = attrs.get("traj_file", "")
        nm = sx.escape(f"tephra_{i}")
        ds = sx.escape(f"panel={panel}; s_obs={sobs}; traj={traj}")
        lines.extend(
            [
                "    <Placemark>",
                f"      <name>{nm}</name>",
                f"      <description>{ds}</description>",
                "      <styleUrl>#tephraStyle</styleUrl>",
                f"      <Point><coordinates>{float(lo):.8f},{float(la):.8f},0</coordinates></Point>",
                "    </Placemark>",
            ]
        )
    lines.extend(["  </Document>", "</kml>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_esri_ascii(path: Path, field: np.ndarray, x_vec_km: np.ndarray, y_vec_km: np.ndarray) -> None:
    if len(x_vec_km) < 2 or len(y_vec_km) < 2:
        raise ValueError("Grid must have at least 2 cells in each axis.")
    dx_m = float(abs(x_vec_km[1] - x_vec_km[0]) * 1000.0)
    dy_m = float(abs(y_vec_km[1] - y_vec_km[0]) * 1000.0)

    x_use = np.array(x_vec_km, dtype=float)
    y_use = np.array(y_vec_km, dtype=float)
    data_use = np.array(field, dtype=float)

    if not np.isclose(dx_m, dy_m, rtol=1e-6, atol=1e-9):
        # ESRI ASCII supports one cellsize only; resample to a square grid automatically.
        cellsize_km = min(abs(x_use[1] - x_use[0]), abs(y_use[1] - y_use[0]))
        x_new = np.arange(float(np.min(x_use)), float(np.max(x_use)) + 0.5 * cellsize_km, cellsize_km)
        y_new = np.arange(float(np.min(y_use)), float(np.max(y_use)) + 0.5 * cellsize_km, cellsize_km)

        # 1) interpolate along x for each row
        row_interp = np.vstack([np.interp(x_new, x_use, row) for row in data_use])
        # 2) interpolate along y for each x column
        data_use = np.vstack([np.interp(y_new, y_use, row_interp[:, j]) for j in range(len(x_new))]).T

        x_use = x_new
        y_use = y_new
        dx_m = dy_m = float(cellsize_km * 1000.0)

    ncols = int(len(x_use))
    nrows = int(len(y_use))
    cellsize = dx_m
    xllcorner = float(np.min(x_use) * 1000.0)
    yllcorner = float(np.min(y_use) * 1000.0)
    nodata = -9999.0

    data = np.array(data_use, dtype=float)
    data = np.flipud(data)
    data = np.where(np.isfinite(data), data, nodata)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"ncols {ncols}\n")
        f.write(f"nrows {nrows}\n")
        f.write(f"xllcorner {xllcorner:.6f}\n")
        f.write(f"yllcorner {yllcorner:.6f}\n")
        f.write(f"cellsize {cellsize:.6f}\n")
        f.write(f"NODATA_value {nodata}\n")
        np.savetxt(f, data, fmt="%.8g")


def _wkt_from_epsg(epsg: int) -> str:
    try:
        import pyproj  # type: ignore

        return pyproj.CRS.from_epsg(int(epsg)).to_wkt("WKT1_ESRI")
    except Exception:
        if int(epsg) == 32651:
            return (
                'PROJCS["WGS_1984_UTM_Zone_51N",'
                'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
                'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
                'PROJECTION["Transverse_Mercator"],'
                'PARAMETER["False_Easting",500000.0],'
                'PARAMETER["False_Northing",0.0],'
                'PARAMETER["Central_Meridian",123.0],'
                'PARAMETER["Scale_Factor",0.9996],'
                'PARAMETER["Latitude_Of_Origin",0.0],'
                'UNIT["Meter",1.0]]'
            )
        return ""


def _write_prj(path: Path, epsg: int) -> None:
    wkt = _wkt_from_epsg(epsg)
    if wkt:
        path.write_text(wkt, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tephra rasters (.asc only) using panel definitions from volume outputs.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--full-maps-dir", default=f"{erno}/maps/full maps")
    parser.add_argument("--sites-csv", default=f"{volc}/brgycindem_table.csv")
    parser.add_argument("--traj-dir", default=f"{erno}/traj{erno}")
    parser.add_argument("--volume-summary-csv", default=None, help="Defaults to <full-maps-dir>/volume/volume_summary.csv")
    parser.add_argument("--vent-x", type=float, default=514495.0)
    parser.add_argument("--vent-y", type=float, default=1150889.0)
    parser.add_argument("--sigma-km", type=float, default=0.55)
    parser.add_argument("--impact-threshold", type=float, default=1e-6)
    parser.add_argument("--nx", type=int, default=260)
    parser.add_argument("--ny", type=int, default=220)
    parser.add_argument("--target-max-s-obs-kgm2", type=float, default=1.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_dir = Path(args.base_dir).resolve() if args.base_dir else project_root

    full_maps_dir = Path(args.full_maps_dir)
    if not full_maps_dir.is_absolute():
        full_maps_dir = base_dir / full_maps_dir
    contour_dir = full_maps_dir / "contour"
    volume_dir = full_maps_dir / "volume"
    shp_dir = full_maps_dir / "shapefiles"
    shp_dir.mkdir(parents=True, exist_ok=True)

    sites_path = resolve_input_path(args.sites_csv, base_dir)
    traj_dir = resolve_input_path(args.traj_dir, base_dir)
    volume_summary_csv = (
        resolve_input_path(args.volume_summary_csv, base_dir)
        if args.volume_summary_csv
        else volume_dir / "volume_summary.csv"
    )
    contour_rank_xlsx = contour_dir / "most_affected_barangays_per_panel.xlsx"

    sites = pd.read_csv(sites_path).copy()
    if "FID" not in sites.columns:
        sites["FID"] = np.arange(len(sites))
    sites["FID"] = pd.to_numeric(sites["FID"], errors="coerce")
    sites["Long"] = pd.to_numeric(sites["Long"], errors="coerce")
    sites["Lat"] = pd.to_numeric(sites["Lat"], errors="coerce")
    sites["Z"] = pd.to_numeric(sites["Z"], errors="coerce") if "Z" in sites.columns else np.nan
    bcol = next((c for c in ["Barangay", "BRGY", "Brgy", "barangay", "name", "Name"] if c in sites.columns), None)
    sites["barangay_name"] = sites[bcol].astype(str) if bcol else [f"Site_{i+1:03d}" for i in range(len(sites))]

    if not volume_summary_csv.exists():
        raise FileNotFoundError(f"Volume summary not found: {volume_summary_csv}")
    volume_summary = pd.read_csv(volume_summary_csv).copy()
    if "trajectory_file" not in volume_summary.columns:
        raise ValueError("Volume summary must contain a 'trajectory_file' column.")

    # Build the same grid basis used by the volume program.
    xs_rel = (sites["Long"].to_numpy(dtype=float) - args.vent_x) / 1000.0
    ys_rel = (sites["Lat"].to_numpy(dtype=float) - args.vent_y) / 1000.0
    x_min = min(xs_rel.min(), -7.0) - 0.5
    x_max = max(xs_rel.max(), 5.0) + 0.5
    y_min = min(ys_rel.min(), -4.0) - 0.5
    y_max = max(ys_rel.max(), 5.0) + 0.5
    x_vec = np.linspace(x_min, x_max, args.nx)
    y_vec = np.linspace(y_min, y_max, args.ny)
    xg, yg = np.meshgrid(x_vec, y_vec)

    created = 0
    skipped = 0
    volume_summary = volume_summary.sort_values(["panel", "trajectory_file"], na_position="last") if "panel" in volume_summary.columns else volume_summary
    for _, row in volume_summary.iterrows():
        traj_name = str(row.get("trajectory_file", "")).strip()
        if not traj_name:
            skipped += 1
            continue
        tf = traj_dir / traj_name
        if not tf.exists():
            matches = list(traj_dir.rglob(Path(traj_name).name))
            if not matches:
                print(f"Skip (missing trajectory): {traj_name}")
                skipped += 1
                continue
            tf = matches[0]

        df = pd.read_csv(tf)
        if not {"x0", "y0"}.issubset(df.columns):
            print(f"Skip (trajectory missing x0/y0): {tf.name}")
            skipped += 1
            continue

        tx = (pd.to_numeric(df["x0"], errors="coerce").to_numpy() - float(args.vent_x)) / 1000.0
        ty = (pd.to_numeric(df["y0"], errors="coerce").to_numpy() - float(args.vent_y)) / 1000.0
        mask = np.isfinite(tx) & np.isfinite(ty)

        sigma_km = float(row["sigma_km"]) if "sigma_km" in row and pd.notna(row["sigma_km"]) else float(args.sigma_km)
        impact_threshold = (
            float(row["impact_threshold_kg_m2"])
            if "impact_threshold_kg_m2" in row and pd.notna(row["impact_threshold_kg_m2"])
            else float(args.impact_threshold)
        )
        target_max = (
            float(row["max_s_obs_kg_m2"])
            if "max_s_obs_kg_m2" in row and pd.notna(row["max_s_obs_kg_m2"])
            else float(args.target_max_s_obs_kgm2)
        )
        target_max = max(target_max, 0.0)

        field = make_field(tx[mask], ty[mask], xg, yg, sigma_km, target_max)
        field[field < impact_threshold] = 0.0

        vt = float(row["velocity_m_s"]) if "velocity_m_s" in row and pd.notna(row["velocity_m_s"]) else velocity_from_name(tf)
        vel_tag = f"{vt:.3f}mps" if vt is not None and np.isfinite(vt) else "unknown_v"
        panel_dir = shp_dir / f"{vel_tag}_{tf.stem}"
        panel_dir.mkdir(parents=True, exist_ok=True)

        raster_path = panel_dir / "tephra_s_obs.asc"
        _write_esri_ascii(raster_path, field, x_vec, y_vec)
        created += 1
        print(f"Wrote {raster_path}")

    print(f"ASC export complete: {created} created, {skipped} skipped")
    print(f"Output folder: {shp_dir.resolve()}")


if __name__ == "__main__":
    main()

