import argparse
import datetime as dt
import math
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_project_loads import (
    DEFAULT_VENT_X,
    DEFAULT_VENT_Y,
    VOLC,
    aggregate_time_series,
    build_grid,
    build_sigma_field,
    estimate_sigma_km,
    load_total_time_series,
    parse_time_columns,
    prepare_sites,
    resolve_input_path,
)


UTM_51N_PRJ = (
    'PROJCS["WGS_1984_UTM_Zone_51N",'
    'GEOGCS["GCS_WGS_1984",'
    'DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["False_Easting",500000.0],'
    'PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",123.0],'
    'PARAMETER["Scale_Factor",0.9996],'
    'PARAMETER["Latitude_Of_Origin",0.0],'
    'UNIT["Meter",1.0]]'
)


def parse_levels(raw: str) -> list[float]:
    levels = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(levels) < 2:
        raise ValueError("Need at least two contour levels.")
    unique = sorted(dict.fromkeys(levels))
    if len(unique) < 2:
        raise ValueError("Need at least two distinct contour levels.")
    return unique


def build_composite_field(
    sites: pd.DataFrame,
    total_ts: pd.DataFrame,
    vent_x: float,
    vent_y: float,
    nx: int,
    ny: int,
    grid_margin_km: float,
    impact_threshold: float,
    support_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregated = aggregate_time_series(total_ts, parse_time_columns(total_ts), "total_load")
    merged = sites.merge(aggregated, on="site", how="left")
    merged["total_load"] = pd.to_numeric(merged["total_load"], errors="coerce").fillna(0.0)
    merged["Lat"] = pd.to_numeric(merged["Lat"], errors="coerce")
    merged["Long"] = pd.to_numeric(merged["Long"], errors="coerce")
    merged = merged.dropna(subset=["Lat", "Long"]).reset_index(drop=True)

    xs = (merged["Long"].to_numpy(dtype=float) - vent_x) / 1000.0
    ys = (merged["Lat"].to_numpy(dtype=float) - vent_y) / 1000.0
    values = merged["total_load"].to_numpy(dtype=float)

    sigma_km = estimate_sigma_km(xs, ys)
    margin_km = max(float(grid_margin_km), sigma_km * 4.5)
    x_vec, y_vec = build_grid(
        xs,
        ys,
        nx,
        ny,
        (float(np.nanmin(xs)) - margin_km, float(np.nanmax(xs)) + margin_km),
        (float(np.nanmin(ys)) - margin_km, float(np.nanmax(ys)) + margin_km),
    )
    xg, yg = np.meshgrid(x_vec, y_vec)
    field, support = build_sigma_field(xs, ys, values, xg, yg)
    support_cutoff = np.exp(-float(max(support_decay, 0.1)))
    field = np.where(np.isfinite(field), field, 0.0)
    field[(support < support_cutoff) | (field < max(float(impact_threshold), 0.0))] = 0.0
    return x_vec, y_vec, field


def contour_polygons_utm(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    field: np.ndarray,
    levels: list[float],
    vent_x: float,
    vent_y: float,
) -> list[tuple[list[list[tuple[float, float]]], dict[str, object]]]:
    xg, yg = np.meshgrid(x_vec, y_vec)
    masked = np.ma.masked_where(field <= 0.0, field)
    fig, ax = plt.subplots()
    cs = ax.contourf(xg, yg, masked, levels=levels, antialiased=True)
    polygons: list[tuple[list[list[tuple[float, float]]], dict[str, object]]] = []
    for idx, segs in enumerate(cs.allsegs):
        if idx + 1 >= len(levels):
            break
        lower = float(levels[idx])
        upper = float(levels[idx + 1])
        for seg in segs:
            if len(seg) < 4:
                continue
            ring = [(float(vent_x + x * 1000.0), float(vent_y + y * 1000.0)) for x, y in seg]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            polygons.append(([[*ring]], {"lower": lower, "upper": upper, "label": f"{lower:g}_{upper:g}"}))
    plt.close(fig)
    return polygons


def _truncate_field_name(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch == "_")
    return (cleaned or "field")[:10]


def _write_dbf(path: Path, fields: list[tuple[str, str, int, int]], rows: list[dict[str, object]]) -> None:
    nrec = len(rows)
    nfields = len(fields)
    header_len = 32 + 32 * nfields + 1
    rec_len = 1 + sum(f[2] for f in fields)
    today = dt.date.today()

    with path.open("wb") as handle:
        handle.write(struct.pack("<BBBBIHH20x", 3, today.year - 1900, today.month, today.day, nrec, header_len, rec_len))
        for name, ftype, size, dec in fields:
            name_b = _truncate_field_name(name).encode("ascii", errors="ignore")[:10]
            name_b = name_b + b"\x00" * (11 - len(name_b))
            handle.write(struct.pack("<11sc4xBB14x", name_b, ftype.encode("ascii"), size, dec))
        handle.write(b"\x0D")

        for row in rows:
            handle.write(b" ")
            for name, ftype, size, dec in fields:
                value = row.get(name, "")
                if ftype == "C":
                    text = str(value if value is not None else "")[:size].ljust(size)
                elif ftype == "N":
                    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                        text = "".rjust(size)
                    elif dec > 0:
                        text = f"{float(value):>{size}.{dec}f}"
                    else:
                        text = f"{int(round(float(value))):>{size}d}"
                else:
                    text = "".ljust(size)
                handle.write(text.encode("ascii", errors="ignore"))
        handle.write(b"\x1A")


def _write_polygon_shapefile(
    base: Path,
    polygons: list[tuple[list[list[tuple[float, float]]], dict[str, object]]],
    fields: list[tuple[str, str, int, int]],
) -> None:
    shp_path = base.with_suffix(".shp")
    shx_path = base.with_suffix(".shx")
    dbf_path = base.with_suffix(".dbf")

    all_x = [x for parts, _ in polygons for ring in parts for x, _ in ring]
    all_y = [y for parts, _ in polygons for ring in parts for _, y in ring]
    if all_x and all_y:
        xmin, ymin, xmax, ymax = min(all_x), min(all_y), max(all_x), max(all_y)
    else:
        xmin = ymin = xmax = ymax = 0.0

    shp_records: list[bytes] = []
    shx_records: list[bytes] = []
    dbf_rows: list[dict[str, object]] = []
    offset_words = 50
    rec_no = 1

    for parts, attrs in polygons:
        valid_parts = [ring for ring in parts if len(ring) >= 4]
        if not valid_parts:
            continue
        xvals = [x for ring in valid_parts for x, _ in ring]
        yvals = [y for ring in valid_parts for _, y in ring]
        box = (min(xvals), min(yvals), max(xvals), max(yvals))
        num_parts = len(valid_parts)
        num_points = sum(len(ring) for ring in valid_parts)
        part_offsets: list[int] = []
        cursor = 0
        for ring in valid_parts:
            part_offsets.append(cursor)
            cursor += len(ring)
        parts_blob = b"".join(struct.pack("<i", int(offset)) for offset in part_offsets)
        pts_blob = b"".join(struct.pack("<2d", float(x), float(y)) for ring in valid_parts for x, y in ring)
        content = struct.pack("<i4d2i", 5, box[0], box[1], box[2], box[3], num_parts, num_points) + parts_blob + pts_blob
        content_len_words = len(content) // 2
        shp_records.append(struct.pack(">ii", rec_no, content_len_words) + content)
        shx_records.append(struct.pack(">ii", offset_words, content_len_words))
        dbf_rows.append(attrs)
        offset_words += 4 + content_len_words
        rec_no += 1

    shp_file_len_words = offset_words
    shx_file_len_words = 50 + 4 * len(shx_records)

    with shp_path.open("wb") as shp:
        shp.write(struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, shp_file_len_words))
        shp.write(struct.pack("<ii4d4d", 1000, 5, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))
        for record in shp_records:
            shp.write(record)

    with shx_path.open("wb") as shx:
        shx.write(struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, shx_file_len_words))
        shx.write(struct.pack("<ii4d4d", 1000, 5, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))
        for record in shx_records:
            shx.write(record)

    _write_dbf(dbf_path, fields, dbf_rows)


def _write_prj(path: Path) -> None:
    path.write_text(UTM_51N_PRJ, encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export composite load contours from plot_project_loads inputs as a polygon shapefile.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--sites-csv", default=f"{VOLC}/NEGROS_AFFECTED.csv")
    parser.add_argument("--load-ts-csv", default="4_load_ts.csv")
    parser.add_argument("--out-dir", default="generated_shapefiles")
    parser.add_argument("--out-name", default="composite_load_contours")
    parser.add_argument("--vent-x", type=float, default=DEFAULT_VENT_X)
    parser.add_argument("--vent-y", type=float, default=DEFAULT_VENT_Y)
    parser.add_argument("--nx", type=int, default=700)
    parser.add_argument("--ny", type=int, default=600)
    parser.add_argument("--grid-margin-km", type=float, default=10.0)
    parser.add_argument("--impact-threshold", type=float, default=0.1)
    parser.add_argument("--support-decay", type=float, default=10.0)
    parser.add_argument("--levels", default="0.1,0.5,1,5,10,25,50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent

    sites_path = resolve_input_path(args.sites_csv, base_dir)
    load_ts_path = resolve_input_path(args.load_ts_csv, base_dir)

    sites = prepare_sites(pd.read_csv(sites_path))
    total_ts = load_total_time_series(load_ts_path)
    levels = parse_levels(args.levels)

    x_vec, y_vec, field = build_composite_field(
        sites=sites,
        total_ts=total_ts,
        vent_x=float(args.vent_x),
        vent_y=float(args.vent_y),
        nx=int(args.nx),
        ny=int(args.ny),
        grid_margin_km=float(args.grid_margin_km),
        impact_threshold=float(args.impact_threshold),
        support_decay=float(args.support_decay),
    )

    polygons = contour_polygons_utm(
        x_vec=x_vec,
        y_vec=y_vec,
        field=field,
        levels=levels,
        vent_x=float(args.vent_x),
        vent_y=float(args.vent_y),
    )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / args.out_name

    fields = [
        ("lower", "N", 16, 8),
        ("upper", "N", 16, 8),
        ("label", "C", 24, 0),
    ]
    _write_polygon_shapefile(out_base, polygons, fields)
    _write_prj(out_base.with_suffix(".prj"))

    print(f"Saved shapefile: {out_base.with_suffix('.shp')}")
    print(f"Polygon records: {len(polygons)}")


if __name__ == "__main__":
    main()
