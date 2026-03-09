from __future__ import annotations

import argparse
import datetime as dt
import math
import pathlib
import struct
import sys
from typing import Iterable

from FEB262026.program_for_maps.kmzgenerator import (
    find_batch_root,
    find_default_template,
    latlon_to_utm,
    load_template_kml,
    looks_like_local_meters,
    parse_ascii_grid,
    parse_utm_from_prj,
    utm_to_latlon,
    find_vent_lonlat,
)


THICKNESS_LEVELS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
ARRIVAL_LEVELS = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 24.0, 36.0]

WGS84_GEOGCS_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
)


def level_config(mode: str) -> tuple[list[float], str, str]:
    if mode == "thickness":
        return THICKNESS_LEVELS, "Final Deposit Thickness", "mm"
    return ARRIVAL_LEVELS, "Final Ashfall Arrival Time", "hrs"


def infer_mode(template_kmz: pathlib.Path | None, explicit_mode: str) -> str:
    if explicit_mode != "auto":
        return explicit_mode
    if template_kmz is None:
        return "thickness"
    name = template_kmz.name.lower()
    if "arrival" in name:
        return "arrival"
    return "thickness"


def _center_grid(ncols: int, nrows: int, xll: float, yll: float, dx: float, dy: float) -> tuple[list[float], list[float]]:
    xs = [xll + (c + 0.5) * dx for c in range(ncols)]
    ys = [yll + (nrows - r - 0.5) * dy for r in range(nrows)]
    return xs, ys


def _interp_point(xa: float, ya: float, va: float, xb: float, yb: float, vb: float, level: float) -> tuple[float, float]:
    if vb == va:
        return (xa + xb) * 0.5, (ya + yb) * 0.5
    t = (level - va) / (vb - va)
    return xa + t * (xb - xa), ya + t * (yb - ya)


def _extract_segments(values: list[float], xs: list[float], ys: list[float], ncols: int, nrows: int, level: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def v(r: int, c: int) -> float:
        return values[r * ncols + c]

    for r in range(nrows - 1):
        y_top = ys[r]
        y_bot = ys[r + 1]
        for c in range(ncols - 1):
            x_left = xs[c]
            x_right = xs[c + 1]

            v0 = v(r, c)
            v1 = v(r, c + 1)
            v2 = v(r + 1, c + 1)
            v3 = v(r + 1, c)

            corners = [
                (x_left, y_top, v0),
                (x_right, y_top, v1),
                (x_right, y_bot, v2),
                (x_left, y_bot, v3),
            ]
            edge_nodes = [(0, 1), (1, 2), (2, 3), (3, 0)]
            points: dict[int, tuple[float, float]] = {}

            for ei, (a, b) in enumerate(edge_nodes):
                xa, ya, va = corners[a]
                xb, yb, vb = corners[b]
                if (va >= level) == (vb >= level):
                    continue
                points[ei] = _interp_point(xa, ya, va, xb, yb, vb, level)

            if len(points) == 2:
                p = list(points.values())
                segs.append((p[0], p[1]))
            elif len(points) == 4:
                center = (v0 + v1 + v2 + v3) * 0.25
                if center >= level:
                    pairings = [(0, 1), (2, 3)]
                else:
                    pairings = [(0, 3), (1, 2)]
                for e0, e1 in pairings:
                    segs.append((points[e0], points[e1]))

    return segs


def _connect_segments(segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    if not segments:
        return []
    tol = 1e-8

    def key(p: tuple[float, float]) -> tuple[int, int]:
        return round(p[0] / tol), round(p[1] / tol)

    refs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, (a, b) in enumerate(segments):
        refs.setdefault(key(a), []).append((i, 0))
        refs.setdefault(key(b), []).append((i, 1))

    used = [False] * len(segments)
    lines: list[list[tuple[float, float]]] = []

    def pop_next(endpoint: tuple[float, float], prefer_idx: int | None = None) -> tuple[int, int] | None:
        k = key(endpoint)
        candidates = refs.get(k, [])
        for seg_i, seg_end in candidates:
            if used[seg_i]:
                continue
            if prefer_idx is None or seg_i != prefer_idx:
                return seg_i, seg_end
        return None

    for i, seg in enumerate(segments):
        if used[i]:
            continue
        used[i] = True
        a, b = seg
        line = [a, b]

        while True:
            nxt = pop_next(line[-1])
            if nxt is None:
                break
            seg_i, seg_end = nxt
            used[seg_i] = True
            s0, s1 = segments[seg_i]
            line.append(s0 if seg_end == 1 else s1)

        while True:
            nxt = pop_next(line[0])
            if nxt is None:
                break
            seg_i, seg_end = nxt
            used[seg_i] = True
            s0, s1 = segments[seg_i]
            line.insert(0, s1 if seg_end == 1 else s0)

        if len(line) >= 2:
            lines.append(line)

    return lines


def _write_dbf(path: pathlib.Path, fields: list[tuple[str, str, int, int]], rows: list[dict[str, object]]) -> None:
    nrec = len(rows)
    nfields = len(fields)
    header_len = 32 + 32 * nfields + 1
    rec_len = 1 + sum(f[2] for f in fields)
    today = dt.date.today()

    with path.open("wb") as f:
        f.write(struct.pack("<BBBBIHH20x", 3, today.year - 1900, today.month, today.day, nrec, header_len, rec_len))
        for name, ftype, size, dec in fields:
            bname = name.encode("ascii", errors="ignore")[:10]
            bname = bname + b"\x00" * (11 - len(bname))
            f.write(struct.pack("<11sc4xBB14x", bname, ftype.encode("ascii"), size, dec))
        f.write(b"\x0D")

        for row in rows:
            f.write(b" ")
            for name, ftype, size, dec in fields:
                v = row.get(name, "")
                if ftype == "C":
                    txt = str(v if v is not None else "")[:size].ljust(size)
                else:
                    num = float(v) if v not in ("", None) else math.nan
                    if math.isnan(num) or math.isinf(num):
                        txt = "".rjust(size)
                    elif dec > 0:
                        txt = f"{num:>{size}.{dec}f}"
                    else:
                        txt = f"{int(round(num)):>{size}d}"
                f.write(txt.encode("ascii", errors="ignore"))
        f.write(b"\x1A")


def _flatten(parts: Iterable[list[tuple[float, float]]]) -> tuple[list[int], list[tuple[float, float]]]:
    part_idx: list[int] = []
    pts: list[tuple[float, float]] = []
    for part in parts:
        if len(part) < 2:
            continue
        part_idx.append(len(pts))
        pts.extend(part)
    return part_idx, pts


def _write_polyline_shapefile(
    base: pathlib.Path,
    records: list[tuple[list[list[tuple[float, float]]], dict[str, object]]],
    fields: list[tuple[str, str, int, int]],
) -> None:
    shp_path = pathlib.Path(str(base) + ".shp")
    shx_path = pathlib.Path(str(base) + ".shx")
    dbf_path = pathlib.Path(str(base) + ".dbf")

    all_pts = [pt for parts, _ in records for part in parts for pt in part]
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    else:
        xmin = ymin = xmax = ymax = 0.0

    shp_records: list[bytes] = []
    shx_records: list[bytes] = []
    dbf_rows: list[dict[str, object]] = []
    offset_words = 50
    rec_no = 1

    for parts, attrs in records:
        part_idx, pts = _flatten(parts)
        if len(pts) < 2 or not part_idx:
            continue
        xvals = [p[0] for p in pts]
        yvals = [p[1] for p in pts]
        box = (min(xvals), min(yvals), max(xvals), max(yvals))
        num_parts = len(part_idx)
        num_pts = len(pts)
        parts_bin = struct.pack("<" + "i" * num_parts, *part_idx)
        pts_bin = b"".join(struct.pack("<2d", float(x), float(y)) for x, y in pts)
        content = struct.pack("<i4d2i", 3, box[0], box[1], box[2], box[3], num_parts, num_pts) + parts_bin + pts_bin
        clen_words = len(content) // 2
        shp_records.append(struct.pack(">2i", rec_no, clen_words) + content)
        shx_records.append(struct.pack(">2i", offset_words, clen_words))
        offset_words += 4 + clen_words
        dbf_rows.append(attrs)
        rec_no += 1

    shp_len_words = offset_words
    shx_len_words = 50 + 4 * len(shx_records)

    with shp_path.open("wb") as shp:
        shp.write(struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, shp_len_words))
        shp.write(struct.pack("<ii4d4d", 1000, 3, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))
        for rec in shp_records:
            shp.write(rec)

    with shx_path.open("wb") as shx:
        shx.write(struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, shx_len_words))
        shx.write(struct.pack("<ii4d4d", 1000, 3, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))
        for rec in shx_records:
            shx.write(rec)

    _write_dbf(dbf_path, fields, dbf_rows)


def _write_prj(path: pathlib.Path) -> None:
    path.write_text(WGS84_GEOGCS_PRJ, encoding="utf-8")


def _series_metadata(level_index: int, level_value: float, variable: str, units: str, when: str) -> dict[str, object]:
    return {
        "ORG": "User",
        "VOLC": "Kanlaon",
        "RUN DATE": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "WINDFRMT": "N/A",
        "RUN CLASS": "N/A",
        "E_STIME": "",
        "E_PLMH": "",
        "E_DUR": "",
        "E_VOL": "",
        "URL": "",
        "VAR": variable,
        "VALUE": level_value,
        "UNITS": units,
        "INDEX": level_index,
        "TIME": when,
    }


def generate_shp(
    grid_path: pathlib.Path,
    output_base: pathlib.Path,
    mode: str,
    input_crs: str,
    utm_zone: int | None,
    utm_hemi: str | None,
    template_kmz: pathlib.Path | None,
    timestamp: str,
) -> int:
    ncols, nrows, xll, yll, dx, dy, nodata, values = parse_ascii_grid(grid_path)
    values = [0.0 if v == nodata else v for v in values]

    if input_crs == "auto":
        parsed = parse_utm_from_prj(grid_path.with_suffix(".prj"))
        input_crs = "utm" if parsed else "lonlat"
        if parsed:
            utm_zone, utm_hemi = parsed

    if input_crs == "utm" and (utm_zone is None or utm_hemi is None):
        raise ValueError("UTM conversion selected but zone/hemisphere missing.")

    vent_anchor: tuple[float, float] | None = None
    if input_crs == "utm" and utm_zone is not None and looks_like_local_meters(xll, yll, dx, dy, utm_zone):
        if template_kmz is not None:
            tree, _ = load_template_kml(template_kmz)
            vent = find_vent_lonlat(tree.getroot())
            if vent is not None:
                ve, vn, _, _ = latlon_to_utm(vent[0], vent[1], zone=utm_zone)
                vent_anchor = (ve, vn)

    xs, ys = _center_grid(ncols, nrows, xll, yll, dx, dy)
    if input_crs == "utm":
        lon_xs = [0.0] * len(xs)
        lat_ys = [0.0] * len(ys)
        y_mid = ys[len(ys) // 2]
        x_mid = xs[len(xs) // 2]
        for i, x in enumerate(xs):
            xx = x + (vent_anchor[0] if vent_anchor else 0.0)
            yy = y_mid + (vent_anchor[1] if vent_anchor else 0.0)
            lon_xs[i], _ = utm_to_latlon(xx, yy, utm_zone, utm_hemi)
        for j, y in enumerate(ys):
            xx = x_mid + (vent_anchor[0] if vent_anchor else 0.0)
            yy = y + (vent_anchor[1] if vent_anchor else 0.0)
            _, lat_ys[j] = utm_to_latlon(xx, yy, utm_zone, utm_hemi)
        xs = lon_xs
        ys = lat_ys

    levels, var_name, units = level_config(mode)
    records: list[tuple[list[list[tuple[float, float]]], dict[str, object]]] = []

    for idx, level in enumerate(levels):
        segs = _extract_segments(values, xs, ys, ncols, nrows, level)
        lines = _connect_segments(segs)
        if not lines:
            continue
        records.append((lines, _series_metadata(idx, level, var_name, units, timestamp)))

    fields = [
        ("ORG", "C", 50, 0),
        ("VOLC", "C", 42, 0),
        ("RUN DATE", "C", 20, 0),
        ("WINDFRMT", "C", 5, 0),
        ("RUN CLASS", "C", 20, 0),
        ("E_STIME", "C", 20, 0),
        ("E_PLMH", "C", 20, 0),
        ("E_DUR", "C", 20, 0),
        ("E_VOL", "C", 20, 0),
        ("URL", "C", 80, 0),
        ("VAR", "C", 24, 0),
        ("VALUE", "N", 24, 15),
        ("UNITS", "C", 10, 0),
        ("INDEX", "N", 10, 0),
        ("TIME", "C", 20, 0),
    ]

    output_base.parent.mkdir(parents=True, exist_ok=True)
    _write_polyline_shapefile(output_base, records, fields)
    _write_prj(pathlib.Path(str(output_base) + ".prj"))
    print(f"Wrote {pathlib.Path(str(output_base) + '.shp')} with {len(records)} contour records.")
    return len(records)


def apply_ide_defaults(args: argparse.Namespace, argv: list[str]) -> argparse.Namespace:
    if len(argv) > 1:
        return args

    search_root = args.data_root if args.data_root is not None else pathlib.Path.cwd()
    search_root = pathlib.Path(search_root)
    if args.template_kmz is None:
        args.template_kmz = find_default_template(search_root)

    if args.batch_root is None and args.grid is None:
        detected_batch = find_batch_root(search_root, args.grid_name)
        if detected_batch is not None:
            args.batch_root = detected_batch
            if args.output_dir is None:
                args.output_dir = search_root / "shp_per_scenario"
        else:
            grid_files = sorted(search_root.rglob(args.grid_name))
            if grid_files:
                args.grid = grid_files[0]
                if args.output_base is None:
                    args.output_base = search_root / f"generated_{grid_files[0].stem}"

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contour polyline SHP from ASCII tephra grids.")
    parser.add_argument("--grid", type=pathlib.Path)
    parser.add_argument("--output-base", type=pathlib.Path, help="Output path without extension, e.g. ./depothik")
    parser.add_argument("--batch-root", type=pathlib.Path, help="Root folder containing scenario subfolders.")
    parser.add_argument("--data-root", type=None, help="Root directory to search for template/grid when running without explicit paths.")
    parser.add_argument("--grid-name", default="tephra_s_obs.asc", help="Grid filename expected in each scenario folder.")
    parser.add_argument("--output-dir", type=pathlib.Path, help="Output directory for batch SHP files.")
    parser.add_argument("--template-kmz", type=pathlib.Path, default=None, help="Optional, used to anchor local-meter grids to vent.")
    parser.add_argument("--mode", choices=["auto", "thickness", "arrival"], default="auto")
    parser.add_argument("--input-crs", choices=["auto", "lonlat", "utm"], default="auto")
    parser.add_argument("--utm-zone", type=int, default=None)
    parser.add_argument("--utm-hemi", choices=["N", "S"], default=None)
    parser.add_argument("--time-label", default=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    args = parser.parse_args()
    args = apply_ide_defaults(args, sys.argv)

    mode = infer_mode(args.template_kmz, args.mode)
    if args.batch_root:
        if not args.output_dir:
            raise ValueError("--output-dir is required with --batch-root")
        scenarios = sorted([p for p in args.batch_root.iterdir() if p.is_dir()])
        if not scenarios:
            raise ValueError(f"No scenario folders found in {args.batch_root}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        created = 0
        for scenario in scenarios:
            grid_path = scenario / args.grid_name
            if not grid_path.exists():
                print(f"Skip {scenario.name}: missing {args.grid_name}")
                continue
            output_base = args.output_dir / scenario.name
            generate_shp(
                grid_path=grid_path,
                output_base=output_base,
                mode=mode,
                input_crs=args.input_crs,
                utm_zone=args.utm_zone,
                utm_hemi=args.utm_hemi,
                template_kmz=args.template_kmz,
                timestamp=args.time_label,
            )
            created += 1
        print(f"Batch complete: {created} SHP file(s) created in {args.output_dir}")
        return

    if not args.grid or not args.output_base:
        raise ValueError("Single-file mode requires --grid and --output-base.")
    generate_shp(
        grid_path=args.grid,
        output_base=args.output_base,
        mode=mode,
        input_crs=args.input_crs,
        utm_zone=args.utm_zone,
        utm_hemi=args.utm_hemi,
        template_kmz=args.template_kmz,
        timestamp=args.time_label,
    )


if __name__ == "__main__":
    main()
