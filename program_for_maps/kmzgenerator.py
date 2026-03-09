from __future__ import annotations

import argparse
import math
import pathlib
import sys
import zipfile
import xml.etree.ElementTree as et

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
NS = {"k": KML_NS}

et.register_namespace("", KML_NS)
et.register_namespace("gx", GX_NS)


THICKNESS_BINS = [
    (0.01, 0.03, ".01-.03mm"),
    (0.03, 0.10, ".03-0.1mm"),
    (0.10, 0.30, "0.1-0.3mm"),
    (0.30, 1.00, "0.3-1.0mm"),
    (1.00, 3.00, "1.0-3.0mm"),
    (3.00, 10.0, "3.0-10.mm"),
    (10.0, 30.0, "10.-30.mm"),
    (30.0, 100.0, "30.-100mm"),
    (100.0, 300.0, "100-300mm"),
    (300.0, 1000.0, "300-1k_mm"),
    (1000.0, float("inf"), ">>>>1k_mm"),
]

ARRIVAL_BINS = [
    (0.0, 3.0, "00-03_hrs"),
    (3.0, 6.0, "03-06_hrs"),
    (6.0, 9.0, "06-09_hrs"),
    (9.0, 12.0, "09-12_hrs"),
    (12.0, 15.0, "12-15_hrs"),
    (15.0, 18.0, "15-18_hrs"),
    (18.0, 24.0, "18-24_hrs"),
    (24.0, 36.0, "24-36_hrs"),
    (36.0, float("inf"), ">>>36_hrs"),
]


def choose_style(value: float, mode: str) -> str:
    bins = THICKNESS_BINS if mode == "thickness" else ARRIVAL_BINS
    for lower, upper, style_id in bins:
        if lower <= value < upper:
            return style_id
    return bins[-1][2]


def infer_mode(template_kmz: pathlib.Path, explicit_mode: str) -> str:
    if explicit_mode != "auto":
        return explicit_mode
    name = template_kmz.name.lower()
    if "thickness" in name:
        return "thickness"
    if "arrival" in name:
        return "arrival"
    raise ValueError("Could not infer mode. Use --mode thickness or --mode arrival.")


def load_template_kml(kmz_path: pathlib.Path) -> tuple[et.ElementTree, str]:
    with zipfile.ZipFile(kmz_path, "r") as archive:
        kml_names = [n for n in archive.namelist() if n.lower().endswith(".kml")]
        if not kml_names:
            raise ValueError(f"No KML found inside {kmz_path}")
        kml_name = kml_names[0]
        raw_kml = archive.read(kml_name).decode("utf-8", errors="replace")

    if "xmlns:gx=" not in raw_kml:
        raw_kml = raw_kml.replace(
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">',
            1,
        )

    tree = et.ElementTree(et.fromstring(raw_kml))
    return tree, kml_name


def find_final_folder(root: et.Element) -> et.Element:
    for folder in root.findall(".//k:Folder", NS):
        name_node = folder.find("k:name", NS)
        if name_node is not None and (name_node.text or "").strip() == "Final":
            return folder
    raise ValueError("Template KML does not contain a Folder named 'Final'.")


def parse_ascii_grid(grid_path: pathlib.Path) -> tuple[int, int, float, float, float, float, float, list[float]]:
    with grid_path.open("r", encoding="utf-8", errors="replace") as handle:
        header = [handle.readline() for _ in range(6)]
        header_tokens = {}
        for line in header:
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            values = [float(v) for v in parts[1:]]
            header_tokens[key] = values

        ncols = int(header_tokens["NCOLS"][0])
        nrows = int(header_tokens["NROWS"][0])
        xll = float(header_tokens["XLLCORNER"][0])
        yll = float(header_tokens["YLLCORNER"][0])
        cellsize = header_tokens["CELLSIZE"]
        dx = float(cellsize[0])
        dy = float(cellsize[1]) if len(cellsize) > 1 else dx
        nodata = float(header_tokens["NODATA_VALUE"][0])

        values: list[float] = []
        for line in handle:
            striped = line.strip()
            if not striped:
                continue
            values.extend(float(v) for v in striped.split())

    expected = ncols * nrows
    if len(values) != expected:
        raise ValueError(f"Grid size mismatch: expected {expected} values, found {len(values)}")

    return ncols, nrows, xll, yll, dx, dy, nodata, values


def parse_utm_from_prj(prj_path: pathlib.Path) -> tuple[int, str] | None:
    if not prj_path.exists():
        return None
    text = prj_path.read_text(encoding="utf-8", errors="replace")
    upper = text.upper()
    marker = "UTM_ZONE_"
    idx = upper.find(marker)
    if idx < 0:
        return None
    tail = upper[idx + len(marker):]
    digits = []
    hemi = "N"
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
            continue
        if ch in ("N", "S"):
            hemi = ch
            break
        if digits:
            break
    if not digits:
        return None
    return int("".join(digits)), hemi


def utm_to_latlon(easting: float, northing: float, zone: int, hemisphere: str) -> tuple[float, float]:
    # WGS84 UTM to geographic conversion for Google Earth KML output.
    a = 6378137.0
    e_sq = 0.0066943799901413165
    e_prime_sq = e_sq / (1.0 - e_sq)
    k0 = 0.9996

    x = easting - 500000.0
    y = northing
    if hemisphere.upper().startswith("S"):
        y -= 10000000.0

    m = y / k0
    mu = m / (a * (1 - e_sq / 4 - 3 * e_sq * e_sq / 64 - 5 * e_sq ** 3 / 256))

    e1 = (1 - math.sqrt(1 - e_sq)) / (1 + math.sqrt(1 - e_sq))
    j1 = 3 * e1 / 2 - 27 * e1 ** 3 / 32
    j2 = 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32
    j3 = 151 * e1 ** 3 / 96
    j4 = 1097 * e1 ** 4 / 512

    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)

    c1 = e_prime_sq * cos_fp ** 2
    t1 = tan_fp ** 2
    r1 = a * (1 - e_sq) / ((1 - e_sq * sin_fp ** 2) ** 1.5)
    n1 = a / math.sqrt(1 - e_sq * sin_fp ** 2)
    d = x / (n1 * k0)

    q1 = n1 * tan_fp / r1
    q2 = d ** 2 / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e_prime_sq) * d ** 4 / 24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * e_prime_sq - 3 * c1 ** 2) * d ** 6 / 720
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * d ** 3 / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e_prime_sq + 24 * t1 ** 2) * d ** 5 / 120
    lon = (zone - 1) * 6 - 180 + 3 + math.degrees((q5 - q6 + q7) / cos_fp)
    lat = math.degrees(lat)
    return lon, lat


def latlon_to_utm(lon: float, lat: float, zone: int | None = None) -> tuple[float, float, int, str]:
    a = 6378137.0
    f = 1 / 298.257223563
    e_sq = f * (2 - f)
    e_prime_sq = e_sq / (1 - e_sq)
    k0 = 0.9996

    if zone is None:
        zone = int((lon + 180) / 6) + 1
    hemisphere = "N" if lat >= 0 else "S"

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
        aa
        + (1 - t + c) * aa ** 3 / 6
        + (5 - 18 * t + t ** 2 + 72 * c - 58 * e_prime_sq) * aa ** 5 / 120
    ) + 500000.0

    northing = k0 * (
        m
        + n
        * math.tan(lat_rad)
        * (
            aa ** 2 / 2
            + (5 - t + 9 * c + 4 * c ** 2) * aa ** 4 / 24
            + (61 - 58 * t + t ** 2 + 600 * c - 330 * e_prime_sq) * aa ** 6 / 720
        )
    )
    if hemisphere == "S":
        northing += 10000000.0
    return easting, northing, zone, hemisphere


def parse_lonlat(coord_text: str) -> tuple[float, float]:
    parts = [p.strip() for p in coord_text.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid coordinate text: {coord_text}")
    return float(parts[0]), float(parts[1])


def find_vent_lonlat(root: et.Element) -> tuple[float, float] | None:
    for pm in root.findall(f".//{{{KML_NS}}}Placemark"):
        style = pm.find(f"{{{KML_NS}}}styleUrl")
        point_coords = pm.find(f".//{{{KML_NS}}}Point/{{{KML_NS}}}coordinates")
        if point_coords is None or point_coords.text is None:
            continue
        if style is not None and (style.text or "").strip() == "#VolcanoMarker":
            return parse_lonlat(point_coords.text.strip())
    return None


def looks_like_local_meters(xll: float, yll: float, dx: float, dy: float, zone: int) -> bool:
    # UTM absolute ranges are roughly: Easting 100000..900000, Northing 0..10000000.
    # If values are small/negative, this is typically local meter offsets from a vent origin.
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return False
    return not (100000.0 <= xll <= 900000.0 and -100000.0 <= yll <= 10000000.0 and 1 <= zone <= 60)


def add_text(parent: et.Element, tag: str, text: str) -> et.Element:
    node = et.SubElement(parent, f"{{{KML_NS}}}{tag}")
    node.text = text
    return node


def format_coord(lon: float, lat: float, alt: float = 0.0) -> str:
    return f"{lon:0.6E},{lat:0.6E},{alt:0.6E}"


def build_placemark(mode: str, value: float, x0: float, y0: float, x1: float, y1: float) -> et.Element:
    placemark = et.Element(f"{{{KML_NS}}}Placemark")
    unit = "mm" if mode == "thickness" else "hrs"
    add_text(placemark, "name", f"{value:9.2f} {unit}")
    add_text(placemark, "styleUrl", f"#{choose_style(value, mode)}")
    # Keep a consistent numeric field for downstream use in Google Earth attributes.
    ext = et.SubElement(placemark, f"{{{KML_NS}}}ExtendedData")
    d = et.SubElement(ext, f"{{{KML_NS}}}Data", name="S_obs")
    add_text(d, "value", f"{value:.8g}")

    multi = et.SubElement(placemark, f"{{{KML_NS}}}MultiGeometry")

    point = et.SubElement(multi, f"{{{KML_NS}}}Point")
    add_text(point, "altitudeMode", "absolute")
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    add_text(point, "coordinates", format_coord(center_x, center_y))

    polygon = et.SubElement(multi, f"{{{KML_NS}}}Polygon")
    add_text(polygon, "extrude", "0")
    add_text(polygon, "tessellate", "1")
    add_text(polygon, "altitudeMode", "clampToGround")
    outer = et.SubElement(polygon, f"{{{KML_NS}}}outerBoundaryIs")
    ring = et.SubElement(outer, f"{{{KML_NS}}}LinearRing")
    coords = et.SubElement(ring, f"{{{KML_NS}}}coordinates")
    coords.text = "\n                 " + "\n                 ".join(
        [
            format_coord(x0, y0),
            format_coord(x1, y0),
            format_coord(x1, y1),
            format_coord(x0, y1),
            format_coord(x0, y0),
        ]
    ) + "\n                   "

    return placemark


def write_kmz(output_kmz: pathlib.Path, kml_name: str, tree: et.ElementTree) -> None:
    output_kmz.parent.mkdir(parents=True, exist_ok=True)
    kml_bytes = et.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(output_kmz, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(kml_name, kml_bytes)


def strip_usgs_tagging(root: et.Element) -> None:
    document = root.find(f"{{{KML_NS}}}Document")
    if document is None:
        return

    markers = ("usgs", "vsc-ash.wr.usgs.gov")
    parent_map = {child: parent for parent in root.iter() for child in parent}
    to_remove: list[et.Element] = []

    for node in root.iter():
        if node is root or node is document:
            continue
        blob = et.tostring(node, encoding="unicode", method="xml").lower()
        if any(m in blob for m in markers):
            to_remove.append(node)

    # Remove deepest nodes first to avoid parent-child removal conflicts.
    for node in sorted(to_remove, key=lambda n: len(list(n.iter())), reverse=True):
        parent = parent_map.get(node)
        if parent is None:
            continue
        try:
            parent.remove(node)
        except ValueError:
            pass


def isolate_final_layer(root: et.Element, final_folder: et.Element) -> None:
    document = root.find(f"{{{KML_NS}}}Document")
    if document is None:
        return

    parent_map = {child: parent for parent in root.iter() for child in parent}
    parent = parent_map.get(final_folder)
    if parent is not None:
        parent.remove(final_folder)
    document.append(final_folder)

    for child in list(document):
        if child is final_folder:
            continue
        tag = child.tag
        if tag in (f"{{{KML_NS}}}Folder", f"{{{KML_NS}}}Placemark", f"{{{KML_NS}}}ScreenOverlay"):
            document.remove(child)


def process_grid(
    template_kmz: pathlib.Path,
    grid_path: pathlib.Path,
    output_kmz: pathlib.Path,
    mode: str,
    min_value: float,
    input_crs: str,
    utm_zone: int | None,
    utm_hemi: str | None,
    remove_usgs: bool,
) -> tuple[int, str]:
    tree, kml_name = load_template_kml(template_kmz)
    root = tree.getroot()
    vent_from_template = find_vent_lonlat(root)
    final_folder = find_final_folder(root)
    if remove_usgs:
        strip_usgs_tagging(root)
    isolate_final_layer(root, final_folder)

    for child in list(final_folder):
        if child.tag == f"{{{KML_NS}}}Placemark":
            final_folder.remove(child)

    ncols, nrows, xll, yll, dx, dy, nodata, values = parse_ascii_grid(grid_path)

    crs = input_crs
    if crs == "auto":
        prj_guess = grid_path.with_suffix(".prj")
        parsed = parse_utm_from_prj(prj_guess)
        if parsed:
            crs = "utm"
            utm_zone, utm_hemi = parsed
        else:
            # If coordinates look like local-meter offsets, prefer UTM.
            if utm_zone is not None and utm_hemi is not None and looks_like_local_meters(xll, yll, dx, dy, utm_zone):
                crs = "utm"
            else:
                crs = "lonlat"

    if crs == "utm" and (utm_zone is None or utm_hemi is None):
        raise ValueError("UTM conversion selected but zone/hemisphere are missing. Provide .prj or --utm-zone/--utm-hemi.")

    vent_anchor: tuple[float, float] | None = None
    if crs == "utm" and utm_zone is not None and looks_like_local_meters(xll, yll, dx, dy, utm_zone):
        if vent_from_template is not None:
            ve, vn, _, _ = latlon_to_utm(vent_from_template[0], vent_from_template[1], zone=utm_zone)
            vent_anchor = (ve, vn)

    added = 0
    for row in range(nrows):
        for col in range(ncols):
            value = values[row * ncols + col]
            if value == nodata or value < min_value:
                continue

            x0_raw = xll + (col * dx)
            x1_raw = xll + ((col + 1) * dx)
            y1_raw = yll + ((nrows - row) * dy)
            y0_raw = yll + ((nrows - row - 1) * dy)

            if crs == "utm":
                if vent_anchor is not None:
                    x0_raw += vent_anchor[0]
                    x1_raw += vent_anchor[0]
                    y0_raw += vent_anchor[1]
                    y1_raw += vent_anchor[1]
                lon0, lat0 = utm_to_latlon(x0_raw, y0_raw, utm_zone, utm_hemi)
                lon1, lat1 = utm_to_latlon(x1_raw, y1_raw, utm_zone, utm_hemi)
                x0, y0, x1, y1 = lon0, lat0, lon1, lat1
            else:
                x0, y0, x1, y1 = x0_raw, y0_raw, x1_raw, y1_raw

            final_folder.append(build_placemark(mode, value, x0, y0, x1, y1))
            added += 1

    write_kmz(output_kmz, pathlib.Path(output_kmz).with_suffix(".kml").name, tree)
    crs_note = crs if vent_anchor is None else f"{crs}+local-anchor"
    return added, crs_note


def find_default_template(cwd: pathlib.Path) -> pathlib.Path | None:
    kmzs = sorted(cwd.glob("*.kmz"))
    if not kmzs:
        return None

    def is_generated(path: pathlib.Path) -> bool:
        return path.name.lower().startswith("generated_")

    preferred = [p for p in kmzs if "tephra4d" in p.name.lower() and not is_generated(p)]
    if preferred:
        return preferred[0]
    preferred = [p for p in kmzs if ("thickness" in p.name.lower() or "arrival" in p.name.lower()) and not is_generated(p)]
    if preferred:
        return preferred[0]
    preferred = [p for p in kmzs if not is_generated(p)]
    if preferred:
        return preferred[0]
    return kmzs[0]


def find_batch_root(cwd: pathlib.Path, grid_name: str) -> pathlib.Path | None:
    # Prefer event-like folders that contain maps/full maps/shapefiles.
    for child in sorted(cwd.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / "maps" / "full maps" / "shapefiles"
        if candidate.is_dir():
            return candidate

    grid_files = sorted(cwd.rglob(grid_name))
    if not grid_files:
        return None

    # Walk ancestors and pick the first one that clearly contains scenario subfolders.
    first_parent = grid_files[0].parent
    for ancestor in [first_parent, *first_parent.parents]:
        if ancestor == cwd.parent:
            break
        scenario_count = sum(1 for child in ancestor.iterdir() if child.is_dir() and (child / grid_name).is_file())
        if scenario_count >= 2:
            return ancestor
    return None


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
                args.output_dir = search_root / "kmz_per_scenario"
        else:
            grid_files = sorted(search_root.rglob(args.grid_name))
            if grid_files:
                args.grid = grid_files[0]
                if args.output_kmz is None:
                    args.output_kmz = search_root / f"generated_{grid_files[0].stem}.kmz"

    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate KMZ(s) from ASCII grid data using an existing KMZ as template."
    )
    parser.add_argument("--template-kmz", type=pathlib.Path, help="Reference KMZ to copy style/layout from.")
    parser.add_argument("--mode", choices=["auto", "thickness", "arrival"], default="auto")
    parser.add_argument("--min-value", type=float, default=None)
    parser.add_argument("--input-crs", choices=["auto", "lonlat", "utm"], default="utm")
    parser.add_argument("--utm-zone", type=int, default=51)
    parser.add_argument("--utm-hemi", choices=["N", "S"], default="N")
    parser.add_argument("--remove-usgs-tagging", action="store_true", help="Remove USGS screen overlays/branding from output KML.")

    parser.add_argument("--grid", type=pathlib.Path, help="Single grid file (.asc/.txt/.dat).")
    parser.add_argument("--output-kmz", type=pathlib.Path, help="Single output KMZ path.")
    parser.add_argument("--batch-root", type=pathlib.Path, help="Root folder containing scenario subfolders.")
    parser.add_argument("--data-root", type=pathlib.Path, help="Root directory to search for template/grid when running without explicit paths.")
    parser.add_argument("--grid-name", default="tephra_s_obs.asc", help="Grid filename expected in each scenario folder.")
    parser.add_argument("--output-dir", type=pathlib.Path, help="Batch output directory.")
    args = parser.parse_args()
    args = apply_ide_defaults(args, sys.argv)

    if args.template_kmz is None:
        raise ValueError("No template KMZ found. Provide --template-kmz or place a template .kmz in the working directory.")

    mode = infer_mode(args.template_kmz, args.mode)
    min_value = args.min_value if args.min_value is not None else (0.001 if mode == "thickness" else 0.0)

    if args.batch_root:
        if not args.output_dir:
            raise ValueError("--output-dir is required with --batch-root")
        scenarios = sorted([p for p in args.batch_root.iterdir() if p.is_dir()])
        if not scenarios:
            raise ValueError(f"No scenario folders found in {args.batch_root}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        success = 0
        for scenario in scenarios:
            grid_path = scenario / args.grid_name
            if not grid_path.exists():
                print(f"Skip {scenario.name}: missing {args.grid_name}")
                continue
            output_kmz = args.output_dir / f"{scenario.name}.kmz"
            added, crs = process_grid(
                template_kmz=args.template_kmz,
                grid_path=grid_path,
                output_kmz=output_kmz,
                mode=mode,
                min_value=min_value,
                input_crs=args.input_crs,
                utm_zone=args.utm_zone,
                utm_hemi=args.utm_hemi,
                remove_usgs=args.remove_usgs_tagging,
            )
            print(f"Wrote {output_kmz} with {added} placemarks (mode={mode}, min_value={min_value}, crs={crs}).")
            success += 1
        print(f"Batch complete: {success} KMZ file(s) created in {args.output_dir}")
        return

    if not args.grid or not args.output_kmz:
        raise ValueError("Single-file mode requires --grid and --output-kmz.")
    added, crs = process_grid(
        template_kmz=args.template_kmz,
        grid_path=args.grid,
        output_kmz=args.output_kmz,
        mode=mode,
        min_value=min_value,
        input_crs=args.input_crs,
        utm_zone=args.utm_zone,
        utm_hemi=args.utm_hemi,
        remove_usgs=args.remove_usgs_tagging,
    )
    print(f"Wrote {args.output_kmz} with {added} placemarks (mode={mode}, min_value={min_value}, crs={crs}).")


if __name__ == "__main__":
    main()
