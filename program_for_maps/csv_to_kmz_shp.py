import argparse
import csv
import math
import struct
import zipfile
from pathlib import Path


def utm_to_latlon(easting: float, northing: float, zone: int, hemisphere: str) -> tuple[float, float]:
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


def write_kmz(points: list[dict], out_kmz: Path) -> None:
    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "  <Document>",
        "    <name>csv_points</name>",
        '    <Style id="pt"><IconStyle><scale>0.7</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>',
    ]
    for i, p in enumerate(points, start=1):
        name = (p.get("name") or f"point_{i}").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        desc = (p.get("desc") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lon = float(p["lon"])
        lat = float(p["lat"])
        kml_lines.extend(
            [
                "    <Placemark>",
                f"      <name>{name}</name>",
                f"      <description>{desc}</description>",
                "      <styleUrl>#pt</styleUrl>",
                "      <Point>",
                f"        <coordinates>{lon:.8f},{lat:.8f},0</coordinates>",
                "      </Point>",
                "    </Placemark>",
            ]
        )
    kml_lines.extend(["  </Document>", "</kml>"])
    kml_text = "\n".join(kml_lines)

    out_kmz.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_kmz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text.encode("utf-8"))


def write_dbf(path: Path, rows: list[dict]) -> None:
    fields = [
        ("ID", "N", 10, 0),
        ("NAME", "C", 80, 0),
        ("VALUE", "C", 40, 0),
    ]
    nrec = len(rows)
    nfields = len(fields)
    header_len = 32 + 32 * nfields + 1
    rec_len = 1 + sum(f[2] for f in fields)

    with path.open("wb") as f:
        f.write(struct.pack("<BBBBIHH20x", 3, 126, 1, 1, nrec, header_len, rec_len))
        for name, ftype, size, dec in fields:
            bname = name.encode("ascii", errors="ignore")[:10]
            bname = bname + b"\x00" * (11 - len(bname))
            f.write(struct.pack("<11sc4xBB14x", bname, ftype.encode("ascii"), size, dec))
        f.write(b"\x0D")

        for r in rows:
            f.write(b" ")
            vals = {
                "ID": str(int(r["id"])).rjust(10),
                "NAME": str(r.get("name", ""))[:80].ljust(80),
                "VALUE": str(r.get("value", ""))[:40].ljust(40),
            }
            for name, _, size, _ in fields:
                f.write(vals[name][:size].encode("ascii", errors="ignore"))
        f.write(b"\x1A")


def write_shp(base: Path, points: list[dict], coord_system: str) -> None:
    shp_path = base.with_suffix(".shp")
    shx_path = base.with_suffix(".shx")
    dbf_path = base.with_suffix(".dbf")
    prj_path = base.with_suffix(".prj")

    if points:
        xs = [float(p["x"]) for p in points]
        ys = [float(p["y"]) for p in points]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    else:
        xmin = ymin = xmax = ymax = 0.0

    shp_records = []
    shx_records = []
    offset_words = 50
    for i, p in enumerate(points, start=1):
        content = struct.pack("<idd", 1, float(p["x"]), float(p["y"]))
        content_len_words = len(content) // 2
        shp_records.append(struct.pack(">ii", i, content_len_words) + content)
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

    write_dbf(dbf_path, points)

    if coord_system == "lonlat":
        prj = (
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
            'SPHEROID["WGS_1984",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
        )
    else:
        # UTM Zone 51N default projection text.
        prj = (
            'PROJCS["WGS_1984_UTM_Zone_51N",'
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
            'SPHEROID["WGS_1984",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],'
            'PARAMETER["False_Easting",500000],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",123],'
            'PARAMETER["Scale_Factor",0.9996],'
            'PARAMETER["Latitude_Of_Origin",0],'
            'UNIT["Meter",1]]'
        )
    prj_path.write_text(prj, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV points to KMZ and Shapefile.")
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument("--out-dir", default=None, help="Output folder. Default: same as CSV.")
    parser.add_argument("--basename", default=None, help="Output file base name. Default: CSV stem.")
    parser.add_argument("--coord-system", choices=["utm", "lonlat"], default="utm")
    parser.add_argument("--x-col", default="LONG")
    parser.add_argument("--y-col", default="LAT")
    parser.add_argument("--z-col", default="Z")
    parser.add_argument("--name-col", default="Barangay")
    parser.add_argument("--value-col", default="S_obs")
    parser.add_argument("--utm-zone", type=int, default=51)
    parser.add_argument("--utm-hemi", choices=["N", "S"], default="N")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--skip-zero-triple", action="store_true", default=True)
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.basename if args.basename else csv_path.stem
    out_base = out_dir / base
    out_kmz = out_dir / f"{base}.kmz"

    points_for_kmz: list[dict] = []
    points_for_shp: list[dict] = []
    skipped = 0
    with csv_path.open("r", encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            try:
                x = float((row.get(args.x_col) or "").strip())
                y = float((row.get(args.y_col) or "").strip())
            except Exception:
                skipped += 1
                continue

            z = None
            z_raw = row.get(args.z_col)
            if z_raw is not None and str(z_raw).strip() != "":
                try:
                    z = float(str(z_raw).strip())
                except Exception:
                    z = None

            if args.skip_zero_triple:
                z0 = (z is not None and abs(z) < 1e-9) or (z is None)
                if abs(x) < 1e-9 and abs(y) < 1e-9 and z0:
                    skipped += 1
                    continue

            name = (row.get(args.name_col) or f"point_{idx}").strip()
            value = (row.get(args.value_col) or "").strip()
            desc = f"{args.name_col}: {name}"
            if value:
                desc += f" | {args.value_col}: {value}"

            if args.coord_system == "utm":
                lon, lat = utm_to_latlon(x, y, args.utm_zone, args.utm_hemi)
            else:
                lon, lat = x, y

            points_for_kmz.append(
                {
                    "name": name,
                    "desc": desc,
                    "lon": lon,
                    "lat": lat,
                }
            )
            points_for_shp.append(
                {
                    "id": idx,
                    "name": name,
                    "value": value,
                    "x": x,
                    "y": y,
                }
            )

    write_kmz(points_for_kmz, out_kmz)
    write_shp(out_base, points_for_shp, args.coord_system)
    print(f"CSV: {csv_path}")
    print(f"KMZ: {out_kmz}")
    print(f"SHP: {out_base.with_suffix('.shp')}")
    print(f"Points written: {len(points_for_shp)} | Skipped: {skipped}")


if __name__ == "__main__":
    main()

