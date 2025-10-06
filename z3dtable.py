#!/usr/bin/env python3
"""
Z3D Table Editor — RX/RXC helper (with keyboard shortcuts, RXC dropdown, and RXC→Z3D apply)

- CSV-like table for .z3d files (recursive folder load)
- Start/end times (local tz), duration, rate
- Edit UTM easting/northing inline (double-click), write back to headers (Lat/Lon in radians)
- Load TX folder and highlight overlapping RX files (green with black text)
- Load RXC (Waypoints.rxc), auto-match nearest station, and edit RXC_STN via dropdown
- Command to apply RXC station coords to selected Z3D files (updates file headers)
- Menus + scrollbars + sortable headers + "Fit Columns to Window" view preset
- Proper accelerator labels + working keyboard shortcuts (macOS/Windows/Linux)

All indentation uses spaces (4 spaces). No tabs.
"""

from __future__ import annotations
import argparse
import csv
import io
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, font as tkfont

# ---------- Helpers: file gathering ----------

def _gather_z3d_recursive(root: Path) -> List[Path]:
    files: List[Path] = []
    for fp in root.rglob("*"):
        try:
            if fp.is_file() and fp.suffix.lower() == ".z3d":
                files.append(fp)
        except Exception:
            pass
    return sorted(files)

# ---------- Time / GPS helpers ----------

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_UTC_LEAP_SECONDS = 18  # constant (update if needed)

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

def to_zone(dt_utc: datetime, tz: str) -> datetime:
    if tz is None:
        return dt_utc
    if ZoneInfo is not None:
        try:
            return dt_utc.astimezone(ZoneInfo(tz))
        except Exception:
            pass
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", tz.strip())
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hh, mm = int(m.group(2)), int(m.group(3))
        return dt_utc.astimezone(timezone(sign * timedelta(hours=hh, minutes=mm)))
    return dt_utc

def find_gps_markers_int32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.int32 or arr.size < 3:
        return np.array([], dtype=np.int64)
    p1 = np.where((arr[:-2] == 0x7FFFFFFF) & (arr[1:-1] == -2147483648))[0]
    p0 = np.where((arr[:-2] == -1) & (arr[1:-1] == -2147483648))[0]
    if p1.size == 0 and p0.size == 0:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate([p1, p0]))

def first_last_sow(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        arr = np.fromfile(path, dtype="<i4")
    except Exception:
        return None, None
    idx = find_gps_markers_int32(arr)
    if idx.size == 0:
        try:
            arr_be = np.fromfile(path, dtype=">i4")
            idx = find_gps_markers_int32(arr_be)
            if idx.size == 0:
                return None, None
            arr_u = arr_be.view(np.uint32)
        except Exception:
            return None, None
    else:
        arr_u = arr.view(np.uint32)
    idx = idx[idx + 2 < arr_u.size]
    if idx.size == 0:
        return None, None
    gps_words = arr_u[idx + 2]
    sow = np.floor(gps_words / 1024.0).astype(np.int64)  # whole second
    return int(sow[0]), int(sow[-1])

def gps_to_utc(gps_seconds: int) -> datetime:
    return GPS_EPOCH + timedelta(seconds=gps_seconds - GPS_UTC_LEAP_SECONDS)

# ---------- WGS84 / UTM ----------

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1 - WGS84_E2)
K0 = 0.9996

def utm_zone_from_lon(lon_deg: float) -> int:
    z = int((lon_deg + 180) // 6) + 1
    return max(1, min(60, z))

def latlon_to_utm(lat_deg: float, lon_deg: float) -> Tuple[float, float, int, str]:
    zone = utm_zone_from_lon(lon_deg)
    hem = 'N' if lat_deg >= 0 else 'S'
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    e2 = WGS84_E2
    a = WGS84_A
    sinp = math.sin(phi)
    cosp = math.cos(phi)
    tanp = math.tan(phi)
    N = a / math.sqrt(1 - e2 * sinp * sinp)
    T = tanp * tanp
    C = WGS84_EP2 * cosp * cosp
    A = (lam - lam0) * cosp
    e4 = e2 * e2
    e6 = e4 * e2
    M = a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
             - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
             + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
             - (35 * e6 / 3072) * math.sin(6 * phi))
    E = K0 * N * (A + (1 - T + C) * A ** 3 / 6
                  + (5 - 18 * T + T ** 2 + 72 * C - 58 * WGS84_EP2) * A ** 5 / 120) + 500000.0
    Nnorth = K0 * (M + N * tanp * (A ** 2 / 2
                   + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                   + (61 - 58 * T + T ** 2 + 600 * C - 330 * WGS84_EP2) * A ** 6 / 720))
    if lat_deg < 0:
        Nnorth += 10000000.0
    return E, Nnorth, zone, hem

def utm_to_latlon(E: float, Nnorth: float, zone: int, hem: str) -> Tuple[float, float]:
    e2 = WGS84_E2
    ep2 = WGS84_EP2
    a = WGS84_A
    x = E - 500000.0
    y = Nnorth
    if hem.upper().startswith('S'):
        y -= 10000000.0
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)

    M = y / K0
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 * e2 * e2 / 256))

    phi1 = (mu
        + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
        + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
        + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
        + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))

    sin1 = math.sin(phi1)
    cos1 = math.cos(phi1)
    tan1 = math.tan(phi1)
    N1 = a / math.sqrt(1 - e2 * sin1 * sin1)
    R1 = N1 * (1 - e2) / (1 - e2 * sin1 * sin1)
    D = x / (N1 * K0)

    lat = (phi1 - (N1 * tan1 / R1) * (D * D / 2
        - (5 + 3 * tan1 * tan1 + 10 * ep2 * cos1 * cos1 - 4 * ep2 * ep2 - 9 * ep2) * D ** 4 / 24
        + (61 + 90 * tan1 * tan1 + 298 * ep2 * cos1 * cos1 + 45 * tan1 ** 4 - 252 * ep2 - 3 * ep2 ** 2) * D ** 6 / 720))

    lon = (lam0 + (D - (1 + 2 * tan1 * tan1 + ep2 * cos1 * cos1) * D ** 3 / 6
        + (5 - 2 * ep2 * cos1 * cos1 + 28 * tan1 * tan1 - 3 * ep2 ** 2 + 8 * ep2 + 24 * tan1 ** 4) * D ** 5 / 120) / cos1)

    return math.degrees(lat), math.degrees(lon)

# ---------- Z3D parsing & editing ----------

@dataclass
class Z3DRow:
    ch: int
    file: Path
    start_time_local: str
    end_time_local: str
    duration_min: float
    rate_Hz: float
    easting_m: float
    northing_m: float
    zone: int
    hem: str
    rxc_stn: str = ""
    rxc_e: float = 0.0
    rxc_n: float = 0.0
    start_utc: Optional[datetime] = None
    end_utc: Optional[datetime] = None

HEADER_SCAN_BYTES = 2_000_000  # 2 MB

# bytes regex for Lat/Lon (accept = or := and scientific notation)
NUM = rb"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
LAT_PAT = re.compile(rb"\b(?:Lat|Latitude)\s*[:=]{1,2}\s*" + NUM, re.IGNORECASE)
LON_PAT = re.compile(rb"\b(?:Lon|Long|Longitude)\s*[:=]{1,2}\s*" + NUM, re.IGNORECASE)

def parse_header_for_week_rate_latlon(path: Path) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    with path.open("rb") as f:
        raw = f.read(HEADER_SCAN_BYTES)

    def _find_first_int(pattern: bytes) -> Optional[int]:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1).decode("ascii", "ignore"))
        except Exception:
            return None

    def _find_first_float(pattern: bytes) -> Optional[float]:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1).decode("ascii", "ignore"))
        except Exception:
            return None

    week = _find_first_int(br"gpsweek\s*[:=]{1,2}\s*(\d+)")
    rate = _find_first_float(br"rate\s*[:=]{1,2}\s*([0-9]+(?:\.[0-9]+)?)")

    lat = lon = None
    lats = list(LAT_PAT.finditer(raw))
    lons = list(LON_PAT.finditer(raw))
    if lats and lons:
        lat_b = lats[-1].group(1)
        lon_b = lons[-1].group(1)
        lat_v = float(lat_b.decode("ascii", "ignore"))
        lon_v = float(lon_b.decode("ascii", "ignore"))
        # If stored in radians (typical), convert to degrees
        if abs(lat_v) <= 3.2 and abs(lon_v) <= 3.2:
            lat = math.degrees(lat_v)
            lon = math.degrees(lon_v)
        else:
            lat = lat_v
            lon = lon_v
    return week, rate, lat, lon

def _byte_replace_all(raw: bytearray, pattern: re.Pattern, new_val: float) -> int:
    cnt = 0
    for m in list(pattern.finditer(raw)):
        start, end = m.span(1)
        old_field = raw[start:end]
        old_len = len(old_field)
        if b'e' in old_field or b'E' in old_field:
            new_str = f"{new_val:.12E}"
        else:
            new_str = f"{new_val:.12f}"
        new_bytes = new_str.encode("ascii")
        if len(new_bytes) < old_len:
            new_bytes = new_bytes + b" " * (old_len - len(new_bytes))
        else:
            new_bytes = new_bytes[:old_len]
        raw[start:end] = new_bytes
        cnt += 1
    return cnt

def update_header_latlon_inplace(path: Path, lat_rad: float, lon_rad: float) -> int:
    with path.open("r+b") as f:
        raw = bytearray(f.read(HEADER_SCAN_BYTES))
        changed = 0
        changed += _byte_replace_all(raw, LAT_PAT, lat_rad)
        changed += _byte_replace_all(raw, LON_PAT, lon_rad)
        f.seek(0)
        f.write(raw)
    return changed

# ---------- RXC parsing ----------

def _parse_float_maybe(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        try:
            return float(s.replace(",", "."))
        except Exception:
            return None

def _parse_rxc_generic(path: Path) -> List[dict]:
    """
    ZenPlan Waypoints.rxc friendly parser with header preamble tolerance.
    It searches for the header line containing 'Rx.Stn' and parses CSV data below it.
    """
    lines = path.read_text(errors="ignore").splitlines()

    zone_hint = None
    for ln in lines:
        lns = ln.strip()
        if lns.startswith("$Survey.UTMZone"):
            try:
                zone_hint = int(lns.split("=", 1)[1].strip().split()[0])
            except Exception:
                pass

    header_idx = None
    for i, ln in enumerate(lines):
        if "Rx.Stn" in ln or "rx.stn" in ln.lower():
            header_idx = i
            break
    if header_idx is None:
        return []

    headers = [h.strip() for h in lines[header_idx].split(",")]
    data_lines: List[str] = []
    for ln in lines[header_idx + 1:]:
        lns = ln.strip()
        if not lns or lns.startswith("//") or lns.startswith("$"):
            continue
        data_lines.append(ln)

    csv_text = "\n".join([",".join(headers)] + data_lines)
    reader = csv.DictReader(io.StringIO(csv_text))

    def field_by_priority(cands: List[str]) -> Optional[str]:
        for cand in cands:
            for h in headers:
                if h.strip().lower() == cand.lower():
                    return h
        return None

    east_field = field_by_priority(["East0", "Easting0", "X0", "East1", "Easting1", "X1", "East2", "Easting2", "X2", "East", "Easting", "X"])
    north_field = field_by_priority(["North0", "Northing0", "Y0", "North1", "Northing1", "Y1", "North2", "Northing2", "Y2", "North", "Northing", "Y"])
    lat_field = field_by_priority(["GPS.Lat", "Lat", "Latitude"])

    pts: List[dict] = []
    for row in reader:
        stn = str(row.get("Rx.Stn") or row.get("RX.STN") or row.get("Stn") or "").strip()
        if not stn:
            continue
        e_raw = row.get(east_field) if east_field else None
        n_raw = row.get(north_field) if north_field else None
        if e_raw is None or n_raw is None:
            continue
        e = _parse_float_maybe(str(e_raw))
        n = _parse_float_maybe(str(n_raw))
        if e is None or n is None:
            continue
        zone = zone_hint or 0
        hem = "N"
        if lat_field and row.get(lat_field):
            try:
                latv = float(str(row.get(lat_field)))
                hem = "S" if latv < 0 else "N"
            except Exception:
                pass
        pts.append({"stn": stn, "e": float(e), "n": float(n), "zone": int(zone), "hem": hem})
    return pts
    # ---------- Misc helpers ----------

def extract_channel(p: Path) -> int:
    m = re.search(r"[Cc]h(\d+)", p.name)
    return int(m.group(1)) if m else 0

def _open_external(paths: List[Path]) -> Tuple[int, List[str]]:
    if not paths:
        return 0, []
    errs: List[str] = []
    count = 0
    if sys.platform == "darwin":
        try:
            subprocess.check_call(["open"] + [str(p) for p in paths])
            count = len(paths)
        except Exception as e:
            errs.append(str(e))
    elif os.name == "nt":
        for p in paths:
            try:
                os.startfile(str(p))  # type: ignore[attr-defined]
                count += 1
            except Exception as e:
                errs.append(f"{p}: {e}")
    else:
        for p in paths:
            try:
                subprocess.check_call(["xdg-open", str(p)])
                count += 1
            except Exception as e:
                errs.append(f"{p}: {e}")
    return count, errs

def _suggest_tx_initialdir(files: List[Path]) -> Path:
    if not files:
        return Path.home()
    candidates: List[Path] = []
    for f in files:
        parts = list(f.resolve().parts)
        for i, seg in enumerate(parts):
            if seg.lower() == "rx":
                parent = Path(*parts[:i]).resolve() if i > 0 else Path("/")
                candidates.append(parent)
                break
    if candidates:
        return min(candidates, key=lambda x: len(x.as_posix()))
    try:
        from os.path import commonpath
        root = Path(commonpath([str(f.resolve()) for f in files]))
        return root.parent if root.parent.exists() else root
    except Exception:
        return files[0].resolve().parent

# ---------- Core loading ----------

def load_rows(paths: List[Path], tz: str) -> List[Z3DRow]:
    rows: List[Z3DRow] = []
    for p in paths:
        if not p.exists():
            continue
        gpsweek, rate, lat_deg, lon_deg = parse_header_for_week_rate_latlon(p)
        sow0, sow1 = first_last_sow(p)
        start_local = end_local = ""
        start_utc_dt = end_utc_dt = None
        duration_min = 0.0
        if gpsweek is not None and sow0 is not None:
            start_utc_dt = gps_to_utc(gpsweek * 604800 + sow0)
            start_local = to_zone(start_utc_dt, tz).isoformat(sep=" ")
        if gpsweek is not None and sow1 is not None:
            end_utc_dt = gps_to_utc(gpsweek * 604800 + sow1 + 1)
            end_local = to_zone(end_utc_dt, tz).isoformat(sep=" ")
        if (start_utc_dt is not None) and (end_utc_dt is not None):
            duration_min = round((end_utc_dt - start_utc_dt).total_seconds() / 60.0, 3)
        easting = northing = zone = hem = None
        if (lat_deg is not None) and (lon_deg is not None):
            easting, northing, zone, hem = latlon_to_utm(lat_deg, lon_deg)
        rows.append(Z3DRow(
            ch=extract_channel(p),
            file=p,
            start_time_local=start_local,
            end_time_local=end_local,
            duration_min=duration_min,
            rate_Hz=(rate if rate is not None else 0.0),
            easting_m=(round(easting, 3) if easting is not None else 0.0),
            northing_m=(round(northing, 3) if northing is not None else 0.0),
            zone=(zone if zone is not None else 0),
            hem=(hem if hem is not None else "N"),
            start_utc=start_utc_dt,
            end_utc=end_utc_dt,
        ))
    return rows

# ---------- GUI ----------

class Z3DApp(tk.Tk):
    COLS = (
        "Ch", "file", "start_time_local", "end_time_local", "duration_min", "rate_Hz",
        "easting_m", "northing_m", "RXC_STN", "RXC_easting_m", "RXC_northing_m", "dist_to_rxc_m"
    )

    def __init__(self, files: List[Path], tz: str):
        super().__init__()
        self.title("Z3D Table Editor")
        self.geometry("1120x640")

        # State
        self.tz = tz
        self.files = files
        self.rows: List[Z3DRow] = load_rows(files, tz)
        self.row_times: Dict[str, Tuple[Optional[datetime], Optional[datetime]]] = {}
        self.row_geos: Dict[str, Tuple[int, str]] = {}  # iid -> (zone, hem)
        self.rxc_points: List[dict] = []
        self.match_threshold_m: float = 100.0

        # View
        self.fit_to_window = False
        self._fit_var = tk.BooleanVar(value=False)
        self._resize_after_id: Optional[str] = None

        # track any in-place editor widget
        self._active_editor = None

        self._build_widgets()
        self._populate()

    def _build_widgets(self):
        # Accelerator helpers
        MAC = sys.platform == "darwin"

        def _acc(mac_label: str, win_label: str) -> str:
            return mac_label if MAC else win_label

        def _bind_acc(mac_seq: str, win_seq: str, func):
            self.bind_all(mac_seq if MAC else win_seq, lambda e: func())

        # Menubar
        menubar = tk.Menu(self)

        # File
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(
            label="Add RX Folder…",
            command=self.add_rx_folder,
            accelerator=_acc("Cmd+Shift+R", "Ctrl+Shift+R"),
        )
        _bind_acc("<Command-Shift-R>", "<Control-Shift-R>", self.add_rx_folder)

        m_file.add_command(
            label="Load TX Folder…",
            command=self.load_tx_folder,
            accelerator=_acc("Cmd+Shift+T", "Ctrl+Shift+T"),
        )
        _bind_acc("<Command-Shift-T>", "<Control-Shift-T>", self.load_tx_folder)

        m_file.add_command(
            label="Load RXC…",
            command=self.load_rxc,
            accelerator=_acc("Cmd+Shift+X", "Ctrl+Shift+X"),
        )
        _bind_acc("<Command-Shift-X>", "<Control-Shift-X>", self.load_rxc)

        m_file.add_separator()

        m_file.add_command(
            label="Open Selected in External Editor",
            command=self.open_selected_external,
            accelerator=_acc("Cmd+O", "Ctrl+O"),
        )
        _bind_acc("<Command-o>", "<Control-o>", self.open_selected_external)

        m_file.add_command(
            label="Export CSV…",
            command=self.export_csv,
            accelerator=_acc("Cmd+E", "Ctrl+E"),
        )
        _bind_acc("<Command-e>", "<Control-e>", self.export_csv)

        m_file.add_separator()
        m_file.add_command(
            label="Quit",
            command=self.destroy,
            accelerator=_acc("Cmd+Q", "Ctrl+Q"),
        )
        _bind_acc("<Command-q>", "<Control-q>", self.destroy)
        menubar.add_cascade(label="File", menu=m_file)

        # Edit
        m_edit = tk.Menu(menubar, tearoff=0)
        m_edit.add_command(
            label="Apply Changes",
            command=self.apply_changes,
            accelerator=_acc("Cmd+Enter", "Ctrl+Enter"),
        )
        _bind_acc("<Command-Return>", "<Control-Return>", self.apply_changes)

        m_edit.add_command(
            label="Delete Selected",
            command=self.delete_selected,
            accelerator=_acc("Cmd+Backspace", "Del"),
        )
        _bind_acc("<Command-BackSpace>", "<Delete>", self.delete_selected)

        m_edit.add_command(
            label="Delete All Except Highlights",
            command=self.delete_all_except_highlights,
            accelerator=_acc("Cmd+Option+Backspace", "Ctrl+Alt+Del"),
        )
        _bind_acc("<Command-Option-BackSpace>", "<Control-Alt-Delete>", self.delete_all_except_highlights)

        m_edit.add_separator()
        m_edit.add_command(
            label="Refresh",
            command=self.refresh,
            accelerator=_acc("Cmd+R", "Ctrl+R"),
        )
        _bind_acc("<Command-r>", "<Control-r>", self.refresh)

        m_edit.add_command(
            label="Clear Highlights",
            command=self.clear_highlights,
            accelerator=_acc("Cmd+K", "Ctrl+K"),
        )
        _bind_acc("<Command-k>", "<Control-k>", self.clear_highlights)
        menubar.add_cascade(label="Edit", menu=m_edit)

        # RXC
        m_rxc = tk.Menu(menubar, tearoff=0)
        m_rxc.add_command(
            label="Set Max dist (m)…",
            command=self.set_threshold_dialog,
            accelerator=_acc("Cmd+Shift+M", "Ctrl+Shift+M"),
        )
        _bind_acc("<Command-Shift-m>", "<Control-Shift-m>", self.set_threshold_dialog)

        m_rxc.add_command(
            label="Rematch RXC",
            command=self.rematch_rxc,
            accelerator=_acc("Cmd+Shift+G", "Ctrl+Shift+G"),
        )
        _bind_acc("<Command-Shift-g>", "<Control-Shift-g>", self.rematch_rxc)

        m_rxc.add_separator()
        m_rxc.add_command(
            label="Apply RXC → Z3D (Selected)",
            command=self.apply_rxc_to_selected,
            accelerator=_acc("Cmd+Shift+A", "Ctrl+Shift+A"),
        )
        _bind_acc("<Command-Shift-a>", "<Control-Shift-a>", self.apply_rxc_to_selected)

        menubar.add_cascade(label="RXC", menu=m_rxc)

        # View
        self.col_vars = {c: tk.BooleanVar(value=True) for c in self.COLS}
        m_view = tk.Menu(menubar, tearoff=0)
        for c in self.COLS:
            m_view.add_checkbutton(label=c, variable=self.col_vars[c], command=self._update_displaycolumns)
        m_view.add_separator()
        m_view.add_command(label="Show All Columns", command=lambda: self._set_all_columns(True))
        m_view.add_command(label="Hide All But File", command=lambda: self._set_only_file_visible())
        m_view.add_separator()
        m_view.add_checkbutton(
            label="Fit Columns to Window",
            variable=self._fit_var,
            command=lambda: self._toggle_fit_to_window(),
            accelerator=_acc("Cmd+Shift+F", "Ctrl+Shift+F"),
        )
        _bind_acc("<Command-Shift-f>", "<Control-Shift-f>", self._toggle_fit_to_window)
        menubar.add_cascade(label="View", menu=m_view)

        # Help
        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="About", command=lambda: messagebox.showinfo("About", "Z3D Table Editor — RX/RXC helper"))
        menubar.add_cascade(label="Help", menu=m_help)

        self.config(menu=menubar)

        # Treeview + scrollbars
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(container, columns=self.COLS, show="headings", selectmode="extended")
        yscroll = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        try:
            ttk.Sizegrip(container).grid(row=1, column=1, sticky="se")
        except Exception:
            pass

        # Theme & bindings
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Treeview", background="#111315", fieldbackground="#111315", foreground="white", rowheight=22)
        style.map("Treeview", background=[("selected", "#2f6bff")], foreground=[("selected", "white")])

        # Quick keys already bound above; also bind delete keys to tree
        self.tree.bind("<Delete>", lambda e: self.delete_selected())
        self.tree.bind("<BackSpace>", lambda e: self.delete_selected())

        # Headings sortable
        self._base_headings = {c: c for c in self.COLS}
        self._sort_dirs = {c: None for c in self.COLS}
        for col in self.COLS:
            self.tree.heading(col, text=col, command=lambda c=col: self._on_heading_click(c))
            self.tree.column(col, anchor="w")

        # TX highlight tag (green with black text)
        self.tree.tag_configure("tx_overlap", background="#9bf0a2", foreground="black")

        # Inline edit bindings
        self.tree.bind("<Double-1>", self._begin_edit)

        # Refit on resize
        self.bind("<Configure>", self._on_configure)

    # ----- In-place editor manager -----
    def _set_active_editor(self, widget):
        try:
            if self._active_editor is not None and self._active_editor.winfo_exists():
                self._active_editor.destroy()
        except Exception:
            pass
        self._active_editor = widget

    # ----- View helpers -----
    def _visible_columns(self) -> List[str]:
        vis = [c for c, v in self.col_vars.items() if v.get()]
        if not vis:
            vis = ["file"] if "file" in self.COLS else [self.COLS[0]]
            self.col_vars[vis[0]].set(True)
        return vis

    def _update_displaycolumns(self):
        self.tree.configure(displaycolumns=self._visible_columns())
        if self.fit_to_window:
            self._fit_columns_to_window()
        else:
            self._autosize_columns()

    def _set_all_columns(self, flag: bool):
        for v in self.col_vars.values():
            v.set(flag)
        self._update_displaycolumns()

    def _set_only_file_visible(self):
        for c, v in self.col_vars.items():
            v.set(c == "file")
        self._update_displaycolumns()

    # ----- Sorting -----
    def _on_heading_click(self, col: str):
        cur = self._sort_dirs.get(col)
        new_dir = True if cur is None else (not cur)
        for c in self.COLS:
            self._sort_dirs[c] = None
            self.tree.heading(c, text=self._base_headings[c])
        self._sort_dirs[col] = new_dir
        arrow = " ▲" if new_dir else " ▼"
        self.tree.heading(col, text=self._base_headings[col] + arrow)
        self._sort_tree_by(col, ascending=new_dir)

    def _sort_tree_by(self, col: str, ascending: bool = True):
        items = [(self.tree.set(iid, col), iid) for iid in self.tree.get_children("")]
        def _key(x):
            v = x[0]
            try:
                return float(v)
            except Exception:
                return str(v).lower()
        items.sort(key=_key, reverse=not ascending)
        for pos, (_, iid) in enumerate(items):
            self.tree.move(iid, "", pos)

    # ----- displaycolumns normalization -----
    def _normalized_display_columns(self) -> List[str]:
        dc = self.tree.cget("displaycolumns")
        if isinstance(dc, str):
            return list(self.COLS) if dc == "#all" else [dc]
        cols = list(dc)
        if len(cols) == 1 and cols[0] == "#all":
            return list(self.COLS)
        return cols

    # ----- Column sizing -----
    def _fit_columns_to_window(self):
        vis_cols = self._normalized_display_columns()
        if not vis_cols:
            return
        self.update_idletasks()
        total_w = max(100, self.tree.winfo_width() - 4)
        per = max(10, int(total_w / len(vis_cols)))
        for c in vis_cols:
            self.tree.column(c, width=per, stretch=True)

    def _toggle_fit_to_window(self):
        self.fit_to_window = bool(self._fit_var.get())
        if self.fit_to_window:
            self._fit_columns_to_window()
        else:
            self._autosize_columns()

    def _autosize_columns(self):
        fnt = tkfont.nametofont("TkTextFont")
        PAD = 24
        MIN_W = 10
        MAX_W = 600
        vis_cols = self._normalized_display_columns()
        for col in vis_cols:
            header = self._base_headings.get(col, col)
            max_text = str(header)
            for iid in self.tree.get_children(""):
                val = self.tree.set(iid, col)
                if len(str(val)) > len(max_text):
                    max_text = str(val)
            width_px = fnt.measure(max_text) + PAD
            width_px = max(MIN_W, min(MAX_W, width_px))
            self.tree.column(col, width=width_px, stretch=False)

    def _on_configure(self, event):
        if not self.fit_to_window:
            return
        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(120, self._fit_columns_to_window)

    def _populate(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.row_times.clear()
        self.row_geos.clear()
        for r in self.rows:
            self.tree.insert("", "end", iid=str(r.file), values=(
                r.ch,
                str(r.file),
                r.start_time_local,
                r.end_time_local,
                r.duration_min,
                r.rate_Hz,
                r.easting_m,
                r.northing_m,
                r.rxc_stn,
                r.rxc_e,
                r.rxc_n,
                "",  # dist_to_rxc_m
            ))
            self.row_times[str(r.file)] = (r.start_utc, r.end_utc)
            self.row_geos[str(r.file)] = (r.zone, r.hem)
        self.after(50, self._fit_columns_to_window if self.fit_to_window else self._autosize_columns)
            # ----- Inline editors -----
    def _begin_edit(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        col_index = int(col_id.strip("#")) - 1
        col_name = self.COLS[col_index]

        if col_name == "RXC_STN":
            return self._edit_rxc_stn(row_id)

        if col_name not in ("easting_m", "northing_m"):
            return
        bbox = self.tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, w, h = bbox
        value = self.tree.set(row_id, col_name)
        entry = tk.Entry(self.tree)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, value)
        entry.focus_set()
        entry.bind("<Return>", lambda e: _commit())
        entry.bind("<Escape>", lambda e: entry.destroy())
        self._set_active_editor(entry)

        def _commit():
            new_val = entry.get().strip()
            entry.destroy()
            try:
                fval = float(new_val)
            except Exception:
                messagebox.showerror("Invalid number", f"'{new_val}' is not a valid float")
                return
            self.tree.set(row_id, col_name, fval)

    def _edit_rxc_stn(self, row_id: str):
        if not self.rxc_points:
            messagebox.showwarning("No RXC loaded", "Load an RXC file first (File → Load RXC…).")
            return
        stations = sorted({str(p.get("stn", "")).strip() for p in self.rxc_points if str(p.get("stn", "")).strip()})
        if not stations:
            messagebox.showwarning("No stations", "No station names found in the loaded RXC.")
            return

        col_index = self.COLS.index("RXC_STN") + 1
        col_id = f"#{col_index}"
        bbox = self.tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, w, h = bbox

        current = str(self.tree.set(row_id, "RXC_STN")).strip()
        preselect = current if current in stations else None
        if preselect is None:
            try:
                e = float(self.tree.set(row_id, "easting_m"))
                n = float(self.tree.set(row_id, "northing_m"))
                zone, hem = self.row_geos.get(row_id, (0, "N"))
                best = None
                best_d2 = None
                for p in self.rxc_points:
                    if p.get("zone") not in (0, zone):
                        continue
                    if str(p.get("hem", "N")).upper()[:1] != str(hem).upper()[:1]:
                        continue
                    de = float(p["e"]) - e
                    dn = float(p["n"]) - n
                    d2 = de * de + dn * dn
                    if (best_d2 is None) or (d2 < best_d2):
                        best_d2 = d2
                        best = p
                if best is not None:
                    preselect = str(best.get("stn", "")).strip() or None
            except Exception:
                pass

        combo = ttk.Combobox(self.tree, values=stations, state="readonly")
        combo.set(preselect if (preselect in stations) else (stations[0] if stations else ""))
        combo.place(x=x, y=y, width=w, height=h)
        combo.focus_set()

        def _commit(_e=None):
            val = combo.get().strip()
            self._apply_rxc_selection(row_id, val)
            combo.destroy()

        combo.bind("<<ComboboxSelected>>", _commit)
        combo.bind("<Return>", _commit)
        combo.bind("<Escape>", lambda e: combo.destroy())
        self._set_active_editor(combo)

    def _apply_rxc_selection(self, row_id: str, stn: str):
        if not stn:
            return
        cand = None
        for p in self.rxc_points:
            if str(p.get("stn", "")).strip() == stn:
                cand = p
                break
        if cand is None:
            return
        self.tree.set(row_id, "RXC_STN", stn)
        try:
            self.tree.set(row_id, "RXC_easting_m", round(float(cand.get("e")), 3))
            self.tree.set(row_id, "RXC_northing_m", round(float(cand.get("n")), 3))
        except Exception:
            self.tree.set(row_id, "RXC_easting_m", "")
            self.tree.set(row_id, "RXC_northing_m", "")
        # update distance
        try:
            e = float(self.tree.set(row_id, "easting_m"))
            n = float(self.tree.set(row_id, "northing_m"))
            de = float(cand.get("e")) - e
            dn = float(cand.get("n")) - n
            self.tree.set(row_id, "dist_to_rxc_m", round((de * de + dn * dn) ** 0.5, 2))
        except Exception:
            self.tree.set(row_id, "dist_to_rxc_m", "")

    # ----- RXC threshold/matching helpers -----
    def _get_threshold(self) -> float:
        return float(self.match_threshold_m)

    def set_threshold_dialog(self):
        val = simpledialog.askfloat(
            "Max dist (m)",
            "Match only if nearest RXC is within this distance (meters):",
            initialvalue=self.match_threshold_m,
            minvalue=0.0,
        )
        if val is not None:
            self.match_threshold_m = float(val)
            self.rematch_rxc()

    def rematch_rxc(self):
        self._match_rows_to_rxc()

    def _match_rows_to_rxc(self):
        if not self.rxc_points:
            return
        threshold = self._get_threshold()
        for iid in self.tree.get_children():
            vals = self.tree.item(iid, "values")
            try:
                e = float(vals[6])
                n = float(vals[7])
                zone, hem = self.row_geos.get(iid, (0, "N"))
                hem = str(hem).upper()[:1]
            except Exception:
                continue
            best = None
            best_d2 = None
            for p in self.rxc_points:
                if p.get("zone") not in (0, zone):
                    continue
                if p.get("hem", "N").upper()[:1] != hem:
                    continue
                de = p["e"] - e
                dn = p["n"] - n
                d2 = de * de + dn * dn
                if (best_d2 is None) or (d2 < best_d2):
                    best_d2 = d2
                    best = p
            if best is not None:
                dist = math.sqrt(best_d2) if best_d2 is not None else float("inf")
                if dist <= threshold:
                    self.tree.set(iid, "RXC_STN", best["stn"])
                    self.tree.set(iid, "RXC_easting_m", round(best["e"], 3))
                    self.tree.set(iid, "RXC_northing_m", round(best["n"], 3))
                    self.tree.set(iid, "dist_to_rxc_m", round(dist, 2))
                else:
                    self.tree.set(iid, "RXC_STN", "")
                    self.tree.set(iid, "RXC_easting_m", "")
                    self.tree.set(iid, "RXC_northing_m", "")
                    self.tree.set(iid, "dist_to_rxc_m", "")
            else:
                self.tree.set(iid, "RXC_STN", "")
                self.tree.set(iid, "RXC_easting_m", "")
                self.tree.set(iid, "RXC_northing_m", "")
                self.tree.set(iid, "dist_to_rxc_m", "")

    # ----- Commands -----
    def apply_changes(self):
        sel = list(self.tree.selection())
        if not sel:
            messagebox.showwarning("No rows selected", "Select one or more rows to apply changes to.")
            return
        rows = self._collect_rows_from_tree(sel)
        changed_files = 0
        total_fields = 0
        for r in rows:
            if not r.file.exists():
                continue
            try:
                lat_deg, lon_deg = utm_to_latlon(r.easting_m, r.northing_m, r.zone, r.hem)
            except Exception as e:
                messagebox.showerror("UTM conversion failed", f"{r.file.name}: {e}")
                continue
            lat_rad = math.radians(lat_deg)
            lon_rad = math.radians(lon_deg)
            try:
                n_changed = update_header_latlon_inplace(r.file, lat_rad, lon_rad)
                total_fields += n_changed
                if n_changed > 0:
                    changed_files += 1
            except Exception as e:
                messagebox.showerror("File update failed", f"{r.file.name}: {e}")
        self.refresh()
        if total_fields == 0:
            messagebox.showwarning("No fields changed", "No 'Lat'/'Lon' numeric fields matched in the header.")
        else:
            messagebox.showinfo("Done", f"Updated {changed_files} file(s), changed {total_fields} numeric field(s).")

    def apply_rxc_to_selected(self):
        """
        Use RXC coords (if selected per row) to update Z3D easting/northing and write headers (Lat/Lon in radians).
        """
        sel = list(self.tree.selection())
        if not sel:
            messagebox.showwarning("No rows selected", "Select one or more rows first.")
            return
        updated = 0
        wrote = 0
        for iid in sel:
            rxce = self.tree.set(iid, "RXC_easting_m")
            rxcn = self.tree.set(iid, "RXC_northing_m")
            if not rxce or not rxcn:
                continue
            try:
                e = float(rxce)
                n = float(rxcn)
            except Exception:
                continue
            # Update UI
            self.tree.set(iid, "easting_m", round(e, 3))
            self.tree.set(iid, "northing_m", round(n, 3))
            # Write header using row's implicit zone/hem
            zone, hem = self.row_geos.get(iid, (0, "N"))
            try:
                lat_deg, lon_deg = utm_to_latlon(e, n, zone, hem)
                lat_rad = math.radians(lat_deg)
                lon_rad = math.radians(lon_deg)
                file_path = Path(self.tree.item(iid, "values")[1])
                n_changed = update_header_latlon_inplace(file_path, lat_rad, lon_rad)
                if n_changed > 0:
                    wrote += 1
            except Exception:
                pass
            updated += 1
        self.refresh()
        if updated == 0:
            messagebox.showwarning("Nothing applied", "No selected rows had RXC coordinates.")
        else:
            messagebox.showinfo("Done", f"Applied RXC coords to {updated} row(s); wrote headers for {wrote} file(s).")

    def delete_selected(self):
        items = self.tree.selection()
        if not items:
            return
        files = [self.tree.item(i, "values")[1] for i in items]
        if not messagebox.askyesno("Confirm delete", f"Delete {len(files)} file(s)? This cannot be undone."):
            return
        deleted = 0
        errors = 0
        for fp in files:
            try:
                os.remove(fp)
                self.tree.delete(fp)
                if fp in self.row_times:
                    del self.row_times[fp]
                if fp in self.row_geos:
                    del self.row_geos[fp]
                deleted += 1
            except Exception:
                errors += 1
        if errors:
            messagebox.showwarning("Completed", f"Deleted {deleted} file(s); {errors} error(s).")
        else:
            messagebox.showinfo("Deleted", f"Deleted {deleted} file(s).")

    def delete_all_except_highlights(self):
        all_iids = list(self.tree.get_children())
        protected = [iid for iid in all_iids if "tx_overlap" in self.tree.item(iid, "tags")]
        deletable = [iid for iid in all_iids if iid not in protected]
        if not deletable:
            messagebox.showinfo("Nothing to delete", "No non-highlighted rows to delete.")
            return
        if not protected:
            if not messagebox.askyesno("Delete ALL files?", f"No rows are highlighted.\n\nThis will delete ALL {len(deletable)} files shown.\n\nProceed?"):
                return
        else:
            if not messagebox.askyesno("Confirm delete", f"Delete {len(deletable)} file(s) and keep {len(protected)} highlighted?"):
                return
        deleted = 0
        errors = 0
        for iid in deletable:
            try:
                fp = self.tree.item(iid, "values")[1]
                Path(fp).unlink(missing_ok=False)
                self.tree.delete(iid)
                if iid in self.row_times:
                    del self.row_times[iid]
                if iid in self.row_geos:
                    del self.row_geos[iid]
                deleted += 1
            except Exception:
                errors += 1
        if errors:
            messagebox.showwarning("Completed", f"Deleted {deleted - errors} file(s); {errors} error(s).")
        else:
            messagebox.showinfo("Deleted", f"Deleted {len(deletable)} file(s).")

    def export_csv(self):
        out = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not out:
            return
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.COLS)
            for iid in self.tree.get_children():
                w.writerow(self.tree.item(iid, "values"))
        messagebox.showinfo("Exported", f"Wrote {out}")

    def open_selected_external(self):
        items = self.tree.selection()
        if not items:
            messagebox.showwarning("No selection", "Select one or more rows to open.")
            return
        paths = [Path(self.tree.item(i, "values")[1]) for i in items]
        count, errs = _open_external(paths)
        if errs:
            messagebox.showwarning("Opened with some errors", f"Opened {count} file(s).\nErrors:\n" + "\n".join(errs[:10]))
        else:
            if count > 1:
                messagebox.showinfo("Opened", f"Opened {count} file(s) in your default editor.")

    def clear_highlights(self):
        for iid in self.tree.get_children():
            self.tree.item(iid, tags=())

    def _overlaps(self, a: Tuple[Optional[datetime], Optional[datetime]], b: Tuple[datetime, datetime]) -> bool:
        a0, a1 = a
        b0, b1 = b
        if a0 is None or a1 is None or b0 is None or b1 is None:
            return False
        return max(a0, b0) < min(a1, b1)

    def _compute_tx_intervals_in_folder(self, folder: Path) -> List[Tuple[datetime, datetime]]:
        tx_intervals: List[Tuple[datetime, datetime]] = []
        for p in _gather_z3d_recursive(folder):
            gpsweek, rate, lat_deg, lon_deg = parse_header_for_week_rate_latlon(p)
            sow0, sow1 = first_last_sow(p)
            if (gpsweek is None) or (sow0 is None) or (sow1 is None):
                continue
            start_utc = gps_to_utc(gpsweek * 604800 + sow0)
            end_utc = gps_to_utc(gpsweek * 604800 + sow1 + 1)
            tx_intervals.append((start_utc, end_utc))
        return tx_intervals

    def add_rx_folder(self):
        folder_path = filedialog.askdirectory(title="Select RX root folder (loads *.z3d recursively)")
        if not folder_path:
            return
        folder = Path(folder_path)
        new_files = _gather_z3d_recursive(folder)
        if not new_files:
            messagebox.showwarning("No Z3D files", "No .z3d files found under that folder.")
            return
        current_paths = [Path(self.tree.item(i, "values")[1]) for i in self.tree.get_children()]
        combined: List[Path] = []
        seen: set[str] = set()
        for p in current_paths + new_files:
            if p.exists() and str(p) not in seen:
                combined.append(p)
                seen.add(str(p))
        self.rows = load_rows(combined, self.tz)
        self.clear_highlights()
        self._populate()
        self._match_rows_to_rxc()

    def load_tx_folder(self):
        current_files = [Path(self.tree.item(i, "values")[1]) for i in self.tree.get_children()]
        init_dir = _suggest_tx_initialdir(current_files)
        folder_path = filedialog.askdirectory(title="Select TX folder (contains .z3d)", initialdir=str(init_dir))
        if not folder_path:
            return
        folder = Path(folder_path)
        tx_intervals = self._compute_tx_intervals_in_folder(folder)
        if not tx_intervals:
            messagebox.showwarning("No TX windows", "No valid TX .z3d intervals found in the selected folder.")
            return
        self.clear_highlights()
        for iid in self.tree.get_children():
            rx = self.row_times.get(iid, (None, None))
            if any(self._overlaps(rx, tx) for tx in tx_intervals):
                self.tree.item(iid, tags=("tx_overlap",))
        messagebox.showinfo("Done", "Highlighted RX files that overlap with any TX interval (green).")

    def load_rxc(self):
        path = filedialog.askopenfilename(title="Select RXC file", filetypes=[("RXC files", "*.rxc"), ("All files", "*.*")])
        if not path:
            return
        try:
            pts = _parse_rxc_generic(Path(path))
        except Exception as e:
            messagebox.showerror("RXC parse failed", f"{e}")
            return
        if not pts:
            messagebox.showwarning("No stations found", "Could not parse any RX.STN entries from the RXC file.")
            return
        self.rxc_points = pts
        self._match_rows_to_rxc()
        messagebox.showinfo("RXC loaded", f"Loaded {len(pts)} station(s) and assigned nearest matches.")

    def refresh(self):
        paths = [Path(self.tree.item(i, "values")[1]) for i in self.tree.get_children()]
        self.rows = load_rows(paths, self.tz)
        self.clear_highlights()
        self._populate()
        self._match_rows_to_rxc()

    # Utility to collect selected rows (from tree values)
    def _collect_rows_from_tree(self, only_iids: Optional[List[str]] = None) -> List[Z3DRow]:
        rows: List[Z3DRow] = []
        iids = only_iids if only_iids is not None else list(self.tree.get_children())
        for iid in iids:
            vals = self.tree.item(iid, "values")
            file = Path(vals[1])
            zone, hem = self.row_geos.get(str(file), (0, "N"))
            rows.append(Z3DRow(
                ch=int(vals[0]) if str(vals[0]).strip() else extract_channel(file),
                file=file,
                start_time_local=str(vals[2]),
                end_time_local=str(vals[3]),
                duration_min=float(vals[4]) if vals[4] != "" else 0.0,
                rate_Hz=float(vals[5]) if vals[5] != "" else 0.0,
                easting_m=float(vals[6]) if vals[6] != "" else 0.0,
                northing_m=float(vals[7]) if vals[7] != "" else 0.0,
                zone=zone,
                hem=hem,
                rxc_stn=str(vals[8]) if len(vals) > 8 else "",
                rxc_e=float(vals[9]) if len(vals) > 9 and str(vals[9]) != "" else 0.0,
                rxc_n=float(vals[10]) if len(vals) > 10 and str(vals[10]) != "" else 0.0,
            ))
        return rows

def main():
    ap = argparse.ArgumentParser(
        description="Z3D CSV-like editor with UTM, TX highlighting, RXC matching, RXC_STN dropdown, RXC→Z3D apply, and keyboard shortcuts"
    )
    ap.add_argument("--tz", default="UTC", help="Timezone for local time columns (IANA like 'Europe/Paris' or '+07:00').")
    ap.add_argument("files", nargs="+", help="Z3D files or directories (recursive).")
    args = ap.parse_args()

    files_in: List[Path] = []
    seen: set[str] = set()
    for item in args.files:
        path = Path(item)
        if path.is_dir():
            for fp in _gather_z3d_recursive(path):
                if str(fp) not in seen:
                    files_in.append(fp)
                    seen.add(str(fp))
        else:
            if path.exists() and str(path) not in seen:
                files_in.append(path)
                seen.add(str(path))
    if not files_in:
        print("No .z3d files found. If you are inside the RX folder, pass '.' or a subfolder.")
        return
    app = Z3DApp(files_in, tz=args.tz)
    app.mainloop()

if __name__ == "__main__":
    main()