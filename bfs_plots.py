
#!/usr/bin/env python3
r"""
bfs_plots.py - final plots of the BFS_TMR case (OpenFOAM v2312, simpleFoam).

Supports:
  - kOmegaSST
  - SpalartAllmaras

Reads the results directly from the case folder (also from Windows, via
\\wsl.localhost\Ubuntu\...) and saves the PNGs in the output folder.

Plots produced
  01_residui.png         initial residuals per iteration
  02_probe_U.png         Ux at the volume probes
  03_probe_p.png         p at the volume probes
  04_probe_cf.png        Cf at the probes on the downstream floor
  05_cf.png              Cf along the lower wall
  06_cp.png              Cp along the lower wall
  07_profili_U.png       U/Uref at x/H = -4, 1, 4, 6, 10
  08_profili_uv.png      -u'v'/Uref^2 at x/H = -4, 1, 4, 6, 10
  riepilogo.txt          validation metrics

For Spalart-Allmaras:
  - if the R field is available, it is read directly;
  - if R is not available, Rxy is reconstructed from nut and gradU.

Usage
  python bfs_plots.py
  python bfs_plots.py --model SpalartAllmaras
  python bfs_plots.py --model kOmegaSST
  python bfs_plots.py --case PATH --out folder --marks 2000 4000
  python bfs_plots.py --inspect-exp

Experimental data:
  cf.exp.dat, cp.expnew.dat, profiles.exp.dat
  in the "dati_sperimentali" folder next to the script.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------

DISTRO = "Ubuntu"
CASE_LINUX = "/home/nicola/OpenFOAM/nicola-v2312/run/BFS_TMR"

MODEL_DEFAULT = "SpalartAllmaras"
MODELS = ("kOmegaSST", "SpalartAllmaras")

H = 0.0127
U_REF_EXP = 44.2
X_CP_ZERO = 40.0
CF_UPSTREAM_REF = 0.0029
REATTACH_EXP = (6.26, 0.10)

STATIONS = [-4, 1, 4, 6, 10]

UPSTREAM_WALL = "wall_bottom_upstream"
DOWNSTREAM_WALL = "wall_bottom_downstream"

EXP_DIR = HERE / "dati_sperimentali"
EXP_CF_FILE = "cf.exp.dat"
EXP_CP_FILE = "cp.expnew.dat"
EXP_PROFILES_FILE = "profiles.exp.dat"

TMR_DIR = HERE / "dati_tmr"
TMR_MODEL = "sst"
TMR_LABEL = "TMR: CFL3D, SST"

CF_COLS = (0, 1)
CP_COLS = (0, 1)
PROFILE_COLS = {"y": 1, "u": 2, "uv": 6}
PROFILE_STATION_COL = None
PROFILE_ORDER = [-4, 1, 4, 6, 10]
EXP_UV_SIGN = "auto"

PROFILE_YLIM = (0.0, 3.0)
PROFILE_YLIM_UP = (1.0, 4.0)
CF_XLIM = (-5.0, 40.0)


def model_label(model):
    return "k-omega SST" if model == "kOmegaSST" else "Spalart-Allmaras"


# ----------------------------------------------------------------------------
# READING UTILITIES
# ----------------------------------------------------------------------------

def find_case_dir() -> Path:
    if os.name != "nt":
        return Path(CASE_LINUX)

    rel = CASE_LINUX.lstrip("/")
    for root in (rf"\\wsl.localhost\{DISTRO}", rf"\\wsl$\{DISTRO}"):
        case = Path(root) / rel
        if case.exists():
            return case

    raise FileNotFoundError(
        f"Caso non trovato in WSL ({DISTRO}): {CASE_LINUX}. Usa --case."
    )


def numeric_dirs(parent: Path):
    out = []
    if parent.is_dir():
        for d in parent.iterdir():
            if d.is_dir():
                try:
                    out.append((float(d.name), d))
                except ValueError:
                    pass
    return sorted(out)


def latest_time_dir(case: Path) -> Path:
    dirs = numeric_dirs(case)
    if not dirs:
        raise FileNotFoundError(f"Nessuna cartella tempo in {case}")
    return dirs[-1][1]


def _block(text: str, name: str, start: int = 0):
    m = re.search(rf"(?<!\w){re.escape(name)}\s*\{{", text[start:])
    if not m:
        return None

    i = start + m.end()
    depth, j = 1, i

    while depth and j < len(text):
        depth += (text[j] == "{") - (text[j] == "}")
        j += 1

    return text[i:j - 1]


def _parse_list(rest: str, ncomp: int) -> np.ndarray:
    """Read List<type> N (...) or List<type> N{value}."""
    mm = re.match(r"\s*List<\w+>\s*(\d+)\s*([({])", rest)

    if not mm:
        raise ValueError("formato della lista non riconosciuto")

    n = int(mm.group(1))

    if n == 0:
        return np.empty((0, ncomp))

    clean = lambda t: t.replace("(", " ").replace(")", " ").split()

    if mm.group(2) == "{":
        end = rest.index("}", mm.end())
        values = np.array(clean(rest[mm.end():end]), dtype=float)
        return np.tile(values, (n, 1))

    i = mm.end()
    depth, j = 1, i

    while depth:
        depth += (rest[j] == "(") - (rest[j] == ")")
        j += 1

    return np.array(clean(rest[i:j - 1]), dtype=float).reshape(n, ncomp)


def read_patch_field(path: Path, patch: str, ncomp: int) -> np.ndarray:
    """Values of a patch from an ASCII OpenFOAM field."""
    text = path.read_text(encoding="utf-8", errors="replace")
    k = text.find("boundaryField")
    body = _block(text, patch, max(k, 0))

    if body is None:
        raise KeyError(f"patch '{patch}' non trovata in {path.name}")

    m = re.search(r"\bvalue\s+(uniform|nonuniform)\b", body)

    if not m:
        raise KeyError(
            f"'{path.name}': la patch '{patch}' non ha 'value'. "
            "Per zeroGradient usa i valori di cella."
        )

    if m.group(1) == "uniform":
        raise ValueError(
            f"'{path.name}': la patch '{patch}' ha valore uniforme"
        )

    return _parse_list(body[m.end():], ncomp)


def read_internal(path: Path, ncomp: int) -> np.ndarray:
    """Read the nonuniform internalField of an ASCII OpenFOAM field."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"internalField\s+(uniform|nonuniform)\b", text)

    if not m or m.group(1) == "uniform":
        raise ValueError(
            f"'{path.name}': internalField assente o uniforme"
        )

    return _parse_list(text[m.end():], ncomp)


def read_xy(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", ndmin=2)


# ----------------------------------------------------------------------------
# RECONSTRUCTION OF THE TURBULENT STRESS TENSOR FOR SA
# ----------------------------------------------------------------------------

def sa_reynolds_stress_xy(cells):
    """
    Reconstruct Rxy with the turbulent-viscosity hypothesis:

        Rxy = -nut * (dUx/dy + dUy/dx)

    The vertical derivative is computed inside each column.
    The horizontal derivative is computed between adjacent columns,
    interpolating the values at the cell centres.

    Returns Rxy, i.e. the xy component of the tensor R.
    The -u'v' profile is then obtained as -Rxy.
    """

    ux = cells["ux"]
    cols = cells["cols"]

    U = cells["U"]
    nut = cells["nut"]
    y = cells["y"]

    Rxy = np.full(len(y), np.nan)

    for i, col in enumerate(cols):
        yc = y[col]
        Uxc = U[col, 0]

        # dUx/dy in the current column
        dUxdY = np.gradient(Uxc, yc)

        # dUy/dx in the current column
        if i == 0:
            other = cols[i + 1]
            x0, x1 = ux[i], ux[i + 1]
            Uy0 = U[col, 1]
            Uy1 = np.interp(yc, y[other], U[other, 1])
            dUydX = (Uy1 - Uy0) / (x1 - x0)

        elif i == len(cols) - 1:
            other = cols[i - 1]
            x0, x1 = ux[i - 1], ux[i]
            Uy0 = np.interp(yc, y[other], U[other, 1])
            Uy1 = U[col, 1]
            dUydX = (Uy1 - Uy0) / (x1 - x0)

        else:
            left = cols[i - 1]
            right = cols[i + 1]

            UyL = np.interp(yc, y[left], U[left, 1])
            UyR = np.interp(yc, y[right], U[right, 1])

            dUydX = (UyR - UyL) / (ux[i + 1] - ux[i - 1])

        Rxy[col] = -nut[col] * (dUxdY + dUydX)

    return Rxy


# ----------------------------------------------------------------------------
# CELL FIELDS
# ----------------------------------------------------------------------------

def load_cells(case: Path, model: str):
    tdir = latest_time_dir(case)

    C = read_internal(tdir / "C", 3)

    cells = dict(
        x=C[:, 0],
        y=C[:, 1],
        U=read_internal(tdir / "U", 3),
        nut=read_internal(tdir / "nut", 1)[:, 0],
        p=read_internal(tdir / "p", 1)[:, 0],
        time=tdir.name,
        tdir=tdir,
    )

    # Grouping of the cells into vertical columns.
    # The mesh is structured: cells in the same column have similar x/H.
    xh = cells["x"] / H
    order_x = np.argsort(xh)

    col_id = np.empty(len(xh), dtype=int)
    col_id[order_x] = np.concatenate(
        ([0], np.cumsum(np.diff(xh[order_x]) > 1e-3))
    )

    order = np.lexsort((cells["y"], col_id))
    _, starts = np.unique(col_id[order], return_index=True)
    ends = list(starts[1:]) + [len(order)]

    cells["ux"] = np.array([
        xh[order[s:e]].mean()
        for s, e in zip(starts, ends)
    ])

    cells["cols"] = [
        order[s:e]
        for s, e in zip(starts, ends)
    ]

    d = np.diff(cells["ux"])

    print(
        "  colonne:", len(cells["ux"]),
        "| min dx/H:", d.min(),
        "| celle per colonna:", sorted({len(c) for c in cells["cols"]})
    )

    # Read R, if present.
    R_paths = [
        tdir / "turbulenceProperties:R",
        tdir / "R",
    ]

    Rxy = None

    for path in R_paths:
        if not path.is_file():
            continue

        try:
            R = read_internal(path, 6)
            Rxy = R[:, 1]
            print(f"  Campo R letto da: {path.name}")
            break
        except Exception as exc:
            print(f"  ATTENZIONE: impossibile leggere {path.name}: {exc}")

    # If R is not available, for SA we reconstruct Rxy from nut and gradU.
    if Rxy is None and model == "SpalartAllmaras":
        print(
            "  Campo R non disponibile: ricostruisco Rxy da nut e gradU "
            "(modello a viscosita' turbolenta)."
        )
        Rxy = sa_reynolds_stress_xy(cells)

    elif Rxy is None:
        print(
            "  ATTENZIONE: campo R non leggibile. "
            "Esegui simpleFoam -postProcess -latestTime -func R."
        )

    cells["Rxy"] = Rxy

    return cells


def profile_at(cells, xh: float):
    """
    Vertical profile at x/H = xh on the native cell centres.

    -u'v' = -Rxy.
    For SA, Rxy can be reconstructed from nut and gradU if R is not available.
    """

    if cells.get("Rxy") is None:
        raise ValueError(
            "Campo Rxy mancante: per SST esegui postProcess -func R; "
            "per SA verifica la disponibilita' di nut e U."
        )

    ux, cols = cells["ux"], cells["cols"]

    i = int(np.searchsorted(ux, xh))

    if i == 0 or i >= len(ux):
        raise ValueError(f"x/H={xh} fuori dalle colonne di celle")

    L, R = cols[i - 1], cols[i]

    y = cells["y"]
    yL = y[L]

    w = (xh - ux[i - 1]) / (ux[i] - ux[i - 1])

    def right_on_left(vals):
        return np.interp(yL, y[R], vals[R])

    u = (
        (1.0 - w) * cells["U"][L, 0]
        + w * right_on_left(cells["U"][:, 0])
    )

    uv = -(
        (1.0 - w) * cells["Rxy"][L]
        + w * right_on_left(cells["Rxy"])
    )

    return dict(
        y=yL / H,
        y_m=yL,
        u=u,
        minus_uv=uv,
    )


def wall_pressure(cells):
    """p of the first cell above the floor (zeroGradient)."""
    ux, cols = cells["ux"], cells["cols"]
    return ux, np.array([cells["p"][c[0]] for c in cols])


# ----------------------------------------------------------------------------
# WALL DATA: Cf, Cp
# ----------------------------------------------------------------------------

def load_wall(case: Path, u_ref: float, cells):
    tdir = cells["tdir"]

    xs, taus, which = [], [], []

    for patch in (UPSTREAM_WALL, DOWNSTREAM_WALL):
        C = read_patch_field(tdir / "C", patch, 3)
        tau = read_patch_field(tdir / "wallShearStress", patch, 3)

        xs.append(C[:, 0] / H)
        taus.append(tau[:, 0])
        which.append(np.full(len(C), patch == DOWNSTREAM_WALL))

    x = np.concatenate(xs)
    tau = np.concatenate(taus)
    down = np.concatenate(which)

    order = np.argsort(x)
    x, tau, down = x[order], tau[order], down[order]

    # Upstream, in the attached boundary layer, Cf must be positive.
    up_mask = (x > -20) & (x < -2)
    sign = 1.0 if np.median(tau[up_mask]) >= 0 else -1.0

    q = 0.5 * u_ref ** 2
    cf = sign * tau / q

    xcp, pw = wall_pressure(cells)
    cp = (pw - np.interp(X_CP_ZERO, xcp, pw)) / q

    return dict(
        x=x,
        cf=cf,
        down=down,
        sign=sign,
        time=cells["time"],
        xcp=xcp,
        cp=cp,
    )


def reattachment(x, cf, down):
    """x/H where Cf goes from negative to positive."""
    m = down & (x > 0.5) & (x < 20)
    xx, cc = x[m], cf[m]

    idx = [
        i for i in range(len(cc) - 1)
        if cc[i] < 0 <= cc[i + 1]
    ]

    if not idx:
        return None

    i = idx[-1]

    return float(
        xx[i] - cc[i] * (xx[i + 1] - xx[i]) / (cc[i + 1] - cc[i])
    )


# ----------------------------------------------------------------------------
# EXPERIMENTAL DATA
# ----------------------------------------------------------------------------

def _try_floats(s: str):
    toks = s.replace(",", " ").split()

    if not toks:
        return None

    try:
        return [
            float(t.replace("D", "E").replace("d", "e"))
            for t in toks
        ]
    except ValueError:
        return None


def _to_array(rows):
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], dtype=float)


def read_numeric_blocks(path: Path):
    """Blocks of numeric rows separated by text rows."""
    blocks, cur, header = [], [], ""

    for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():

        s = line.strip()
        nums = _try_floats(s)

        if nums is not None and len(nums) >= 2:
            cur.append(nums)
        else:
            if cur:
                blocks.append((header, _to_array(cur)))
                cur = []

            if s:
                header = s

    if cur:
        blocks.append((header, _to_array(cur)))

    return blocks


def inspect_exp():
    for name in (EXP_CF_FILE, EXP_CP_FILE, EXP_PROFILES_FILE):
        path = EXP_DIR / name

        print("=" * 70)
        print(path)

        if not path.is_file():
            print("  NON TROVATO")
            continue

        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()

        print("  prime righe:")

        for line in lines[:6]:
            print("   |", line)

        for i, (hdr, arr) in enumerate(read_numeric_blocks(path)):
            print(
                f"  blocco {i}: intestazione={hdr!r} "
                f"righe={arr.shape[0]} colonne={arr.shape[1]}"
            )
            print(f"            prima riga={arr[0]}")


def load_exp_table(name: str, cols):
    path = EXP_DIR / name

    if not path.is_file():
        return None

    blocks = read_numeric_blocks(path)

    if not blocks:
        return None

    arr = max(blocks, key=lambda b: b[1].shape[0])[1]

    return arr[:, cols[0]], arr[:, cols[1]]


def load_exp_profiles():
    path = EXP_DIR / EXP_PROFILES_FILE

    if not path.is_file():
        return {}

    blocks = read_numeric_blocks(path)
    res = {}

    if PROFILE_STATION_COL is not None:
        data = np.vstack([b for _, b in blocks])
        col = np.round(data[:, PROFILE_STATION_COL], 6)

        for xs in np.unique(col):
            res[float(xs)] = data[col == xs]

        return res

    for idx, (hdr, arr) in enumerate(blocks):
        m = re.search(
            r"x\s*/\s*h\s*=?\s*(-?\d+(?:\.\d+)?)",
            hdr,
            re.I
        )

        if m:
            res[float(m.group(1))] = arr
        elif len(blocks) == len(PROFILE_ORDER):
            res[float(PROFILE_ORDER[idx])] = arr

    if not res:
        print(
            "  ATTENZIONE: non riesco ad assegnare i blocchi di "
            "profiles.exp.dat alle stazioni x/H; usa --inspect-exp."
        )

    return res


# ----------------------------------------------------------------------------
# TMR DATA
# ----------------------------------------------------------------------------

def find_tmr_file(key: str):
    if not TMR_DIR.is_dir():
        return None

    hits = sorted(TMR_DIR.glob(f"*_{key}_{TMR_MODEL}.dat"))

    return hits[0] if hits else None


def _read_tmr_zones(path: Path):
    zones = {}

    for hdr, arr in read_numeric_blocks(path):
        m = re.search(
            r"x\s*/\s*h\s*=\s*(?:approx\w*\s*)?(-?\d+(?:\.\d+)?)",
            hdr,
            re.I
        )

        if m:
            zones[float(m.group(1))] = arr

    return zones


def load_tmr():
    tmr = {}

    for key in ("cf", "cp", "vel", "uv"):
        path = find_tmr_file(key)

        if path is None:
            print(
                f"  TMR: file '*_{key}_{TMR_MODEL}.dat' "
                f"non trovato in {TMR_DIR}"
            )
            continue

        if key in ("cf", "cp"):
            blocks = read_numeric_blocks(path)
            arr = max(blocks, key=lambda b: b[1].shape[0])[1]
            arr = arr[np.argsort(arr[:, 0], kind="stable")]
            tmr[key] = arr[:, :2]
        else:
            tmr[key] = _read_tmr_zones(path)

        print(f"  TMR: {path.name} letto")

    return tmr


def tmr_reattachment(tmr):
    if "cf" not in tmr:
        return None

    x, cf = tmr["cf"][:, 0], tmr["cf"][:, 1]

    return reattachment(
        x,
        cf,
        np.ones(len(x), dtype=bool)
    )


def add_tmr_cf(ax, tmr):
    if tmr and "cf" in tmr:
        x, cf = tmr["cf"][:, 0], tmr["cf"][:, 1]
        keep = np.abs(x) > 0.05
        ax.plot(
            x[keep], cf[keep], "--",
            color="tab:red", lw=1.4, label=TMR_LABEL
        )


def add_tmr_cp(ax, tmr):
    if tmr and "cp" in tmr:
        x, cp = tmr["cp"][:, 0], tmr["cp"][:, 1]
        ax.plot(
            x,
            cp - np.interp(X_CP_ZERO, x, cp),
            "--",
            color="tab:red",
            lw=1.4,
            label=TMR_LABEL
        )


def add_tmr_profile(ax, tmr, xh, key):
    if not tmr:
        return

    zones = tmr.get("vel" if key == "u" else "uv", {})
    z = zones.get(float(xh))

    if z is None:
        return

    val = z[:, 2].copy()

    if key == "uv":
        if np.nanmax(np.abs(val)) > 0.05:
            val = val / 1000.0

        val = -val

    ax.plot(
        val, z[:, 1], "--",
        color="tab:red", lw=1.4, label=TMR_LABEL
    )


# ----------------------------------------------------------------------------
# CFD / TMR / EXPERIMENT METRICS
# ----------------------------------------------------------------------------

def _extremum(x, v, lo, hi, kind):
    x, v = np.asarray(x, float), np.asarray(v, float)

    m = (x > lo) & (x < hi) & np.isfinite(v)

    if not m.any():
        return None

    i = int(
        np.argmin(v[m]) if kind == "min"
        else np.argmax(v[m])
    )

    return float(v[m][i]), float(x[m][i])


def _exp_cf():
    t = load_exp_table(EXP_CF_FILE, CF_COLS)

    if t is None:
        return None

    xe, ce = np.asarray(t[0], float), np.asarray(t[1], float)

    if np.nanmax(np.abs(ce)) > 0.05:
        ce = ce * 1e-3

    return xe, ce


def _exp_uv_peak(xh):
    e = load_exp_profiles().get(float(xh))

    if e is None or PROFILE_COLS["uv"] >= e.shape[1]:
        return None

    ev = e[:, PROFILE_COLS["uv"]] / 1000.0

    s = EXP_UV_SIGN

    if s == "auto":
        s = (
            -1.0
            if abs(np.nanmin(ev)) > abs(np.nanmax(ev))
            else 1.0
        )

    ev = ev * s

    i = int(np.nanargmax(ev))

    return float(ev[i]), float(e[i, PROFILE_COLS["y"]])


def collect_metrics(wall, profiles, tmr, u_ref):
    rows = []

    def add(label, kind, c, t, e):
        rows.append((label, kind, c, t, e))

    x, cf, down = wall["x"], wall["cf"], wall["down"]

    xe_cf = _exp_cf()
    tc = tmr.get("cf") if tmr else None

    add(
        "Riattacco x/H",
        "pos",
        reattachment(x, cf, down),
        tmr_reattachment(tmr) if tmr else None,
        REATTACH_EXP[0]
    )

    add(
        "Cf a x/H = -4",
        "val",
        float(np.interp(-4.0, x, cf)),
        float(np.interp(-4.0, tc[:, 0], tc[:, 1]))
        if tc is not None else None,
        float(np.interp(-4.0, *xe_cf))
        if xe_cf is not None else None
    )

    c = _extremum(x[down], cf[down], 0.5, 20, "min")
    t = (
        _extremum(tc[:, 0], tc[:, 1], 0.5, 20, "min")
        if tc is not None else None
    )
    e = (
        _extremum(*xe_cf, 0.5, 20, "min")
        if xe_cf is not None else None
    )

    for label, kind, idx in (
        ("Cf minimo", "val", 0),
        ("x/H del Cf minimo", "pos", 1)
    ):
        add(
            label,
            kind,
            *(None if r is None else r[idx] for r in (c, t, e))
        )

    xcp, cp = wall["xcp"], wall["cp"]
    tp = tmr.get("cp") if tmr else None

    if tp is not None:
        tp = np.column_stack([
            tp[:, 0],
            tp[:, 1] - np.interp(X_CP_ZERO, tp[:, 0], tp[:, 1])
        ])

    ep = load_exp_table(EXP_CP_FILE, CP_COLS)

    for name, lo, hi, kind in (
        ("Cp minimo", 0, 8, "min"),
        ("Cp massimo", 5, 20, "max")
    ):
        c = _extremum(xcp, cp, lo, hi, kind)

        t = (
            _extremum(tp[:, 0], tp[:, 1], lo, hi, kind)
            if tp is not None else None
        )

        e = (
            _extremum(ep[0], ep[1], lo, hi, kind)
            if ep is not None else None
        )

        for label, k, idx in (
            (name, "val", 0),
            (f"x/H del {name}", "pos", 1)
        ):
            add(
                label,
                k,
                *(None if r is None else r[idx] for r in (c, t, e))
            )

    for xh in STATIONS:
        if xh < 0:
            continue

        p = profiles.get(xh)

        c = (
            _extremum(
                p["y"],
                p["minus_uv"] / u_ref ** 2,
                -1,
                1e9,
                "max"
            )
            if p else None
        )

        z = tmr.get("uv", {}).get(float(xh)) if tmr else None
        t = None

        if z is not None:
            v = z[:, 2] / (
                1000.0
                if np.nanmax(np.abs(z[:, 2])) > 0.05
                else 1.0
            )

            t = _extremum(z[:, 1], -v, -1, 1e9, "max")

        e = _exp_uv_peak(xh)

        for label, kind, idx in (
            (f"Picco -u'v'/Uref^2 a x/H = {xh} (x1e3)", "val", 0),
            (f"y/H del picco a x/H = {xh}", "pos", 1)
        ):
            scale = 1e3 if idx == 0 else 1.0

            add(
                label,
                kind,
                *(
                    None if r is None else r[idx] * scale
                    for r in (c, t, e)
                )
            )

    return rows


def format_metrics(rows):
    def f(v):
        return "        --" if v is None else f"{v:10.4g}"

    def d(c, o, kind):
        if c is None or o is None:
            return "      --"

        if kind == "pos":
            return f"{c - o:+8.2f} "

        return (
            f"{100.0 * (c - o) / abs(o):+7.1f}% "
            if o != 0 else "      --"
        )

    head = (
        f"{'':44s}{'CFD':>10s}{'TMR':>10s}{'Esp.':>10s}"
        f"   {'vs TMR':>8s} {'vs Esp.':>8s}"
    )

    out = [head, "-" * len(head)]

    for label, kind, c, t, e in rows:
        out.append(
            f"{label:44s}{f(c)}{f(t)}{f(e)}   "
            f"{d(c, t, kind)} {d(c, e, kind)}"
        )

    out.append(
        "(scarti: in % per i valori, assoluti per le posizioni; "
        "esperimento = punti discreti dei file)"
    )

    return "\n".join(out)


# ----------------------------------------------------------------------------
# MONITORS: RESIDUALS AND PROBES
# ----------------------------------------------------------------------------

def read_solverinfo(case: Path):
    base = case / "postProcessing" / "solverInfo"

    rows_by_time, header = {}, None

    for _, d in numeric_dirs(base):
        f = d / "solverInfo.dat"

        if not f.is_file():
            continue

        for line in f.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():

            if line.startswith("#"):
                toks = line.lstrip("#").split()

                if toks and toks[0] == "Time":
                    header = toks

                continue

            toks = line.split()

            if header and toks:
                try:
                    rows_by_time[float(toks[0])] = toks
                except ValueError:
                    pass

    if not rows_by_time or header is None:
        raise FileNotFoundError(
            f"solverInfo.dat non trovato/vuoto in {base}"
        )

    times = sorted(rows_by_time)
    cols = {}

    for j, name in enumerate(header):
        vals = []

        for t in times:
            r = rows_by_time[t]

            try:
                vals.append(float(r[j]))
            except (IndexError, ValueError):
                vals.append(np.nan)

        cols[name] = np.array(vals)

    return np.array(times), cols


def read_probes(case: Path, folder: str, field: str):
    base = case / "postProcessing" / folder

    data_by_time, locs, ncomp = {}, [], None

    for _, d in numeric_dirs(base):
        f = d / field

        if not f.is_file():
            continue

        locs_here = []

        for line in f.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():

            if line.startswith("#"):
                m = re.match(
                    r"#\s*Probe\s+\d+\s*\(([^)]*)\)",
                    line
                )

                if m:
                    locs_here.append([
                        float(v) for v in m.group(1).split()
                    ])

                continue

            if not line.strip():
                continue

            vals = [
                float(v)
                for v in line.replace("(", " ").replace(")", " ").split()
            ]

            n = len(locs_here)

            if n == 0:
                continue

            ncomp = (len(vals) - 1) // n
            data_by_time[vals[0]] = np.array(
                vals[1:]
            ).reshape(n, ncomp)

        if locs_here:
            locs = locs_here

    if not data_by_time:
        raise FileNotFoundError(
            f"nessun dato in {base}/*/{field}"
        )

    times = np.array(sorted(data_by_time))
    data = np.array([data_by_time[t] for t in times])

    return times, data, np.array(locs)


# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------

def _marks(ax, marks):
    for m in marks or []:
        ax.axvline(m, color="gray", ls=":", lw=1)


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"  salvato: {name}")


def plot_residuals(case, out, marks, model):
    t, cols = read_solverinfo(case)

    names = [
        c for c in cols
        if c.endswith("_initial") and not c.startswith("Uz")
    ]

    if model == "SpalartAllmaras":
        order = {
            "Ux": 0,
            "Uy": 1,
            "p": 2,
            "nuTilda": 3,
        }
    else:
        order = {
            "Ux": 0,
            "Uy": 1,
            "p": 2,
            "k": 3,
            "omega": 4,
        }

    names.sort(
        key=lambda c: order.get(c.split("_")[0], 9)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for c in names:
        v = cols[c]

        if np.all(~(v > 0)) or np.all(np.isnan(v)):
            continue

        ax.semilogy(
            t,
            v,
            label=c.replace("_initial", "")
        )

    _marks(ax, marks)

    ax.set_xlabel("iterazione")
    ax.set_ylabel("residuo iniziale")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=3)
    ax.set_title(f"Residui - {model_label(model)}")

    save(fig, out, "01_residui.png")


def _group_probes(locs):
    groups = {}

    for i, (x, y, _) in enumerate(locs):
        groups.setdefault(round(x / H), []).append(
            (round(y / H, 2), i)
        )

    return groups


def plot_probes(case, out, marks, field, fname, title, comp):
    t, data, locs = read_probes(
        case, "probesFlow", field
    )

    groups = _group_probes(locs)
    keys = sorted(groups)

    fig, axs = plt.subplots(
        1,
        len(keys),
        figsize=(3.6 * len(keys), 4),
        sharex=True
    )

    axs = np.atleast_1d(axs)

    for ax, k in zip(axs, keys):
        for yh, i in groups[k]:
            ax.plot(
                t,
                data[:, i, comp],
                label=f"y/H = {yh:g}"
            )

        _marks(ax, marks)

        ax.set_title(f"x/H = {k}")
        ax.set_xlabel("iterazione")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axs[0].set_ylabel(title)

    save(fig, out, fname)


def plot_probe_cf(case, out, marks, sign, u_ref):
    t, data, locs = read_probes(
        case, "probesWallShear", "wallShearStress"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (x, _, _) in enumerate(locs):
        ax.plot(
            t,
            sign * data[:, i, 0] / (0.5 * u_ref ** 2),
            label=f"x/H = {x / H:g}"
        )

    _marks(ax, marks)

    ax.set_xlabel("iterazione")
    ax.set_ylabel("Cf")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    save(fig, out, "04_probe_cf.png")


def plot_cf(wall, out, model, tmr=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        wall["x"],
        wall["cf"],
        label=f"CFD ({model_label(model)})"
    )

    exp = load_exp_table(EXP_CF_FILE, CF_COLS)

    if exp is not None:
        xe, ce = exp

        if np.nanmax(np.abs(ce)) > 0.05:
            print(
                "  Cf sperimentale con valori > 0.05: "
                "lo moltiplico per 1e-3."
            )
            ce = ce * 1e-3

        ax.plot(
            xe, ce, "ko", ms=4,
            label="Esperimento (Driver & Seegmiller)"
        )

    add_tmr_cf(ax, tmr)

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlim(*CF_XLIM)
    ax.set_xlabel("x/H")
    ax.set_ylabel("Cf")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save(fig, out, "05_cf.png")


def plot_cp(wall, out, model, tmr=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        wall["xcp"],
        wall["cp"],
        label=f"CFD ({model_label(model)}), Cp(x/H=40)=0"
    )

    exp = load_exp_table(EXP_CP_FILE, CP_COLS)

    if exp is not None:
        ax.plot(
            exp[0], exp[1], "ko", ms=4,
            label="Esperimento (shifted)"
        )

    add_tmr_cp(ax, tmr)

    ax.set_xlim(*CF_XLIM)
    ax.set_xlabel("x/H")
    ax.set_ylabel("Cp")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save(fig, out, "06_cp.png")


def plot_profiles(profiles, u_ref, out, tmr=None):
    exp = load_exp_profiles()
    exp_sign = EXP_UV_SIGN

    for key, fname, xlabel in (
        ("u", "07_profili_U.png", r"$U/U_{ref}$"),
        ("uv", "08_profili_uv.png", r"$-\overline{u'v'}/U_{ref}^2$")
    ):
        fig, axs = plt.subplots(
            1,
            len(STATIONS),
            figsize=(3.4 * len(STATIONS), 5)
        )

        for ax, xh in zip(np.atleast_1d(axs), STATIONS):
            p = profiles.get(xh)

            if p is not None:
                val = (
                    p["u"] / u_ref
                    if key == "u"
                    else p["minus_uv"] / u_ref ** 2
                )

                ax.plot(val, p["y"], label="CFD")

            e = exp.get(float(xh))

            if e is not None:
                ci = PROFILE_COLS[key]

                if ci < e.shape[1]:
                    ev = e[:, ci].copy()

                    if key == "uv":
                        ev = ev / 1000.0

                        s = exp_sign

                        if s == "auto":
                            s = (
                                -1.0
                                if abs(np.nanmin(ev)) > abs(np.nanmax(ev))
                                else 1.0
                            )

                        ev = ev * s

                    ax.plot(
                        ev,
                        e[:, PROFILE_COLS["y"]],
                        "ko",
                        ms=3,
                        label="Esperimento"
                    )

            add_tmr_profile(ax, tmr, xh, key)

            ax.set_ylim(
                *(PROFILE_YLIM_UP if xh < 0 else PROFILE_YLIM)
            )

            ax.set_title(f"x/H = {xh}")
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)

        np.atleast_1d(axs)[0].set_ylabel("y/H")
        np.atleast_1d(axs)[0].legend(fontsize=8)

        save(fig, out, fname)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def run_section(name, func, *a, **k):
    print(f"[{name}]")

    try:
        return func(*a, **k)

    except Exception as exc:
        print(
            f"  SALTATO: {type(exc).__name__}: {exc}"
        )

        if os.environ.get("BFS_DEBUG"):
            traceback.print_exc()

        return None


def main():
    ap = argparse.ArgumentParser(
        description="Plot finali del caso BFS_TMR"
    )

    ap.add_argument(
        "--case",
        default=None,
        help="cartella del caso (default: caso WSL)"
    )

    ap.add_argument(
        "--out",
        default=str(HERE / "risultati_bfs"),
        help="cartella dei PNG"
    )

    ap.add_argument(
        "--marks",
        type=float,
        nargs="*",
        default=[],
        help="iterazioni in cui disegnare una linea verticale"
    )

    ap.add_argument(
        "--norm",
        choices=("exp", "cfd"),
        default="exp",
        help="Uref: sperimentale oppure CFD"
    )

    ap.add_argument(
        "--model",
        choices=MODELS,
        default=MODEL_DEFAULT,
        help="modello di turbolenza"
    )

    ap.add_argument(
        "--inspect-exp",
        action="store_true",
        help="mostra la struttura dei file sperimentali ed esce"
    )

    ap.add_argument(
        "--no-tmr",
        action="store_true",
        help="non sovrappone i dati TMR"
    )

    args = ap.parse_args()

    if args.inspect_exp:
        inspect_exp()
        return 0

    model = args.model

    case = Path(args.case) if args.case else find_case_dir()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"Modello: {model}\n"
        f"Caso:    {case}\n"
        f"Output:  {out}\n"
    )

    # ------------------------------------------------------------------------
    # Cell fields and CFD profiles
    # ------------------------------------------------------------------------

    cells = run_section(
        "lettura campi (C, U, nut, p, R)",
        load_cells,
        case,
        model
    )

    profiles = {}

    if cells:
        for xh in STATIONS:
            r = run_section(
                f"profilo x/H={xh}",
                profile_at,
                cells,
                xh
            )

            if r is not None:
                profiles[xh] = r

    # ------------------------------------------------------------------------
    # Uref
    # ------------------------------------------------------------------------

    u_center = None

    if -4 in profiles:
        p = profiles[-4]
        u_center = float(
            np.interp(5.0 * H, p["y_m"], p["u"])
        )

    u_ref = (
        U_REF_EXP
        if args.norm == "exp" or u_center is None
        else u_center
    )

    # ------------------------------------------------------------------------
    # Wall
    # ------------------------------------------------------------------------

    wall = (
        run_section(
            "parete: Cf, Cp",
            load_wall,
            case,
            u_ref,
            cells
        )
        if cells else None
    )

    sign = wall["sign"] if wall else 1.0

    # ------------------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------------------

    run_section(
        "residui",
        plot_residuals,
        case,
        out,
        args.marks,
        model
    )

    run_section(
        "probe U",
        plot_probes,
        case,
        out,
        args.marks,
        "U",
        "02_probe_U.png",
        "Ux [m/s]",
        0
    )

    run_section(
        "probe p",
        plot_probes,
        case,
        out,
        args.marks,
        "p",
        "03_probe_p.png",
        "p [m2/s2]",
        0
    )

    run_section(
        "probe Cf",
        plot_probe_cf,
        case,
        out,
        args.marks,
        sign,
        u_ref
    )

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------

    tmr = (
        {}
        if args.no_tmr
        else (run_section("dati TMR", load_tmr) or {})
    )

    if wall:
        run_section(
            "Cf",
            plot_cf,
            wall,
            out,
            model,
            tmr
        )

        run_section(
            "Cp",
            plot_cp,
            wall,
            out,
            model,
            tmr
        )

    if profiles:
        run_section(
            "profili",
            plot_profiles,
            profiles,
            u_ref,
            out,
            tmr
        )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    lines = [
        f"Caso: {case}",
        f"Modello: {model}"
    ]

    xr_tmr = None

    if wall:
        xr = reattachment(
            wall["x"],
            wall["cf"],
            wall["down"]
        )

        lines.append(
            f"Tempo (iterazione) dei risultati: {wall['time']}"
        )

        lines.append(
            f"Segno applicato a wallShearStress: {wall['sign']:+.0f}"
        )

        if xr is not None:
            lines.append(
                f"Riattacco CFD: x/H = {xr:.2f} "
                f"(sperimentale {REATTACH_EXP[0]} +- {REATTACH_EXP[1]})"
            )
        else:
            lines.append(
                "Riattacco CFD: nessun passaggio da Cf<0 a Cf>0 "
                "trovato tra x/H=0.5 e 20"
            )

        cf_m4 = float(
            np.interp(-4.0, wall["x"], wall["cf"])
        )

        lines.append(
            f"Cf a x/H=-4: {cf_m4:.5f} "
            f"(riferimento usato in bfs_mesh.py: {CF_UPSTREAM_REF})"
        )

        xr_tmr = tmr_reattachment(tmr)

    if xr_tmr is not None:
        lines.append(
            f"Riattacco {TMR_LABEL}: x/H = {xr_tmr:.2f}"
        )

    if u_center is not None:
        lines.append(
            f"U centro canale a x/H=-4: {u_center:.3f} m/s "
            f"(target sperimentale {U_REF_EXP}; "
            f"rapporto {u_center / U_REF_EXP:.4f})"
        )

    lines.append(
        f"Uref usato per adimensionalizzare: {u_ref:.3f} m/s "
        f"({args.norm})"
    )

    if wall:
        try:
            lines += [
                "",
                "--- Metriche CFD / TMR / esperimento ---",
                format_metrics(
                    collect_metrics(
                        wall,
                        profiles,
                        tmr,
                        u_ref
                    )
                )
            ]

        except Exception as exc:
            lines.append(
                f"(tabella metriche non calcolata: "
                f"{type(exc).__name__}: {exc})"
            )

    (out / "riepilogo.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\n" + "\n".join(lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())
