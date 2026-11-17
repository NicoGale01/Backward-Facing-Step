#!/usr/bin/env python3
"""
Edit system/controlDict of the backward-facing step case:
  - set the top-level keywords (endTime, etc.)
  - add probes, patchProbes and solverInfo inside the functions block
Reference frame (from bfs_mesh.py): origin at the foot of the step,
downstream floor y = 0, upstream floor y = H, top wall y = 9H, z in [0, 0.1H].

Usage:  python bfs_controldict.py [case_path]
"""
import argparse
import re
import shutil
from pathlib import Path
import subprocess

WSL_CASE_DIR = "/home/nicola/OpenFOAM/nicola-v2312/run/BFS_TMR"


def wsl_to_windows(linux_path: str) -> str:
    r = subprocess.run(
        ["wsl.exe", "--", "wslpath", "-w", linux_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        raise SystemExit(f"wslpath non è riuscito a convertire {linux_path}:\n{r.stderr}")
    return out

# ----------------------------- PARAMETERS ----------------------------
H = 0.0127
Z_PROBE = 0.05 * H            # half of the extruded thickness (dz = 0.1 H)
EPS = 0.01 * H                # small offset to avoid landing on a cell face
LOWER_WALL = "wall_bottom_downstream"

TOP_LEVEL = {
    "endTime": "6000",
    # "writeInterval": "500",
    # "purgeWrite": "0",
}

# x/H -> list of y/H
PROBE_POINTS_H = {
    -4.0: [5.0],            # upstream channel centre (y = 5H)
    1.0: [0.5, 1.0, 1.5],
    4.0: [0.5, 1.0, 1.5],
    6.0: [0.5, 1.0, 1.5],
    10.0: [0.5, 1.0, 1.5],
}

# face centres on the downstream floor (cells 0.5H wide: centres at k*0.5 + 0.25)
WALL_POINTS_H = [2.25, 4.25, 5.25, 5.75, 6.25, 6.75, 7.25, 10.25, 20.25]

BEGIN = "    // >>> BEGIN monitor generati da bfs_controldict.py"
END = "    // <<< END monitor generati da bfs_controldict.py"
# ---------------------------------------------------------------------


def fmt(v: float) -> str:
    return f"{v:.6g}"


def build_block() -> str:
    probe_lines = []
    for xh, ys in PROBE_POINTS_H.items():
        for yh in ys:
            probe_lines.append(
                f"            ({fmt(xh * H + EPS)} {fmt(yh * H + EPS)} {fmt(Z_PROBE)})"
                f"   // x/H = {xh:g}, y/H = {yh:g}"
            )

    wall_lines = [
        f"            ({fmt(xh * H)} 0 {fmt(Z_PROBE)})   // x/H = {xh:g}"
        for xh in WALL_POINTS_H
    ]

    return "\n".join([
        BEGIN,
        "    #includeFunc solverInfo",
        "",
        "    probesFlow",
        "    {",
        "        type            probes;",
        '        libs            ("libsampling.so");',
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "        fields          (U p);",
        "        probeLocations",
        "        (",
        *probe_lines,
        "        );",
        "    }",
        "",
        "    probesWallShear",
        "    {",
        "        type            patchProbes;",
        '        libs            ("libsampling.so");',
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "        fields          (wallShearStress);",
        f"        patches         ({LOWER_WALL});",
        "        probeLocations",
        "        (",
        *wall_lines,
        "        );",
        "    }",
        END,
    ])


def set_keyword(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^({key}\s+)[^;\n]*;", re.MULTILINE)
    new, n = pattern.subn(rf"\g<1>{value};", text, count=1)
    if n == 0:
        raise SystemExit(f"Keyword '{key}' non trovata nel controlDict")
    return new


def find_functions_body(text: str):
    m = re.search(r"^functions\s*\{", text, re.MULTILINE)
    if not m:
        raise SystemExit("Blocco 'functions' non trovato")
    depth, i = 1, m.end()
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth:
        raise SystemExit("Parentesi graffe non bilanciate in 'functions'")
    return m.end(), i - 1


def insert_block(text: str, block: str) -> str:
    start, close = find_functions_body(text)
    body = text[start:close]
    old = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?",
                     re.DOTALL)
    body = old.sub("", body)
    body = body.rstrip() + "\n\n" + block + "\n"
    return text[:start] + body + text[close:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default=None,
                    help="cartella del caso (default: il caso WSL)")
    args = ap.parse_args()

    case = args.case if args.case else wsl_to_windows(WSL_CASE_DIR)
    path = Path(case) / "system" / "controlDict"
    
    if not path.exists():
        raise SystemExit(f"Non trovo {path}")

    backup = path.with_name("controlDict.bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    text = path.read_text(encoding="utf-8")
    for key, value in TOP_LEVEL.items():
        if value is not None:
            text = set_keyword(text, key, value)
    text = insert_block(text, build_block())

    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(text)
    print(f"controlDict aggiornato: {path}")


if __name__ == "__main__":
    main()