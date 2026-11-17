
#!/usr/bin/env python3
"""
bfs_master.py - master script of the BFS_TMR case (OpenFOAM v2312 in WSL, Python on Windows).

Pipeline
  1. bfs_mesh.py         gmsh mesh -> gmshToFoam -> patch types
                         (+ patch check + checkMesh)
  2. bfs_boundary.py     initial fields and properties of the selected model
  3. bfs_controlDict.py  endTime, probes, solverInfo
  4. bfs_fvsolution.py   linear solvers, SIMPLEC, relaxation
  5. bfs_fvSchemes.py    schemes (stage 1, 2, 3) -> once per stage
  6. simpleFoam          by stage: each stage restarts from the last time written by the previous one
  7. postProcess         wallShearStress, writeCellCentres, R
  8. bfs_plots.py        monitors + validation plots

Scripts 1-5 stay independent: the master launches them with the same Python it was started with.
Put all the scripts in the same folder.

Usage
  python bfs_master.py
  python bfs_master.py --model SpalartAllmaras
  python bfs_master.py --model kOmegaSST
  python bfs_master.py --only-setup
  python bfs_master.py --skip-mesh
  python bfs_master.py --skip-run
  python bfs_master.py --only-plots
  python bfs_master.py --stages 3:6000
  python bfs_master.py --stages 1:2000,2:4000,3:6000
  python bfs_master.py --no-clean
  python bfs_master.py --gui-mesh

Note: the final times of the stages must be multiples of the controlDict writeInterval,
otherwise the next stage does not find a time to restart from.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------

DISTRO = "Ubuntu"
CASE_LINUX = "/home/nicola/OpenFOAM/nicola-v2312/run/BFS_TMR"
OF_BASHRC = "/usr/lib/openfoam/openfoam2312/etc/bashrc"

# Default turbulence model
MODEL = "SpalartAllmaras"  # or "kOmegaSST"

# Turbulence fields for each model
TURB_FIELDS_BY_MODEL = {
    "kOmegaSST": "k|omega",
    "SpalartAllmaras": "nuTilda",
}

# (fvSchemes stage, final iteration).
# By default ONE stage only: same schemes for all 6000 iterations.
# The last iteration must match endTime in bfs_controlDict.py.
# To change schemes: --stages 2:6000
# For a ramp: --stages "1:2000,2:4000,3:6000".
STAGES = [(1, 6000)]

PRINT_EVERY = 100  # print the residuals on screen every N iterations

H = 0.0127


SCRIPTS = {
    "mesh": "bfs_mesh.py",
    "boundary": "bfs_boundary.py",
    "controlDict": "bfs_controlDict.py",
    "fvSolution": "bfs_fvsolution.py",
    "fvSchemes": "bfs_fvSchemes.py",
    "plots": "bfs_plots.py",
}


EXPECTED_PATCH_TYPES = {
    "inlet": "patch",
    "outlet": "patch",
    "symmetry": "symmetry",
    "wall_bottom_upstream": "wall",
    "wall_top": "wall",
    "wall_bottom_downstream": "wall",
    "step": "wall",
    "frontAndBack": "empty",
}


IS_WINDOWS = os.name == "nt"


# ----------------------------------------------------------------------------
# WSL / OPENFOAM
# ----------------------------------------------------------------------------

def banner(text: str):
    print("\n" + "=" * 72 + f"\n {text}\n" + "=" * 72, flush=True)


def bash_argv(cmd: str):
    if IS_WINDOWS:
        return ["wsl.exe", "-d", DISTRO, "--", "bash", "-c", cmd]
    return ["bash", "-c", cmd]


def of(cmd: str) -> str:
    """Command run in the case folder with the OpenFOAM environment loaded."""
    return (
        f"source {OF_BASHRC} >/dev/null 2>&1; "
        f"cd {shlex.quote(CASE_LINUX)} || exit 90; {cmd}"
    )


def wsl_run(cmd: str, stdin_text: str | None = None):
    """Run a bash command; return (returncode, stdout, stderr)."""
    res = subprocess.run(
        bash_argv(cmd),
        input=stdin_text.encode("utf-8") if stdin_text is not None else None,
        capture_output=True,
    )

    return (
        res.returncode,
        res.stdout.decode("utf-8", "replace"),
        res.stderr.decode("utf-8", "replace"),
    )


def write_case_file(rel_path: str, content: str):
    target = f"{CASE_LINUX}/{rel_path}"

    rc, _, err = wsl_run(
        f"cat > {shlex.quote(target)}",
        stdin_text=content
    )

    if rc != 0:
        raise RuntimeError(f"scrittura di {target} fallita: {err.strip()}")


def case_windows() -> Path:
    if not IS_WINDOWS:
        return Path(CASE_LINUX)

    rel = CASE_LINUX.lstrip("/")

    for root in (rf"\\wsl.localhost\{DISTRO}", rf"\\wsl$\{DISTRO}"):
        p = Path(root) / rel

        if p.exists():
            return p

    raise FileNotFoundError(
        f"Cartella del caso non trovata in WSL ({DISTRO}): {CASE_LINUX}. "
        "Controlla che la distro sia avviata e che DISTRO/CASE_LINUX siano corretti."
    )


def run_py(key: str, extra=None, cwd=HERE):
    script = HERE / SCRIPTS[key]

    if not script.is_file():
        raise FileNotFoundError(f"Script mancante: {script}")

    print(
        f"\n>>> {SCRIPTS[key]} {' '.join(extra or [])}",
        flush=True
    )

    rc = subprocess.run(
        [sys.executable, str(script), *(extra or [])],
        cwd=str(cwd)
    ).returncode

    if rc != 0:
        raise RuntimeError(f"{SCRIPTS[key]} terminato con codice {rc}")


# ----------------------------------------------------------------------------
# 1. MESH
# ----------------------------------------------------------------------------

def read_patch_types() -> dict[str, str]:
    rc, out, err = wsl_run(of("cat constant/polyMesh/boundary"))

    if rc != 0 or not out.strip():
        raise FileNotFoundError(
            "constant/polyMesh/boundary non trovato: la mesh non e' stata convertita "
            f"({err.strip()})"
        )

    types = {}

    for name, body in re.findall(
        r"^\s*(\w+)\s*\n\s*\{([^{}]*)\}",
        out,
        flags=re.M
    ):
        if re.search(r"\bnFaces\b", body):
            m = re.search(r"\btype\s+(\w+)\s*;", body)
            types[name] = m.group(1) if m else "?"

    return types


def verify_patches():
    types = read_patch_types()

    wrong = {
        n: (types.get(n), t)
        for n, t in EXPECTED_PATCH_TYPES.items()
        if types.get(n) != t
    }

    extra = sorted(set(types) - set(EXPECTED_PATCH_TYPES))

    if wrong or extra:
        msg = ["Patch della mesh non corrette."]

        for n, (found, wanted) in wrong.items():
            msg.append(f"  {n}: trovato '{found}', atteso '{wanted}'")

        if extra:
            msg.append(f"  patch inattese: {extra}")

        msg.append(
            "Controlla il passaggio 'Configurazione tipi delle patch' di bfs_mesh.py "
            "(constant/polyMesh/boundary)."
        )

        raise RuntimeError("\n".join(msg))

    print(
        "Patch della mesh: OK  "
        + ", ".join(f"{n}={t}" for n, t in sorted(types.items()))
    )


def run_checkmesh():
    print("\n>>> checkMesh (log.checkMesh)", flush=True)

    rc, out, _ = wsl_run(of(
        r"checkMesh -allGeometry -allTopology > log.checkMesh 2>&1; "
        r"grep -E 'cells:|Max aspect ratio|Max non-orthogonality|Mesh OK|Failed|\*\*\*' log.checkMesh"
    ))

    print(out.rstrip() or "(nessun output da checkMesh: guarda log.checkMesh)")


def step_mesh(gui: bool):
    banner("1/8  MESH")

    run_py("mesh", [] if gui else ["-nogui"])

    verify_patches()
    run_checkmesh()


# ----------------------------------------------------------------------------
# 2-5. DICTIONARIES
# ----------------------------------------------------------------------------

def step_dictionaries(first_stage: int):
    banner("2-5/8  GENERAZIONE DIZIONARI")

    run_py("boundary", ["--model", MODEL])
    run_py("controlDict")
    run_py("fvSolution", ["--model", MODEL])
    run_py("fvSchemes", [str(first_stage), "--model", MODEL])


# ----------------------------------------------------------------------------
# 6. STAGED SIMULATION
# ----------------------------------------------------------------------------

def clean_case():
    print(
        "\nPulizia: tempi numerici (tranne 0), postProcessing "
        "e vecchi log.simpleFoam.*"
    )

    rc, _, err = wsl_run(
        of("rm -rf [1-9]* postProcessing log.simpleFoam.* log.pp.*")
    )

    if rc != 0:
        raise RuntimeError(f"pulizia fallita: {err.strip()}")


def set_control(end_time: int, restart: bool):
    cmds = [
        rf"sed -i -E 's/^(endTime[[:space:]]+)[^;]*;/\1{end_time};/' system/controlDict"
    ]

    if restart:
        cmds.append(
            r"sed -i -E 's/^(startFrom[[:space:]]+)[^;]*;/\1latestTime;/' system/controlDict"
        )

    else:
        cmds.append(
            r"sed -i -E 's/^(startFrom[[:space:]]+)[^;]*;/\1startTime;/' system/controlDict"
        )
        cmds.append(
            r"sed -i -E 's/^(startTime[[:space:]]+)[^;]*;/\10;/' system/controlDict"
        )

    cmds.append(
        r"grep -E '^(startFrom|startTime|endTime|writeInterval|writeControl)\b' system/controlDict"
    )

    rc, out, err = wsl_run(of("; ".join(cmds)))

    if rc != 0:
        raise RuntimeError(f"modifica del controlDict fallita: {err.strip()}")

    print(
        "controlDict: "
        + " | ".join(l.strip() for l in out.strip().splitlines())
    )


def latest_time() -> float:
    rc, out, _ = wsl_run(of("foamListTimes -latestTime"))

    try:
        return float(out.strip().splitlines()[-1])

    except (ValueError, IndexError):
        raise RuntimeError("impossibile leggere l'ultimo tempo con foamListTimes")


def run_simplefoam(stage: int, end_time: int):
    log = f"log.simpleFoam.stage{stage}"

    # Turbulence fields to monitor for the selected model
    turb_fields = TURB_FIELDS_BY_MODEL[MODEL]
    residual_pattern = rf"Solving for (Ux|Uy|p|{turb_fields})\b"

    print(
        f"\n>>> simpleFoam (stage {stage}, fino a {end_time}) "
        f"log: {log}",
        flush=True
    )

    proc = subprocess.Popen(
        bash_argv(of(f"set -o pipefail; simpleFoam 2>&1 | tee {log}")),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    show = False
    t0 = time.time()

    try:
        for line in proc.stdout:
            m = re.match(r"^Time = (\d+(?:\.\d+)?)", line)

            if m:
                it = int(float(m.group(1)))
                show = (it % PRINT_EVERY == 0) or it == 1

                if show:
                    print(
                        f"{line.rstrip()}   [{time.time() - t0:6.0f} s]",
                        flush=True
                    )

            elif show and re.search(residual_pattern, line):
                print("   " + line.strip(), flush=True)

            elif re.search(
                r"FOAM FATAL|FOAM exiting|Floating point exception|SIGFPE",
                line
            ):
                print(line.rstrip(), flush=True)

        rc = proc.wait()

    except KeyboardInterrupt:
        print("\nInterrotto: fermo simpleFoam in WSL...")
        wsl_run("pkill simpleFoam")
        proc.terminate()
        raise

    if rc != 0:
        _, tail, _ = wsl_run(of(f"tail -n 30 {log}"))

        raise RuntimeError(
            f"simpleFoam (stage {stage}) terminato con codice {rc}. "
            f"Ultime righe del log:\n{tail}"
        )

    _, nb, _ = wsl_run(of(f"grep -c bounding {log}"))

    print(
        f"Stage {stage} concluso in {time.time() - t0:.0f} s "
        f"| righe 'bounding' nel log: {nb.strip() or 0}"
    )


def step_run(stages, clean: bool):
    banner("6/8  SIMULAZIONE simpleFoam")

    if clean:
        clean_case()

    prev_end = 0

    for i, (stage, end) in enumerate(stages):

        if end <= prev_end:
            raise ValueError("i tempi finali degli stage devono essere crescenti")

        run_py("fvSchemes", [str(stage), "--model", MODEL])

        set_control(end, restart=(i > 0 or not clean))

        if i > 0:
            lt = latest_time()

            if lt != prev_end:
                raise RuntimeError(
                    f"l'ultimo tempo scritto e' {lt:g}, non {prev_end}: "
                    "gli endTime degli stage devono essere multipli del writeInterval."
                )

        run_simplefoam(stage, end)

        prev_end = end


# ----------------------------------------------------------------------------
# 7. POST-PROCESSING
# ----------------------------------------------------------------------------

def step_postprocess():
    banner("7/8  POST-PROCESSING")

    # Profiles and wall pressure are computed in Python from U, nut, p and C
    # (see bfs_plots.py).

    for func, cmd in (
        ("wallShearStress", "postProcess -latestTime -func wallShearStress"),
        ("writeCellCentres", "postProcess -latestTime -func writeCellCentres"),
        ("R", "simpleFoam -postProcess -latestTime -func R"),
    ):

        print(f">>> {cmd}", flush=True)

        rc, out, err = wsl_run(of(
            f"{cmd} > log.pp.{func} 2>&1; "
            f"rc=$?; tail -n 3 log.pp.{func}; exit $rc"
        ))

        if rc != 0:
            raise RuntimeError(
                f"{cmd} fallito (vedi log.pp.{func} nel caso):\n{out}{err}"
            )


# ----------------------------------------------------------------------------
# 8. PLOT
# ----------------------------------------------------------------------------

def step_plots(stages):
    banner("8/8  PLOT")

    marks = [str(end) for _, end in stages[:-1]]
    out_dir = HERE / "risultati_bfs"

    cmd = [
        sys.executable,
        str(HERE / SCRIPTS["plots"]),
        "--case", str(case_windows()),
        "--out", str(out_dir),
        "--model", MODEL,
    ]

    if marks:
        cmd += ["--marks", *marks]

    rc = subprocess.run(cmd, cwd=str(HERE)).returncode

    if rc != 0:
        raise RuntimeError(
            f"{SCRIPTS['plots']} terminato con codice {rc}"
        )

    print(f"\nPlot in: {out_dir}")

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def parse_stages(text: str):
    out = []

    for part in text.split(","):
        s, e = part.split(":")
        out.append((int(s), int(e)))

    return out


def main() -> int:
    global DISTRO, MODEL

    ap = argparse.ArgumentParser(description="Master del caso BFS_TMR")

    ap.add_argument(
        "--model",
        choices=sorted(TURB_FIELDS_BY_MODEL),
        default=MODEL,
        help="Modello di turbolenza: kOmegaSST oppure SpalartAllmaras"
    )

    ap.add_argument(
        "--only-setup",
        action="store_true",
        help="mesh + dizionari, senza simulazione"
    )

    ap.add_argument(
        "--only-plots",
        action="store_true",
        help="solo post-processing e plot"
    )

    ap.add_argument(
        "--skip-mesh",
        action="store_true",
        help="non rigenera la mesh"
    )

    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="non lancia simpleFoam"
    )

    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="non cancella tempi e postProcessing precedenti"
    )

    ap.add_argument(
        "--gui-mesh",
        action="store_true",
        help="apre la finestra di gmsh"
    )

    ap.add_argument(
        "--stages",
        type=parse_stages,
        default=STAGES,
        help='stage fvSchemes e iterazione finale, es. "1:2000,2:4000,3:6000"'
    )

    ap.add_argument(
        "--distro",
        default=DISTRO,
        help="distribuzione WSL"
    )

    args = ap.parse_args()

    DISTRO = args.distro
    MODEL = args.model
    stages = args.stages

    print(f"\nModello di turbolenza selezionato: {MODEL}")
    print(f"Campi turbolenti: {TURB_FIELDS_BY_MODEL[MODEL]}")

    t0 = time.time()

    try:
        for key in SCRIPTS:
            if not (HERE / SCRIPTS[key]).is_file():
                raise FileNotFoundError(
                    f"Script mancante accanto al master: {SCRIPTS[key]}"
                )

        if not args.only_plots:

            if args.skip_mesh:
                banner("1/8  MESH (saltata)")
                verify_patches()

            else:
                step_mesh(args.gui_mesh)

            step_dictionaries(stages[0][0])

            if not (args.only_setup or args.skip_run):
                step_run(stages, clean=not args.no_clean)

        if not args.only_setup:
            step_postprocess()
            step_plots(stages)

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
        return 130

    except Exception as exc:
        print(f"\nERRORE: {exc}", file=sys.stderr)
        return 1

    print(f"\nFinito in {(time.time() - t0) / 60:.1f} min.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
