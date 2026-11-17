"""
bfs_fvschemes.py
Writes the system/fvSchemes file of the backward-facing step case (OpenFOAM v2312, ESI)
directly inside WSL, starting from Python on Windows.

Usage:
    python bfs_fvschemes.py                              # uses STAGE and MODEL defined below
    python bfs_fvschemes.py 2                            # stage from the command line
    python bfs_fvschemes.py 3 --model SpalartAllmaras    # stage and turbulence model

Stage:
    1 = all upwind (robust start)
    2 = linearUpwind on U, upwind on the turbulence quantities
    3 = linearUpwind on U, limitedLinear 1 on the turbulence quantities (second order everywhere)

Turbulence models (transported quantities):
    kOmegaSST       -> k, omega
    SpalartAllmaras -> nuTilda
The turbulence quantities use the same scheme in every stage.
"""

import argparse
import subprocess
import sys

# ----------------------------------------------------------------------------
# TO BE EDITED
# ----------------------------------------------------------------------------
WSL_DISTRO = "Ubuntu"                          # distro name: see `wsl -l` in PowerShell
CASE_DIR = "/home/nicola/OpenFOAM/nicola-v2312/run/BFS_TMR"   # LINUX path of the case folder
STAGE = 1                                      # 1, 2 or 3
MODEL = "SpalartAllmaras"                            # "kOmegaSST" or "SpalartAllmaras"
MAKE_BACKUP = True                             # save the old file as fvSchemes.bak
# ----------------------------------------------------------------------------

# Turbulence quantities transported by each model (each has its own div(phi,...))
TURB_FIELDS = {
    "kOmegaSST": ["k", "omega"],
    "SpalartAllmaras": ["nuTilda"],
}

# Convection schemes for each stage: U and turbulence quantities
STAGES = {
    1: {"U": "bounded Gauss upwind",
        "turb": "bounded Gauss upwind"},
    2: {"U": "bounded Gauss linearUpwind grad(U)",
        "turb": "bounded Gauss upwind"},
    3: {"U": "bounded Gauss linearUpwind grad(U)",
        "turb": "bounded Gauss limitedLinear 1"},
}

# Compatible with the master: bfs_master.py passes the stage as first argument
parser = argparse.ArgumentParser(description="Scrive system/fvSchemes del caso BFS_TMR")
parser.add_argument("stage", nargs="?", type=int, default=STAGE,
                    help="stage degli schemi: 1, 2 o 3")
parser.add_argument("--model", choices=sorted(TURB_FIELDS), default=MODEL,
                    help="modello di turbolenza")
args = parser.parse_args()
STAGE, MODEL = args.stage, args.model

if STAGE not in STAGES:
    sys.exit(f"STAGE deve essere 1, 2 o 3 (ricevuto: {STAGE})")

s = STAGES[STAGE]


def div_line(name, scheme):
    return f"    {name:<18}{scheme};"


DIV_LINES = "\n".join(
    [div_line("div(phi,U)", s["U"])]
    + [div_line(f"div(phi,{f})", s["turb"]) for f in TURB_FIELDS[MODEL]]
)

FVSCHEMES = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Version:  v2312
    \\\\  /    A nd           | Website:  www.openfoam.com
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Generato da bfs_fvschemes.py - STAGE {STAGE} - modello {MODEL}

ddtSchemes
{{
    default         steadyState;
}}

gradSchemes
{{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}}

divSchemes
{{
    default         none;

{DIV_LINES}

    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}

wallDist
{{
    method          meshWave;
}}

// ************************************************************************* //
"""


def wsl(cmd, input_text=None):
    """Run a bash command inside the WSL distro and return (returncode, stdout, stderr)."""
    res = subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "--", "bash", "-c", cmd],
        input=input_text.encode("utf-8") if input_text is not None else None,
        capture_output=True,
    )
    return res.returncode, res.stdout.decode("utf-8", "replace"), res.stderr.decode("utf-8", "replace")


def main():
    target = f"{CASE_DIR}/system/fvSchemes"

    # 1) check that the system folder exists
    rc, _, err = wsl(f"test -d '{CASE_DIR}/system'")
    if rc != 0:
        sys.exit(f"Cartella non trovata: {CASE_DIR}/system\n"
                 f"Controlla WSL_DISTRO e CASE_DIR. {err.strip()}")

    # 2) back up the existing file
    if MAKE_BACKUP:
        rc, _, _ = wsl(f"test -f '{target}' && cp '{target}' '{target}.bak'")
        if rc == 0:
            print(f"Backup creato: {target}.bak")

    # 3) write via stdin: line endings stay LF (no Windows CRLF problems)
    rc, _, err = wsl(f"cat > '{target}'", input_text=FVSCHEMES)
    if rc != 0:
        sys.exit(f"Scrittura fallita: {err.strip()}")

    # 4) read back to confirm
    rc, out, err = wsl(f"grep -E 'div\\(phi' '{target}'")
    print(f"fvSchemes scritto in {target} (STAGE {STAGE}, modello {MODEL}). Schemi di convezione:")
    print(out.rstrip())


if __name__ == "__main__":
    main()