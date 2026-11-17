
"""
bfs_fvsolution.py
Writes the system/fvSolution file of the BFS_TMR case (simpleFoam, OpenFOAM v2312)
directly into the case folder in WSL, launched from Windows.

Usage:
    python bfs_fvsolution.py
    python bfs_fvsolution.py --model kOmegaSST
    python bfs_fvsolution.py --model SpalartAllmaras

Change the parameters in the "PARAMETERS" section.
"""

from pathlib import Path
from string import Template
import shutil
import argparse


# ----------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------

# Case in WSL
DISTRO = "Ubuntu"
CASE_LINUX = "/home/nicola/OpenFOAM/nicola-v2312/run/BFS_TMR"

# Linear solver for pressure (GAMG)
P_RELTOL = 0.1          # if the p residual stalls, try 0.01
P_TOL = 1e-10

# Linear solver for U and the turbulence quantities (smoothSolver)
TRANSPORT_RELTOL = 0.1
TRANSPORT_TOL = 1e-10

# Default turbulence model
MODEL = "SpalartAllmaras"

# Fields transported by each model
TURB_FIELDS_BY_MODEL = {
    "kOmegaSST": "k|omega",
    "SpalartAllmaras": "nuTilda",
}

# SIMPLE algorithm
N_NON_ORTH_CORR = 0      # checkMesh: non-orthogonality 0 -> no correction
CONSISTENT = True        # True = SIMPLEC, False = classic SIMPLE

# Relaxation
# SIMPLEC: p not relaxed (None), U and turbulence at 0.9
# classic SIMPLE: set p = 0.3 and U/turbulence = 0.7
RELAX_P = None
RELAX_U = 0.5
RELAX_TURB = 0.3

# Residual criterion (in simpleFoam it lives here in fvSolution, not in controlDict)
# For the first 6000-iteration test leave it off.
USE_RESIDUAL_CONTROL = False
RESIDUAL_TOL = 1e-5


# ----------------------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera system/fvSolution per il caso BFS_TMR."
    )

    parser.add_argument(
        "--model",
        choices=sorted(TURB_FIELDS_BY_MODEL),
        default=MODEL,
        help="Modello di turbolenza (default: kOmegaSST)."
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# TEMPLATE
# ----------------------------------------------------------------------

TEMPLATE = Template(r"""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  www.openfoam.com
    \\  /    A nd           | Version:  v2312
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       $p_tol;
        relTol          $p_reltol;
    }

    "(U|$turb_fields)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       $tr_tol;
        relTol          $tr_reltol;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors $n_no_corr;
    consistent      $consistent;
$residual_block}

relaxationFactors
{
$fields_block    equations
    {
        U               $relax_u;
        ".*"            $relax_turb;
    }
}

// ************************************************************************* //
""")


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def build_residual_block(turb_fields):
    if not USE_RESIDUAL_CONTROL:
        return ""

    return (
        "\n"
        "    residualControl\n"
        "    {\n"
        f"        p               {RESIDUAL_TOL:g};\n"
        f"        U               {RESIDUAL_TOL:g};\n"
        f'        "({turb_fields})" {RESIDUAL_TOL:g};\n'
        "    }\n"
    )


def build_fields_block():
    if RELAX_P is None:
        return ""

    return (
        "    fields\n"
        "    {\n"
        f"        p               {RELAX_P:g};\n"
        "    }\n\n"
    )


def find_case_dir():
    """Try \\wsl.localhost first, then \\wsl$ (older Windows versions)."""
    rel = CASE_LINUX.lstrip("/")

    for root in (rf"\\wsl.localhost\{DISTRO}", rf"\\wsl$\{DISTRO}"):
        case = Path(root) / rel
        if case.exists():
            return case

    raise FileNotFoundError(
        f"Cartella del caso non trovata in WSL ({DISTRO}): {CASE_LINUX}\n"
        "Controlla che la distro sia avviata (apri un terminale Ubuntu) "
        "e che DISTRO e CASE_LINUX siano corretti."
    )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    model = args.model
    turb_fields = TURB_FIELDS_BY_MODEL[model]

    print(f"Modello di turbolenza selezionato: {model}")
    print(f"Campi turbolenti: {turb_fields}")

    case = find_case_dir()
    system_dir = case / "system"
    system_dir.mkdir(exist_ok=True)
    target = system_dir / "fvSolution"

    text = TEMPLATE.substitute(
        p_tol=f"{P_TOL:g}",
        p_reltol=f"{P_RELTOL:g}",
        tr_tol=f"{TRANSPORT_TOL:g}",
        tr_reltol=f"{TRANSPORT_RELTOL:g}",
        turb_fields=turb_fields,
        n_no_corr=N_NON_ORTH_CORR,
        consistent="yes" if CONSISTENT else "no",
        residual_block=build_residual_block(turb_fields),
        fields_block=build_fields_block(),
        relax_u=f"{RELAX_U:g}",
        relax_turb=f"{RELAX_TURB:g}",
    )

    # Back up the previous file
    if target.exists():
        shutil.copy2(target, system_dir / "fvSolution.bak")

    # newline="\n": OpenFOAM on Linux must not receive Windows CRLF
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print(f"fvSolution scritto in: {target}")
    print(f"Configurazione completata per {model}.")


if __name__ == "__main__":
    main()
