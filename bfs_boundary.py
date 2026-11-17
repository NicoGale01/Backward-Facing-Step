#!/usr/bin/env python3
"""
Generate the initial/boundary-condition dictionaries for the BFS case.

OpenFOAM target: v2312

Standalone use (works from Windows Python and from the WSL terminal):
    python bfs_boundary.py

The case folder lives inside WSL:
    /home/nicola/OpenFOAM/nicola-v2312/run/BFS_TMR
When the script runs with Windows Python, this Linux path is translated
automatically to the Windows view of the WSL filesystem
(\\\\wsl.localhost\\<distro>\\home\\nicola\\...) using "wsl.exe wslpath -w".

Turbulence model (default kOmegaSST):
    --model kOmegaSST        fields U p k omega nut
    --model SpalartAllmaras  fields U p nuTilda nut

Options:
    --case PATH      different case folder (Linux path or Windows path)
    --distro NAME    WSL distribution to use, if you have more than one
    --model NAME     kOmegaSST (default) or SpalartAllmaras
    --no-backup      do not create .bak copies of existing field files

Files written (kOmegaSST):
    0/U, 0/p, 0/k, 0/omega, 0/nut
(SpalartAllmaras: 0/U, 0/p, 0/nuTilda, 0/nut; k and omega are left untouched.)
    constant/transportProperties, constant/turbulenceProperties

Expected layout:
    BFS_TMR/
      0/            (created if missing)
      constant/polyMesh/boundary

The script is also suitable for later invocation from a master Python script.
It validates the mesh patches before writing any field files.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. CASE PARAMETERS
# ---------------------------------------------------------------------------

# First trial values agreed for the BFS case.
U_IN = 42.0                         # m/s
TURBULENCE_INTENSITY = 0.00061      # 0.061%
NU = 1.5606e-5                      # m^2/s
TURBULENT_VISCOSITY_RATIO = 0.009   # nut / nu in the inlet freestream

K_IN = 1.5 * (TURBULENCE_INTENSITY * U_IN) ** 2
NUT_IN_ESTIMATE = TURBULENT_VISCOSITY_RATIO * NU
OMEGA_IN = K_IN / NUT_IN_ESTIMATE

# Spalart-Allmaras: freestream value of the transported variable nuTilda.
# NASA TMR: nuTilda_farfield >= 3 * nu (wall value 0).
NUTILDA_RATIO = 3.0                 # nuTilda / nu in the inlet freestream
CV1 = 7.1                           # SA constant of fv1
NUTILDA_IN = NUTILDA_RATIO * NU
NUT_IN_SA = NUTILDA_IN * NUTILDA_RATIO ** 3 / (NUTILDA_RATIO ** 3 + CV1 ** 3)

TURBULENCE_MODELS = ("kOmegaSST", "SpalartAllmaras")
DEFAULT_MODEL = "SpalartAllmaras"

# Turbulence fields to write for each model (U, p and nut are always written).
FIELDS_BY_MODEL = {
    "kOmegaSST": ("U", "p", "k", "omega", "nut"),
    "SpalartAllmaras": ("U", "p", "nuTilda", "nut"),
}


# Case directory, as seen from inside WSL (Linux path, kept as a plain string
# so that Windows Python does not rewrite the slashes).
WSL_CASE_DIR = "/home/nicola/OpenFOAM/nicola-v2312/run/BFS_TMR"


def linux_to_windows_path(linux_path: str, distro: str | None = None,
                          runner=subprocess.run) -> str:
    """Translate a WSL Linux path into the Windows (UNC) path that Windows
    programs can open, by asking WSL itself via 'wslpath -w'."""
    command = ["wsl.exe"]
    if distro:
        command += ["-d", distro]
    command += ["wslpath", "-w", linux_path]

    try:
        result = runner(command, capture_output=True, text=True,
                        encoding="utf-8", timeout=60)
    except FileNotFoundError as error:
        raise RuntimeError(
            "wsl.exe was not found: is WSL installed and in the Windows PATH?"
        ) from error

    windows_path = result.stdout.strip()
    if result.returncode != 0 or not windows_path:
        raise RuntimeError(
            "Could not translate the WSL path with wslpath.\n"
            f"  command: {' '.join(command)}\n"
            f"  stderr : {result.stderr.strip() or '(empty)'}\n"
            "If you have more than one WSL distribution, pass --distro NAME "
            "(see 'wsl -l' in PowerShell)."
        )
    return windows_path


def resolve_case_dir(case_arg: str, distro: str | None = None,
                     windows: bool | None = None,
                     runner=subprocess.run) -> Path:
    """Return a Path that the current Python can open.

    - Linux/WSL Python: the path is used as it is.
    - Windows Python + a Linux path (starts with '/'): translated with wslpath.
    - Windows Python + a Windows path (C:\\..., \\\\wsl$\\...): used as it is.
    """
    if windows is None:
        windows = os.name == "nt"

    if windows and case_arg.startswith("/"):
        case_arg = linux_to_windows_path(case_arg, distro, runner)

    return Path(case_arg).expanduser()


# ---------------------------------------------------------------------------
# 2. PATCH CLASSIFICATION
# ---------------------------------------------------------------------------

INLET_PATCHES = ("inlet",)
OUTLET_PATCHES = ("outlet",)

WALL_PATCHES = (
    "wall_bottom_upstream",
    "wall_top",
    "wall_bottom_downstream",
    "step",
)

SYMMETRY_PATCHES = ("symmetry",)
EMPTY_PATCHES = ("frontAndBack",)

EXPECTED_PATCHES = set(
    INLET_PATCHES
    + OUTLET_PATCHES
    + WALL_PATCHES
    + SYMMETRY_PATCHES
    + EMPTY_PATCHES
)


# ---------------------------------------------------------------------------
# 3. OPENFOAM DICTIONARY HELPERS
# ---------------------------------------------------------------------------

def foam_header(field_class: str, field_name: str, location: str = "0") -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {field_class};
    location    "{location}";
    object      {field_name};
}}

// Generated by generate_boundary_conditions.py

"""


def patch_entry(patch_name: str, body: str) -> str:
    return f"    {patch_name}\n    {{\n{body}    }}\n\n"


def fixed_value(value: str) -> str:
    return f"        type            fixedValue;\n        value           uniform {value};\n"


def zero_gradient() -> str:
    return "        type            zeroGradient;\n"


def no_slip() -> str:
    return "        type            noSlip;\n"


def symmetry() -> str:
    return "        type            symmetry;\n"


def empty() -> str:
    return "        type            empty;\n"


def calculated(value: str = "0") -> str:
    return f"        type            calculated;\n        value           uniform {value};\n"


def wall_function(function_name: str) -> str:
    return (
        f"        type            {function_name};\n"
        "        value           uniform 0;\n"
    )


def make_boundary_field(field_name: str) -> str:
    """Build the complete boundaryField dictionary for one field."""
    entries: list[str] = []

    if field_name == "U":
        entries.append(patch_entry("inlet", fixed_value(f"({U_IN:.10g} 0 0)")))
        entries.append(patch_entry("outlet", zero_gradient()))
        for name in WALL_PATCHES:
            entries.append(patch_entry(name, no_slip()))

    elif field_name == "p":
        entries.append(patch_entry("inlet", zero_gradient()))
        entries.append(patch_entry("outlet", fixed_value("0")))
        for name in WALL_PATCHES:
            entries.append(patch_entry(name, zero_gradient()))

    elif field_name == "k":
        entries.append(patch_entry("inlet", fixed_value(f"{K_IN:.10g}")))
        entries.append(patch_entry("outlet", zero_gradient()))
        for name in WALL_PATCHES:
            entries.append(patch_entry(name, wall_function("kqRWallFunction")))

    elif field_name == "omega":
        entries.append(patch_entry("inlet", fixed_value(f"{OMEGA_IN:.10g}")))
        entries.append(patch_entry("outlet", zero_gradient()))
        for name in WALL_PATCHES:
            entries.append(patch_entry(name, wall_function("omegaWallFunction")))

    elif field_name == "nuTilda":
        entries.append(patch_entry("inlet", fixed_value(f"{NUTILDA_IN:.10g}")))
        entries.append(patch_entry("outlet", zero_gradient()))
        for name in WALL_PATCHES:
            entries.append(patch_entry(name, fixed_value("0")))

    elif field_name == "nut":
        entries.append(patch_entry("inlet", calculated()))
        entries.append(patch_entry("outlet", calculated()))
        for name in WALL_PATCHES:
            entries.append(
                patch_entry(name, wall_function("nutUSpaldingWallFunction"))
            )

    else:
        raise ValueError(f"Unsupported field: {field_name}")

    for name in SYMMETRY_PATCHES:
        entries.append(patch_entry(name, symmetry()))

    for name in EMPTY_PATCHES:
        entries.append(patch_entry(name, empty()))

    return "boundaryField\n{\n" + "".join(entries) + "}\n"


def make_field(field_name: str, field_class: str,
               dimensions: str, internal_field: str) -> str:
    return (
        foam_header(field_class, field_name)
        + f"dimensions      {dimensions};\n\n"
        + f"internalField   uniform {internal_field};\n\n"
        + make_boundary_field(field_name)
    )


def make_transport_properties() -> str:
    """constant/transportProperties: molecular kinematic viscosity."""
    return (
        foam_header("dictionary", "transportProperties", "constant")
        + "transportModel  Newtonian;\n\n"
        + f"nu              [0 2 -1 0 0 0 0] {NU:.10g};\n"
    )


def make_turbulence_properties(model: str = DEFAULT_MODEL) -> str:
    """constant/turbulenceProperties: RAS model (OpenFOAM ESI v2312)."""
    return (
        foam_header("dictionary", "turbulenceProperties", "constant")
        + "simulationType  RAS;\n\n"
        + "RAS\n{\n"
        + f"    model           {model};\n"
        + "    turbulence      on;\n"
        + "    printCoeffs     on;\n"
        + "}\n"
    )


# ---------------------------------------------------------------------------
# 4. READ AND VALIDATE THE MESH PATCHES
# ---------------------------------------------------------------------------

def strip_foam_comments(text: str) -> str:
    """Remove // and /* ... */ comments before parsing the boundary file."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return text


def read_patch_names(boundary_file: Path) -> set[str]:
    """
    Read patch names from constant/polyMesh/boundary.

    A patch entry is recognized by a dictionary containing both 'type' and
    'nFaces'. This avoids mistaking the top-level FoamFile dictionary for
    a mesh patch and does not depend on a fixed indentation width.
    """
    text = strip_foam_comments(boundary_file.read_text(encoding="utf-8"))

    # Ignore the FoamFile header and inspect the mesh patch list after the
    # opening parenthesis that follows the patch count.
    list_match = re.search(r"\b\d+\s*\(", text)
    if not list_match:
        raise ValueError(
            f"Could not locate the patch list in {boundary_file}"
        )

    patch_list = text[list_match.end():]
    names: set[str] = set()

    # OpenFOAM polyMesh/boundary patch dictionaries are flat: each patch
    # has a name followed by a brace-enclosed dictionary.
    entry_pattern = re.compile(
        r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*\{([^{}]*)\}",
        flags=re.MULTILINE,
    )

    for match in entry_pattern.finditer(patch_list):
        name, body = match.groups()
        if re.search(r"(?m)^\s*type\s+\S+\s*;", body) and re.search(
            r"(?m)^\s*nFaces\s+\d+\s*;", body
        ):
            names.add(name)

    if not names:
        raise ValueError(
            f"No mesh patch entries were parsed from {boundary_file}. "
            "The boundary file format may differ from the expected "
            "OpenFOAM polyMesh/boundary format."
        )

    return names


def validate_mesh(case_dir: Path) -> None:
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"

    if not boundary_file.is_file():
        raise FileNotFoundError(
            f"Mesh boundary file not found:\n  {boundary_file}\n"
            "Run the mesh-generation script and checkMesh first."
        )

    actual = read_patch_names(boundary_file)
    missing = EXPECTED_PATCHES - actual
    unexpected = actual - EXPECTED_PATCHES

    if missing or unexpected:
        details = ["Mesh patch validation failed."]
        if missing:
            details.append(f"Expected patches not found: {sorted(missing)}")
        if unexpected:
            details.append(f"Additional mesh patches found: {sorted(unexpected)}")
        details.append(f"Detected patches: {sorted(actual)}")
        raise RuntimeError("\n".join(details))

    print("Mesh patch validation: OK")
    print("Detected patches:")
    for name in sorted(actual):
        print(f"  - {name}")


# ---------------------------------------------------------------------------
# 5. WRITE FIELD FILES
# ---------------------------------------------------------------------------

FIELD_DEFINITIONS = {
    "U": {
        "class": "volVectorField",
        "dimensions": "[0 1 -1 0 0 0 0]",
        "internal": f"({U_IN:.10g} 0 0)",
    },
    "p": {
        "class": "volScalarField",
        "dimensions": "[0 2 -2 0 0 0 0]",
        "internal": "0",
    },
    "k": {
        "class": "volScalarField",
        "dimensions": "[0 2 -2 0 0 0 0]",
        "internal": f"{K_IN:.10g}",
    },
    "omega": {
        "class": "volScalarField",
        "dimensions": "[0 0 -1 0 0 0 0]",
        "internal": f"{OMEGA_IN:.10g}",
    },
    "nuTilda": {
        "class": "volScalarField",
        "dimensions": "[0 2 -1 0 0 0 0]",
        "internal": f"{NUTILDA_IN:.10g}",
    },
    "nut": {
        "class": "volScalarField",
        "dimensions": "[0 2 -1 0 0 0 0]",
        "internal": "0",
    },
}


def write_text_lf(output_file: Path, content: str, backup: bool = True) -> None:
    """Write a text file with LF line endings, keeping a .bak of any old file."""
    if output_file.exists() and backup:
        backup_file = output_file.with_suffix(output_file.suffix + ".bak")
        shutil.copy2(output_file, backup_file)
        print(f"Backup created: {backup_file.name}")

    # Always write LF line endings (OpenFOAM dictionaries must not get CRLF,
    # which Windows Python would otherwise produce).
    with output_file.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    print(f"Written: {output_file}")


def write_constant_files(case_dir: Path, model: str = DEFAULT_MODEL,
                         backup: bool = True) -> None:
    constant_dir = case_dir / "constant"

    if not constant_dir.is_dir():
        raise FileNotFoundError(f"Constant directory does not exist: {constant_dir}")

    write_text_lf(constant_dir / "transportProperties",
                  make_transport_properties(), backup)
    write_text_lf(constant_dir / "turbulenceProperties",
                  make_turbulence_properties(model), backup)


def write_fields(case_dir: Path, model: str = DEFAULT_MODEL,
                 backup: bool = True) -> None:
    zero_dir = case_dir / "0"

    if not zero_dir.is_dir():
        zero_dir.mkdir()
        print(f"Created missing initial conditions directory: {zero_dir}")

    for field_name in FIELDS_BY_MODEL[model]:
        settings = FIELD_DEFINITIONS[field_name]
        output_file = zero_dir / field_name

        content = make_field(
            field_name=field_name,
            field_class=settings["class"],
            dimensions=settings["dimensions"],
            internal_field=settings["internal"],
        )

        write_text_lf(output_file, content, backup)


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate BFS OpenFOAM boundary-condition dictionaries."
    )
    parser.add_argument(
        "--case",
        type=str,
        default=WSL_CASE_DIR,
        help=f"OpenFOAM case directory (default: {WSL_CASE_DIR}).",
    )
    parser.add_argument(
        "--distro",
        type=str,
        default=None,
        help="WSL distribution name (only needed with several distributions).",
    )
    parser.add_argument(
        "--model",
        choices=TURBULENCE_MODELS,
        default=DEFAULT_MODEL,
        help=f"turbulence model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak copies of existing field files.",
    )
    args = parser.parse_args()

    try:
        case_dir = resolve_case_dir(args.case, args.distro).resolve()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    print("\n==================================================")
    print(" BFS - GENERATE INITIAL AND BOUNDARY CONDITIONS")
    print(" OpenFOAM v2312")
    print(f" Turbulence model: {args.model}")
    print("==================================================")
    print(f"Case directory: {case_dir}\n")

    try:
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory does not exist: {case_dir}")

        # Validate first: no field file is changed if the mesh is not correct.
        validate_mesh(case_dir)

        print("\nWriting field dictionaries...")
        write_fields(case_dir, args.model, backup=not args.no_backup)

        print("\nWriting constant dictionaries...")
        write_constant_files(case_dir, args.model, backup=not args.no_backup)

        print("\n--- Numerical values ---")
        print(f"U_in             = {U_IN:.6g} m/s")
        print(f"nu               = {NU:.6e} m2/s")
        if args.model == "SpalartAllmaras":
            print(f"nuTilda/nu       = {NUTILDA_RATIO:.6g}")
            print(f"nuTilda_in       = {NUTILDA_IN:.6e} m2/s")
            print(f"nut_in (fv1)     = {NUT_IN_SA:.6e} m2/s  (nut/nu = {NUT_IN_SA / NU:.4f})")
        else:
            print(f"I                = {TURBULENCE_INTENSITY:.6g}")
            print(f"nut/nu           = {TURBULENT_VISCOSITY_RATIO:.6g}")
            print(f"k_in             = {K_IN:.6e} m2/s2")
            print(f"nut_in estimate  = {NUT_IN_ESTIMATE:.6e} m2/s")
            print(f"omega_in         = {OMEGA_IN:.6e} 1/s")

        print(f"\nSUCCESS: {len(FIELDS_BY_MODEL[args.model])} field dictionaries (0/) "
              "and two constant dictionaries generated.")
        return 0

    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())