"""
Backward Facing Step (NASA TMR 2DBFS) - geometry and structured mesh in gmsh.

Units: metres. Step height H = 0.0127 m; the coordinates are written as multiples of H.

Usage:
    python bfs_mesh.py            # generates the mesh and opens the gmsh window
    python bfs_mesh.py -nogui     # generates the mesh and prints only the summary
"""

import os
import sys
import shlex
import subprocess
import gmsh

# ----------------------------------------------------------------------------
# MESH PARAMETERS
# ----------------------------------------------------------------------------
# Number of intervals (not nodes)
N_verticali_up = 140         # y from 1 to 9, at x = -130, x = 0, x = 50 (must be EVEN)
N_verticali_down = 60        # y from 0 to 1, at x = 0 (step) and x = 50 (outlet) (must be EVEN)
N_orizzontali_inlet = 260    # x from -130 to 0 (upstream of the step)
N_orizzontali_outlet = 200   # x from 0 to 50 (downstream of the step)

# Clustering towards the horizontal walls (y = 1 upstream, y = 9, y = 0 downstream).
# y+ is computed as in OpenFOAM, i.e. at the CENTRE of the first cell:
#   y+ = (dy_wall / 2) * u_tau / nu
Y_PLUS_TARGET = 0.5          # target y+ at the centre of the first cell

# Estimate of u_tau at x/H = -4 (values from the summary; nu and Cf must be checked)
U_REF = 44.2                 # [m/s]
RE_H = 36000.0               # Reynolds number based on H
CF_REF = 0.0029              # ERCOFTAC, rounded value

# 2D extrusion (a single cell in z, to be used with "empty" patches in OpenFOAM)
ESTRUDI = True
DZ_OVER_H = 0.1              # thickness in z, in units of H (arbitrary in the 2D case)

# Saving the mesh for OpenFOAM (gmshToFoam reads the MSH 2.2 ASCII format)
SALVA_MSH = True
NOME_MSH = "bfs.msh"         # saved in the folder from which you run the script

# OpenFOAM case in WSL: Linux path of the case folder.
WSL_CASE_DIR = "/home/nicola/OpenFOAM/nicola-v2312/run/BFS_TMR"
ESEGUI_CHECKMESH = True      # after closing the Gmsh window

# ----------------------------------------------------------------------------
# GEOMETRY (metres; the coordinates are multiples of H)
# ----------------------------------------------------------------------------
H = 0.0127                 # step height [m]

X_IN = -130.0 * H          # inlet
X_SYM = -110.0 * H         # end of the symmetry segment, start of the no-slip walls
X_STEP = 0.0 * H           # step
X_OUT = 50.0 * H           # outlet
Y_STEP_BOTTOM = 0.0 * H    # floor downstream of the step
Y_STEP_TOP = 1.0 * H       # floor upstream of the step
Y_TOP = 9.0 * H            # top wall
Y_MID_UP = 0.5 * (Y_STEP_TOP + Y_TOP)           # upstream channel mid-height
Y_MID_DOWN = 0.5 * (Y_STEP_BOTTOM + Y_STEP_TOP)  # step mid-height

# The symmetry segment (x from -130 to -110) needs one extra point on the
# upstream walls. N_orizzontali_inlet is the TOTAL number of intervals between
# x = -130 and x = 0: it is split between the symmetry segment and the wall segment
# in proportion to their length, so the spacing stays roughly uniform.
N_sym = max(1, round(N_orizzontali_inlet * (X_SYM - X_IN) / (X_STEP - X_IN)))
N_wall = N_orizzontali_inlet - N_sym

# Estimate of nu, u_tau and first-cell height
NU = U_REF * H / RE_H
U_TAU = U_REF * (CF_REF / 2.0) ** 0.5
DY_WALL = 2.0 * Y_PLUS_TARGET * NU / U_TAU


def growth_ratio(length, n, dy1):
    """Ratio r such that dy1*(r^n - 1)/(r - 1) = length (n cells, the first one dy1 high)."""
    if n * dy1 >= length:
        raise ValueError(
            f"{n} celle alte almeno {dy1:.2e} m coprono già {n * dy1:.3e} m >= {length:.3e} m: "
            "riduci N o aumenta dy_wall.")
    lo, hi = 1.0 + 1e-12, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dy1 * (mid ** n - 1.0) / (mid - 1.0) > length:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_geometry():
    for name, n in (("N_verticali_up", N_verticali_up), ("N_verticali_down", N_verticali_down)):
        if n % 2:
            raise ValueError(f"{name} deve essere pari (l'infittimento è simmetrico: {n} è dispari).")
    n_up_half = N_verticali_up // 2
    n_dn_half = N_verticali_down // 2
    r_up = growth_ratio(0.5 * (Y_TOP - Y_STEP_TOP), n_up_half, DY_WALL)
    r_dn = growth_ratio(0.5 * (Y_STEP_TOP - Y_STEP_BOTTOM), n_dn_half, DY_WALL)

    geo = gmsh.model.geo

    # --- Points ------------------------------------------------------------
    p1 = geo.addPoint(X_IN, Y_STEP_TOP, 0)
    p2 = geo.addPoint(X_SYM, Y_STEP_TOP, 0)
    p3 = geo.addPoint(X_STEP, Y_STEP_TOP, 0)    # step edge
    p4 = geo.addPoint(X_OUT, Y_STEP_TOP, 0)     # end of the internal line y = 1

    p5 = geo.addPoint(X_IN, Y_TOP, 0)
    p6 = geo.addPoint(X_SYM, Y_TOP, 0)
    p7 = geo.addPoint(X_STEP, Y_TOP, 0)
    p8 = geo.addPoint(X_OUT, Y_TOP, 0)

    p9 = geo.addPoint(X_STEP, Y_STEP_BOTTOM, 0)  # step foot
    p10 = geo.addPoint(X_OUT, Y_STEP_BOTTOM, 0)

    # Mid-height points: they only split the vertical segments in two,
    # so the clustering towards the walls can be an exact geometric
    # progression (first cell = DY_WALL on both sides).
    pm_in = geo.addPoint(X_IN, Y_MID_UP, 0)
    pm_mid = geo.addPoint(X_STEP, Y_MID_UP, 0)
    pm_out = geo.addPoint(X_OUT, Y_MID_UP, 0)
    ps_mid = geo.addPoint(X_STEP, Y_MID_DOWN, 0)
    ps_out = geo.addPoint(X_OUT, Y_MID_DOWN, 0)

    # --- Lines (vertical segments are always oriented upwards) -------------
    # Floor upstream of the step (y = 1)
    l_bot_sym = geo.addLine(p1, p2)      # symmetry
    l_bot_wall = geo.addLine(p2, p3)     # no-slip wall

    # Internal line (0,1) -> (50,1): it only splits the downstream blocks,
    # it is NOT a physical boundary and is not put in any physical group.
    l_internal = geo.addLine(p3, p4)

    # Top wall (traversed from x = 50 to x = -130)
    l_top_wall_down = geo.addLine(p8, p7)   # x from 50 to 0
    l_top_wall_up = geo.addLine(p7, p6)     # x from 0 to -110, no-slip
    l_top_sym = geo.addLine(p6, p5)         # x from -110 to -130, symmetry

    # Inlet (x = -130): lower half and upper half
    l_inlet_lo = geo.addLine(p1, pm_in)
    l_inlet_hi = geo.addLine(pm_in, p5)

    # Internal vertical at x = 0, from y = 1 to y = 9 (shared by blocks A and B)
    l_mid_lo = geo.addLine(p3, pm_mid)
    l_mid_hi = geo.addLine(pm_mid, p7)

    # Upper outlet (x = 50, y from 1 to 9)
    l_outup_lo = geo.addLine(p4, pm_out)
    l_outup_hi = geo.addLine(pm_out, p8)

    # Step (x = 0, y from 0 to 1) and lower outlet (x = 50, y from 0 to 1)
    l_step_lo = geo.addLine(p9, ps_mid)
    l_step_hi = geo.addLine(ps_mid, p3)
    l_outdn_lo = geo.addLine(p10, ps_out)
    l_outdn_hi = geo.addLine(ps_out, p4)

    # Downstream floor (y = 0)
    l_bot_down = geo.addLine(p9, p10)

    # --- Surfaces (3 blocks) -----------------------------------------------
    # A: upstream of the step, x from -130 to 0, y from 1 to 9
    loop_a = geo.addCurveLoop([l_bot_sym, l_bot_wall, l_mid_lo, l_mid_hi,
                               l_top_wall_up, l_top_sym, -l_inlet_hi, -l_inlet_lo])
    surf_a = geo.addPlaneSurface([loop_a])

    # B: downstream of the step above y = 1, x from 0 to 50, y from 1 to 9
    loop_b = geo.addCurveLoop([l_internal, l_outup_lo, l_outup_hi, l_top_wall_down,
                               -l_mid_hi, -l_mid_lo])
    surf_b = geo.addPlaneSurface([loop_b])

    # C: downstream of the step below y = 1, x from 0 to 50, y from 0 to 1
    loop_c = geo.addCurveLoop([l_bot_down, l_outdn_lo, l_outdn_hi, -l_internal,
                               -l_step_hi, -l_step_lo])
    surf_c = geo.addPlaneSurface([loop_c])

    geo.synchronize()

    # --- Divisions (transfinite): nodes = intervals + 1 --------------------
    def tc(curve, n_intervals, kind="Progression", coef=1.0):
        gmsh.model.geo.mesh.setTransfiniteCurve(curve, n_intervals + 1, kind, coef)

    # Verticals above y = 1: from the bottom wall towards the centre the elements
    # grow by r_up, from the centre to the top wall they shrink by 1/r_up.
    for lo, hi in ((l_inlet_lo, l_inlet_hi), (l_mid_lo, l_mid_hi), (l_outup_lo, l_outup_hi)):
        tc(lo, n_up_half, coef=r_up)
        tc(hi, n_up_half, coef=1.0 / r_up)

    # Verticals below y = 1 (step and lower outlet): first cell DY_WALL both
    # at the floor (y = 0) and at the top (y = 1), where it joins block B.
    for lo, hi in ((l_step_lo, l_step_hi), (l_outdn_lo, l_outdn_hi)):
        tc(lo, n_dn_half, coef=r_dn)
        tc(hi, n_dn_half, coef=1.0 / r_dn)

    # Horizontals upstream of the step (wall y = 1 and wall y = 9): uniform
    tc(l_bot_sym, N_sym)
    tc(l_bot_wall, N_wall)
    tc(l_top_sym, N_sym)
    tc(l_top_wall_up, N_wall)

    # Horizontals downstream of the step (y = 0, internal y = 1, y = 9): uniform
    for c in (l_bot_down, l_internal, l_top_wall_down):
        tc(c, N_orizzontali_outlet)

    # Transfinite surfaces (the 4 corners are needed because the blocks have more than 4 curves)
    ts = gmsh.model.geo.mesh.setTransfiniteSurface
    ts(surf_a, "Left", [p1, p3, p7, p5])
    ts(surf_b, "Left", [p3, p4, p8, p7])
    ts(surf_c, "Left", [p9, p10, p4, p3])

    # Quadrilateral cells (or hexahedra, after the extrusion)
    surfaces = (surf_a, surf_b, surf_c)
    for s in surfaces:
        gmsh.model.geo.mesh.setRecombine(2, s)

    # Real boundaries; the internal line and the internal verticals are left out.
    boundaries = {
        "inlet": [l_inlet_lo, l_inlet_hi],
        "outlet": [l_outup_lo, l_outup_hi, l_outdn_lo, l_outdn_hi],
        "symmetry": [l_bot_sym, l_top_sym],
        "wall_bottom_upstream": [l_bot_wall],
        "step": [l_step_lo, l_step_hi],
        "wall_bottom_downstream": [l_bot_down],
        "wall_top": [l_top_wall_up, l_top_wall_down],
    }

    info = {
        "r_up": r_up, "r_dn": r_dn,
        "inlet_curves": (l_inlet_lo, l_inlet_hi),
        "step_curves": (l_step_lo, l_step_hi),
    }
    return surfaces, boundaries, info


def add_physical_groups_2d(surfaces, boundaries):
    gmsh.model.geo.synchronize()
    pg, sn = gmsh.model.addPhysicalGroup, gmsh.model.setPhysicalName
    for name, curves in boundaries.items():
        sn(1, pg(1, curves), name)
    sn(2, pg(2, list(surfaces)), "fluid")


def extrude_and_group(surfaces, boundaries):
    """Extrude the 3 blocks together, preserving the shared interfaces."""
    geo = gmsh.model.geo
    dz = DZ_OVER_H * H

    # IMPORTANT: we extrude all the surfaces with a single operation.
    # Extruding them separately risks producing duplicate faces/nodes
    # on the A-B and B-C interfaces, even with an apparently consistent 2D mesh.
    out = geo.extrude(
        [(2, s) for s in surfaces],
        0, 0, dz,
        [1], [], True
    )
    geo.synchronize()

    out_surfaces = [tag for dim, tag in out if dim == 2]
    volumes = [tag for dim, tag in out if dim == 3]

    tol = max(1e-10, abs(dz) * 1e-8)

    # The top surfaces have z = dz; the lateral ones span
    # the whole thickness, from z = 0 to z = dz.
    top_surfaces = []
    lateral = []

    for tag in out_surfaces:
        bb = gmsh.model.getBoundingBox(2, tag)
        zmin, zmax = bb[2], bb[5]

        if abs(zmin - dz) < tol and abs(zmax - dz) < tol:
            top_surfaces.append(tag)
        elif abs(zmin) < tol and abs(zmax - dz) < tol:
            lateral.append(tag)

    if len(volumes) != len(surfaces):
        raise RuntimeError(
            f"Estrusione non riuscita: attesi {len(surfaces)} volumi, "
            f"ottenuti {len(volumes)}."
        )

    if len(top_surfaces) != len(surfaces):
        raise RuntimeError(
            f"Estrusione non riuscita: attese {len(surfaces)} superfici "
            f"superiori, ottenute {len(top_surfaces)}."
        )

    # Match the lateral surfaces to the original physical curves through
    # the bounding box in the xy plane. The internal interfaces have no name
    # and therefore do not become boundary patches.
    key = lambda bb: tuple(round(v, 9) for v in (bb[0], bb[1], bb[3], bb[4]))
    name_of = {}

    for name, curves in boundaries.items():
        for c in curves:
            name_of[key(gmsh.model.getBoundingBox(1, c))] = name

    groups = {}
    for tag in lateral:
        name = name_of.get(key(gmsh.model.getBoundingBox(2, tag)))
        if name:
            groups.setdefault(name, []).append(tag)

    pg, sn = gmsh.model.addPhysicalGroup, gmsh.model.setPhysicalName

    for name, tags in groups.items():
        sn(2, pg(2, tags), name)

    # Front and back: initial surfaces at z=0 and top surfaces at z=dz.
    sn(2, pg(2, list(surfaces) + top_surfaces), "frontAndBack")
    sn(3, pg(3, volumes), "fluid")


def curve_spacings(curves):
    """Cell heights along a vertical line (from minimum to maximum y)."""
    ys = set()
    for c in curves:
        _, coords, _ = gmsh.model.mesh.getNodes(1, c, True)
        ys.update(round(y, 12) for y in list(coords)[1::3])
    ys = sorted(ys)
    return [b - a for a, b in zip(ys, ys[1:])]


def report(info):
    print("\n--- Controllo y+ (stima a priori, al centro della prima cella) ---")
    print(f"nu = {NU:.3e} m2/s | u_tau = {U_TAU:.3f} m/s (Cf = {CF_REF}) | "
          f"prima cella progettata = {DY_WALL:.2e} m = {DY_WALL / H:.2e} H")
    for label, curves in (("x = -130 (y = 1 -> 9)", info["inlet_curves"]),
                          ("step x = 0 (y = 0 -> 1)", info["step_curves"])):
        d = curve_spacings(curves)
        growth = max(max(b / a, a / b) for a, b in zip(d, d[1:]))
        yp = lambda dy: 0.5 * dy * U_TAU / NU
        print(f"{label}: prima cella in basso {d[0]:.2e} m (y+ = {yp(d[0]):.2f}), "
              f"in alto {d[-1]:.2e} m (y+ = {yp(d[-1]):.2f}), crescita max {growth:.3f}")
    print(f"rapporti di crescita: sopra y = 1 -> {info['r_up']:.3f}, sotto -> {info['r_dn']:.3f}")




def run_openfoam_check(msh_path):
    """Copy the mesh into the OpenFOAM WSL case, convert it and run checkMesh."""

    print("\n" + "=" * 72)
    print("IMPORTAZIONE IN OPENFOAM (WSL) E CHECKMESH")
    print("=" * 72)

    msh_path = os.path.abspath(msh_path)

    if not os.path.isfile(msh_path):
        print(f"ERRORE: file MSH non trovato: {msh_path}")
        return

    try:
        # IMPORTANT:
        # We convert the Windows backslashes to slashes before passing
        # the path to WSL, to avoid them being interpreted
        # as escape characters.
        msh_path_wsl_input = msh_path.replace("\\", "/")

        result = subprocess.run(
            ["wsl.exe", "--", "wslpath", "-a", msh_path_wsl_input],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        msh_wsl = result.stdout.strip()

        if not msh_wsl:
            print("ERRORE: wslpath non ha restituito alcun percorso.")
            return

    except FileNotFoundError:
        print("ERRORE: wsl.exe non trovato. Verifica che WSL sia installato.")
        return

    except subprocess.CalledProcessError as exc:
        print("ERRORE durante la conversione del percorso Windows in percorso WSL:")
        print(exc.stderr or exc)
        return

    case_dir = shlex.quote(WSL_CASE_DIR)
    mesh_file = shlex.quote(msh_wsl)

    command = (
        "source /usr/lib/openfoam/openfoam2312/etc/bashrc; "
        "set -e; "
        "cd " + case_dir + "; "
        'echo "Caso OpenFOAM: $PWD"; '

        'command -v gmshToFoam >/dev/null || '
        '{ echo "ERRORE: gmshToFoam non trovato."; exit 127; }; '

        'command -v checkMesh >/dev/null || '
        '{ echo "ERRORE: checkMesh non trovato."; exit 127; }; '

                'echo "\\n--- Rimozione della mesh precedente ---"; '
        "rm -rf constant/polyMesh; "

        'echo "\\n--- Conversione con gmshToFoam ---"; '
        "gmshToFoam " + mesh_file + " && "

        'echo "\\n--- Configurazione tipi delle patch ---"; '
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "import re\n"
        "p = Path('constant/polyMesh/boundary')\n"
        "text = p.read_text()\n"
        "types = {\n"
        "    'inlet': 'patch',\n"
        "    'outlet': 'patch',\n"
        "    'symmetry': 'symmetry',\n"
        "    'wall_bottom_upstream': 'wall',\n"
        "    'wall_top': 'wall',\n"
        "    'wall_bottom_downstream': 'wall',\n"
        "    'step': 'wall',\n"
        "    'frontAndBack': 'empty',\n"
        "}\n"
        "for name, typ in types.items():\n"
        "    pattern = rf'(?m)^(\\\\s*{re.escape(name)}\\\\s*\\\\n\\\\s*\\\\{{)(.*?)(^\\\\s*\\\\}})'\n"
        "    m = re.search(pattern, text, re.DOTALL)\n"
        "    if not m:\n"
        "        raise SystemExit(f'ERRORE: patch {name} non trovata')\n"
        "    body = re.sub(r'\\\\btype\\\\s+\\\\w+\\\\s*;', f'type            {typ};', m.group(2), count=1)\n"
        "    if not re.search(r'\\\\btype\\\\s+\\\\w+\\\\s*;', m.group(2)):\n"
        "        body = f'\\\\n        type            {typ};' + body\n"
        "    text = text[:m.start(2)] + body + text[m.end(2):]\n"
        "p.write_text(text)\n"
        "print('Tutte le patch sono state configurate.')\n"
        "PY\n"
    )

    print(f"Mesh Windows: {msh_path}")
    print(f"Mesh WSL:     {msh_wsl}")
    print(f"Caso WSL:     {WSL_CASE_DIR}\n")

    try:
        completed = subprocess.run(
            ["wsl.exe", "--", "bash", "-lc", command],
            check=False
        )

        print("\n" + "=" * 72)

        if completed.returncode == 0:
            print(
                "PROCEDURA COMPLETATA: "
                "gmshToFoam e checkMesh terminati correttamente."
            )
        else:
            print(
                "PROCEDURA INTERROTTA: "
                f"WSL/OpenFOAM ha restituito il codice {completed.returncode}."
            )

        print("=" * 72)

    except FileNotFoundError:
        print("ERRORE: impossibile avviare wsl.exe.")




def main():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("backward_facing_step")

    surfaces, boundaries, info = build_geometry()
    if ESTRUDI:
        extrude_and_group(surfaces, boundaries)
        gmsh.model.mesh.generate(3)
    else:
        add_physical_groups_2d(surfaces, boundaries)
        gmsh.model.mesh.generate(2)

    types, tags, _ = gmsh.model.mesh.getElements(3 if ESTRUDI else 2)
    print("\nTipi di elemento (3 = quad, 5 = esaedro):", [int(t) for t in types])
    print("Celle:", sum(len(t) for t in tags), "| Nodi:", len(gmsh.model.mesh.getNodes()[0]))
    print(f"Intervalli a monte: {N_sym} (simmetria) + {N_wall} (parete) = {N_orizzontali_inlet}")
    report(info)

    if SALVA_MSH:
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(NOME_MSH)    # with physical groups defined, only the physical groups are saved
        print(f"\nMesh salvata in {os.path.abspath(NOME_MSH)} (MSH 2.2)")

    if "-nogui" not in sys.argv:
        gmsh.option.setNumber("Geometry.Points", 1)
        gmsh.option.setNumber("Geometry.PointNumbers", 0)
        gmsh.option.setNumber("Mesh.SurfaceFaces", 0)   # only the cell edges
        gmsh.option.setNumber("Mesh.Lines", 1)
        gmsh.fltk.run()

    gmsh.finalize()

    # The Gmsh window has been closed (or -nogui was used):
    # automatically start the conversion and check in OpenFOAM/WSL.
    if ESEGUI_CHECKMESH:
        run_openfoam_check(NOME_MSH)


if __name__ == "__main__":
    main()