# Backward-Facing Step: OpenFOAM validation (NASA TMR 2DBFS)

A 2D CFD validation of turbulent flow over a backward-facing step, using OpenFOAM v2312 (ESI) and the steady solver `simpleFoam`. Two turbulence models are compared, k-omega SST and Spalart-Allmaras. The results are checked against the Driver & Seegmiller experiment and the NASA TMR reference CFD (CFL3D).

Everything is driven by Python scripts: mesh, boundary conditions, solver settings, the run itself, post-processing and plots. The full write-up is in [`report/`](report/).

<!-- Add an image here, e.g. the Cf plot: ![Cf along the lower wall](results/05_cf.png) -->

## Main results

| | SA | k-omega SST | Experiment |
| --- | --- | --- | --- |
| Reattachment x/H | 6.09 | 6.48 | 6.26 ± 0.10 |
| Grid convergence (GCI, reattachment) | 0.47% | 0.16-0.39% | |

Both models are close to their TMR counterparts. The grid study shows that the SST difference from the experiment (about +0.22H) is a model effect, not a mesh effect. See the report for Cf, Cp, velocity and Reynolds-stress comparisons.

## Requirements

- OpenFOAM v2312 (ESI), installed in WSL with Ubuntu
- Windows with Python 3 (the scripts also run from a Linux terminal)
- `gmsh` (Python API), `numpy`, `matplotlib`

```
pip install gmsh numpy matplotlib
```

## How to run

Put all scripts in one folder and run the master script:

```
python bfs_MASTER.py --model SpalartAllmaras --stages 1:6000,3:12000
python bfs_MASTER.py --model kOmegaSST --stages 1:6000,3:12000
```

Useful options:

```
--only-setup     mesh and dictionaries only
--skip-mesh      do not rebuild the mesh
--skip-run       do not launch simpleFoam
--only-plots     post-processing and plots only
--no-clean       keep previous time directories
--gui-mesh       open the Gmsh window
```

The `--stages` option is a list of `fvSchemes stage:last iteration`. Stage 1 is all upwind, stage 3 is linearUpwind for U and limitedLinear for the turbulence variables. The final times must be multiples of `writeInterval`.

The final-run mesh uses 400 intervals downstream of the step. Set `N_orizzontali_outlet = 400` in `bfs_mesh.py` before running (140 / 60 wall-normal intervals above / below the step).

## Paths

The scripts use the case folder `/home/<user>/OpenFOAM/<user>-v2312/run/BFS_TMR` and the WSL distribution `Ubuntu` as defaults. Change `CASE_LINUX` and `DISTRO` (or the matching command-line options) to match your machine.

## Experimental and TMR data

The data are not included. Download them from the NASA TMR page for this case, https://turbmodels.larc.nasa.gov/backstep_val.html, and place them like this:

```
dati_sperimentali/
    cf.exp.dat
    cp.expnew.dat
    profiles.exp.dat
dati_tmr/
    *_cf_sst.dat, *_cp_sst.dat, *_vel_sst.dat, *_uv_sst.dat
```

Without these files the plots are still produced, but without the experimental and TMR curves.

## Repository layout

```
scripts/        Python scripts (mesh, boundary, controlDict, fvSolution,
                fvSchemes, master, plots, GCI)
case_snapshot/  system/, 0/ and constant/ of the final runs
results/        riepilogo.txt and the PNG plots
report/         PDF report and figures
```

## Scripts

| Script | What it does |
| --- | --- |
| `bfs_mesh.py` | Gmsh multi-block mesh, gmshToFoam, patch types, checkMesh |
| `bfs_boundary.py` | 0/ fields, transportProperties, turbulenceProperties |
| `bfs_controlDict.py` | run control, probes, solverInfo |
| `bfs_fvSolution.py` | linear solvers, SIMPLEC, relaxation |
| `bfs_fvSchemes.py` | discretisation schemes (stages 1-3) |
| `bfs_MASTER.py` | runs the whole pipeline |
| `bfs_plots.py` | residuals, probes, Cf, Cp, profiles, summary table |
| `bfs_gci.py` | Richardson extrapolation and GCI |

## References

- NASA Turbulence Modeling Resource, 2D Backward-Facing Step validation case.
- Driver, D. M. and Seegmiller, H. L., AIAA Journal 23(2), 1985.
- Celik et al., Journal of Fluids Engineering 130(7), 2008 (GCI procedure).

## License

MIT, see `LICENSE`.
