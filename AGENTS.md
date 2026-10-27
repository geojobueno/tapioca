# tapioca

Python package (`tapioca-mandyoc`) to build inputs for and post-process outputs of the Mandyoc geodynamic code, based on xarray and DataTrees.

## Layout

- `src/tapioca/scenClasses.py`:
  - `MandyocScen`: reads Mandyoc outputs.
  - `MandyocBuilder`, `VelocityFieldBuilder`, `TemperatureFieldBuilder`: the generic input builders.
- `src/tapioca/_variables.py`: constants, `DEFAULT_MATERIAL`/`DEFAULT_MATERIALS`, `PARAMS_TYPES`.
- `src/tapioca/_aux_functions.py`: helpers (param reading, solvers, yield strength).
- `src/tapioca/post_processing.py`, `src/tapioca/mandyocPlotter.py`: post-processing and plotting.
- `src/tapioca/inputs/`: re-exports the generic builders. It is the generic part, so there is no `inputs.generic`.
- `src/tapioca/inputs/svg/`: SVG scenarios (Inkscape drawings): `reader`, `materials`, `scenario` (`SVGScenario`), `cli`.
  - SVG scenarios are built on the generic builders: `SVGScenario.builder` is a `MandyocBuilder`, and every input file is exported through it.
  - Code specific to SVG belongs here, not in the generic builders.
- `src/postrunning/`: shell scripts copied next to the scenario inputs.
- `examples/svg_scenarios/`: example SVGs. Every layer has a reference thickness `h`, except the bottom one (the asthenosphere), which fills the rest of the domain.
- `tests/`: pytest suite.

## Conventions

- `MandyocBuilder` interfaces: ID `-1` is the top layer (sticky air). Interface `i` is the top of layer `i` and carries its material.
- `interfaces.txt` holds the 13 material parameters (`MATERIAL_PARAMETERS`).
- The weak seed layer has `weakening_seed = 1`; every other layer has `-1`.
- The default materials come from `DEFAULT_MATERIALS`. The example SVGs write their own flow-law values (A, n, Q, V) in the layer descriptions.
- Code, docstrings, commits and docs are written in English.

## Working with other branches

Several branches (for example `particles`, `inputs`, `julia_in_chunks`) work on the shared core files at the same time:
`_aux_functions.py`, `_variables.py`, `mandyocPlotter.py`, `post_processing.py`, `scenClasses.py`.

- Keep edits to these files minimal. Prefer adding functions or methods over rewriting existing ones.
- Avoid renames, reformatting and cosmetic edits in these files.
- Before changing them, run `git fetch origin` and check what other branches changed.
- Before a commit or PR, simulate the merges, for example `git merge-tree --write-tree HEAD origin/<branch>`, and report conflicts.
- A fix to code that another branch is rewriting may belong in that branch instead.

## Commands

```bash
conda env create -f environment.yml   # environment "tapioca"
pip install -e ".[test]"
pytest                                 # about 10 s
```

Mandyoc runs are done on the group's cluster, Hypatia.
