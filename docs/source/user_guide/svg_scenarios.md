# Scenarios from SVG files

The geometry of a Mandyoc scenario can be drawn in an SVG file (e.g. in [Inkscape](https://inkscape.org))
and turned into the input files of Mandyoc with `tapioca.inputs`:

```bash
tapioca-svg scenario.svg -o my_scenario
```

or, in Python:

```python
from tapioca.inputs import SVGScenario
SVGScenario("scenario.svg").write_all("my_scenario")
```

The output folder receives `interfaces.txt`, `param.txt`, `input_temperature_0.txt`,
`input_velocity_0.txt`, `vel_bc.txt`, `velz_bc.txt`, `scale_bcv.txt` (accordion), `infos_<folder>.txt`,
a copy of the SVG and the figures `numerical_setup.png` and `initial_temperature.png`.
Examples are in `examples/svg_scenarios`.

It also receives the scripts to run the scenario (`--no-scripts` skips them):

- `run_linux.sh` or `run_mac.sh` (`--system`, default: the system that runs tapioca): runs Mandyoc with
  `$PETSC_DIR/$PETSC_ARCH/bin/mpirun` and then moves the outputs to one folder per variable with
  `0_organize_outputs.sh`. On the Mac, `caffeinate` keeps the computer awake during the run. `PETSC_DIR`,
  `PETSC_ARCH` and `MANDYOC` (the executable) set in the environment override the values of the script;
  `--ncores` and `--mandyoc-options` set the number of MPI processes and the options of Mandyoc;
- `0_organize_outputs.sh`: copy of `src/postrunning/0_organize_outputs.sh`;
- `zipper.sh`: zips the inputs and the outputs in `<folder>.zip`, skipping the files already in the zip,
  so it can be run again to add new outputs.

## Units of rock

Each filled shape is a body of rock described by its object properties (Inkscape: *Object > Object Properties*):

- **Label**: name of the unit (`layer_label`). Shapes with the same label are the same unit.
- **Title**: name of the unit in the legend of the figures (optional).
- **Description**: the rheology in the first line (`wet olivine`, `dry olivine`, `wet quartz`, `dry quartz`,
  `basalt`, `plagioclase` or `air`) and the properties in the next lines:

```
wet quartz
C = 1.0
rho = 2800.0
H = 0.19e-6 / 2800.0
h = 10 km
```

`C` (viscosity scale factor), `rho` (density) and `H` (radiogenic heat production, W/kg) are required.
`A`, `n`, `Q` and `V` are optional and override the values of the rheology. A shape without description
inherits the one of the group (e.g. an Inkscape layer) that contains it; shapes without any description
(annotations, arrows) are ignored. The fill colour is used in the figures.

Shapes are painted in document order, as Inkscape renders them: a simple way to draw is a rectangle of
asthenosphere over the whole domain and the other units on top of it. Two labels are special:
`asthenosphere` (the bottom unit) and `weak_seed` (weak zones, which receive an initial strain of 1.0).
The top unit must be the sticky air.

## Thickness of the units (`h`)

The drawing does not need to be in vertical scale: `h` gives the reference thickness of a unit.

- **Layers** (units that reach both sides of the domain): their most common drawn thickness becomes `h`
  and the same factor is applied in every column, so thinned regions keep their drawn proportion.
- **Bodies** (seeds, basins, weak zones, keels, ...): without `h` they follow the scaling of the unit
  they are drawn over; with `h` they get that thickness and the unit around them absorbs the difference.
- The **bottom unit** (asthenosphere) fills the rest of the domain, so it can not have `h`.

## Domain and numerical experiment

A shape labelled `setup` (e.g. a rectangle without fill in its own Inkscape layer, named `domain` in the
examples, with the units in the layer `lithologies`) gives the model domain:
its bounding box is mapped to the domain and its description gives the dimensions, the resolution and,
optionally, the numerical experiment:

```
Lx = 1600 km
Lz = 400 km
dx = 2 km
dz = 2 km
regime = accordion
velocity = 1 cm/yr
moving_boundary = right
quiescence_time = 300 Myr
time_max = 500 Myr
magmatism = off
magmatism_extraction = off
magmatism_Ts0 = 1080
preset_initial_temperature = ../preliminary
```

Older SVGs with the shape labelled `domain` instead of `setup` are still read, with a warning.

Every setting can also be given to `SVGScenario` or on the command line (e.g. `--regime`, `--dx`), which
overrides the SVG. Defaults:

| Setting | Default | |
|---|---|---|
| `regime` | `rifting` | `rifting` (extension), `compression` (convergence), `accordion` or `quiescence` |
| `velocity` | 1 cm/yr | velocity of the moving boundary (with `symmetric`, the sum of both boundaries) |
| `moving_boundary` | `right` | `left`, `right` or `symmetric` (rifting and accordion) |
| `quiescence_time` | 0 Myr | accordion and quiescence regimes |
| `time_max` | 60 Myr | rifting and compression; accordion: end of the second extension; quiescence: `quiescence_time` (or 1000 Myr) |
| `magmatism`, `magmatism_extraction` | `off` | with both `on`, a basalt layer receives the extracted melt |
| `magmatism_Ts0` | 1080 °C | |
| `preset_initial_temperature` | none | folder of a preliminary scenario (see below) |

The accordion regime (as in the M300 experiments) scales the boundary velocity in time (`scale_bcv.txt`):
extension until 25 Myr, convergence until 75 Myr, quiescence during `quiescence_time` and extension again.

In the rifting and accordion regimes, the velocity is constant from the surface down to 20 km below the
thickest lithosphere of the domain; in the compression regime, down to 20 km below the lithosphere at the
moving boundary (e.g. an oceanic plate pushed towards a continent), and the opposite boundary is fixed.

With `moving_boundary = symmetric` (rifting and accordion), the reference frame is at the centre of the
domain: both boundaries move outwards with `velocity`/2, each with its own return flow below the constant
velocity, so that the integral of the inflow is zero on each boundary.

## Initial temperature

Without `preset_initial_temperature`, the initial temperature comes from a 1D conductive model solved for
each column (lithosphere from the surface to the asthenosphere, radiogenic heat of the units).

With `preset_initial_temperature` (folder with `_output_temperature.nc`, or with `param.txt` and
`temperature_*.txt`), the initial temperature is the horizontal mean of the final temperature of that
preliminary scenario, adapted to the lithosphere of each column: the columns with the lithosphere of the
preliminary scenario (read from its `interfaces.txt`) keep the profile; the others (keels, thinned or oceanic
lithosphere) keep it in the crust, get a linear geotherm in the lithospheric mantle down to the base of the
lithosphere and the asthenospheric profile below it.
