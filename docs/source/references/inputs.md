# Input builders

Input builders are in the subpackage `tapioca.inputs`. They create the input files of Mandyoc
(`interfaces.txt`, `param.txt`, `input_temperature_0.txt`, `input_velocity_0.txt`, ...) from a scenario
drawn in an SVG file. See [Scenarios from SVG files](../user_guide/svg_scenarios.md) for how to draw the scenario.

---

## Scenario builder

The `SVGScenario` class reads the SVG, builds the layers, the initial temperature and the velocity
boundary conditions of the numerical experiment (rifting, compression, accordion or quiescence), and writes
the input files and figures. The source code is in `inputs/scenario.py`.

```{eval-rst}
.. currentmodule:: tapioca.inputs.scenario

.. autosummary::
   :toctree: ../generated/

   SVGScenario
   SVGScenario.write_all
   SVGScenario.write_scripts
   SVGScenario.build_temperature
   SVGScenario.build_velocity
   SVGScenario.param_text
   SVGScenario.scale_bcv
   SVGScenario.infos
   SVGScenario.plot_setup
   SVGScenario.plot_temperature
   SVGScenario.yield_strength_envelope
   read_preset_temperature
   adapt_preset_temperature
```

Example:
```python
from tapioca.inputs import SVGScenario

scenario = SVGScenario("rifting.svg", regime="accordion", quiescence_time=300)
scenario.write_all("my_scenario", particles_per_element=100)  # keyword arguments change param.txt
```

The same from the command line (installed with the package):
```bash
tapioca-svg rifting.svg -o my_scenario --regime accordion --quiescence-time 300
```

The run script of Mandyoc is written for the system that runs tapioca (`run_linux.sh` or `run_mac.sh`).
Another system, the number of MPI processes and the options of Mandyoc may be given:
```bash
tapioca-svg rifting.svg -o my_scenario --system linux --ncores 20 --mandyoc-options="-seed 0,5,8 -strain_seed 0.0,1.0,1.0"
```

---

## Geometry and materials from SVG files

Functions to read the domain and the units of the SVG and build the interfaces. The source code is in `inputs/svg.py`.

```{eval-rst}
.. currentmodule:: tapioca.inputs.svg

.. autosummary::
   :toctree: ../generated/

   read_domain
   read_units
   layers_from_svg
   write_interfaces
   plot_layers
   SVGInputError
```

Only the interfaces from the command line:
```bash
tapioca-svg-interfaces scenario.svg -o interfaces.txt --plot layers.png
```

---

## Materials

Layers of a Mandyoc scenario and their rheologies. The source code is in `inputs/materials.py`.

```{eval-rst}
.. currentmodule:: tapioca.inputs.materials

.. autosummary::
   :toctree: ../generated/

   MandyocLayer
   LithologicalUnit
   WetOlivine
   DryOlivine
   WetQuartz
   DryQuartz
   Basalt
   Plagioclase
   Air
   get_rheology
```
