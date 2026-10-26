# Scenarios from SVG files

One example of each numerical experiment of `tapioca.inputs.SVGScenario`. Open the SVGs in Inkscape
(*Object > Object Properties*) to see the label, title and description of each unit, and the description of
the `domain` rectangle (in the `dominio` layer) with the domain, the resolution and the experiment.

| Folder | Scenario | Experiment |
|---|---|---|
| `rifting` | continental lithosphere (155 km) with a weak seed | rifting, 1 cm/yr, magmatism on |
| `compression` | oceanic lithosphere - continental margin ramp - non-cratonic lithosphere - craton, with a dipping weak zone | compression, the oceanic plate pushed from the left at 1 cm/yr |
| `accordion` | non-cratonic lithosphere between two cratons, with a weak seed (M300) | accordion (extension, convergence, quiescence and extension) |
| `quiescence` | cratonic keel with a mobile belt in a 4000 km domain | quiescence, 1000 Myr |

To create the input files of Mandyoc:

```bash
cd rifting
tapioca-svg rifting.svg -o inputs
```

The settings of the SVG can be changed on the command line, e.g. `tapioca-svg accordion.svg --quiescence-time 300 --dx 1 --dz 1`.

In rifting and accordion, `--moving-boundary symmetric` puts the reference frame at the centre of the domain:
both boundaries move outwards with half of the velocity, e.g. `tapioca-svg rifting.svg -o inputs --moving-boundary symmetric`.
In compression, the opposite boundary of the moving one is fixed.

The `rifting` and `quiescence` SVGs have the line `preset_initial_temperature = ../preliminary` commented
out: uncomment it (or use `--preset-initial-temperature`) with the folder of a preliminary scenario to start
from its final thermal state instead of the conductive model.
