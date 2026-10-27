"""
Input builders for Mandyoc scenarios drawn in SVG files (e.g. made in Inkscape).

- materials: Mandyoc layers (MandyocLayer) and their rheologies;
- reader: geometry and materials of a scenario drawn in an SVG file;
- scenario: SVGScenario, which builds all the input files of a scenario from an SVG
  (interfaces, parameters, initial temperature and velocity for rifting, compression,
  accordion and quiescence experiments).
"""

from .materials import *
from .reader import *
from .scenario import *
