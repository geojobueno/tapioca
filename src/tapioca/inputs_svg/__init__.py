"""
Input builders for Mandyoc scenarios.

- materials: Mandyoc layers (MandyocLayer) and their rheologies;
- svg: geometry and materials of a scenario drawn in an SVG file (e.g. made in Inkscape);
- scenario: SVGScenario, which builds all the input files of a scenario from an SVG
  (interfaces, parameters, initial temperature and velocity for rifting, compression,
  accordion and quiescence experiments).
"""

from .materials import *
from .svg import *
from .scenario import *
