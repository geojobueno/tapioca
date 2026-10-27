"""
Input builders for Mandyoc scenarios.

- MandyocBuilder, VelocityFieldBuilder and TemperatureFieldBuilder: generic builders of the
  interfaces, material fields, velocity and temperature of a scenario, used from scripts;
- svg: scenarios drawn in SVG files (e.g. made in Inkscape), see SVGScenario.
"""

# getting the same version of the whole module
from ..__init__ import __version__

# getting the generic input classes
from .generic_input_classes import * 

from . import svg

__all__ = ["MandyocBuilder", "VelocityFieldBuilder", "TemperatureFieldBuilder", "svg"]