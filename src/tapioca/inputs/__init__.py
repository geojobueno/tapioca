"""
Input builders for Mandyoc scenarios.

- MandyocBuilder, VelocityFieldBuilder and TemperatureFieldBuilder: generic builders of the
  interfaces, material fields, velocity and temperature of a scenario, used from scripts;
- svg: scenarios drawn in SVG files (e.g. made in Inkscape), see SVGScenario.
"""

from ..scenClasses import MandyocBuilder, VelocityFieldBuilder, TemperatureFieldBuilder
from . import svg

__all__ = ["MandyocBuilder", "VelocityFieldBuilder", "TemperatureFieldBuilder", "svg"]
