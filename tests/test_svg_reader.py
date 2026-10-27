"""Tests of the reader of SVG scenarios (tapioca.inputs.svg.reader)."""

import re

import numpy as np
import pytest

from tapioca.inputs.svg import SVGInputError, layers_from_svg, layers_to_builder, read_domain, read_units, write_interfaces
from tapioca.inputs.svg.reader import eval_number, parse_description

from conftest import SVG_SCENARIOS, quiet, svg_path


def _layers(svg, Nx=161):
    domain = read_domain(str(svg))
    with quiet():
        return domain, layers_from_svg(str(svg), domain["Lx"], domain["Lz"], Nx, domain_id=domain["id"],
                                       min_thickness=domain.get("min_thickness"), verbose=False)


# Descriptions ##################################################################

def test_eval_number_accepts_only_arithmetic():
    assert eval_number("1.67e-6 / 2700.0") == pytest.approx(1.67e-6 / 2700.0)
    assert eval_number("2**3 - 1") == 7.0
    with pytest.raises(ValueError):
        eval_number("__import__('os')")


def test_parse_description():
    rheology, props, thickness = parse_description("wet quartz\nC = 1.0\nrho = 2700\nH = 1e-10\nh = 25 km\ncohesion_max = 3e7", "test")
    assert rheology.name == "wet_quartz"
    assert props == {"effective_viscosity_scale_factor": 1.0, "density": 2700.0,
                     "radiogenic_heat_production": 1e-10, "cohesion_max": 3e7}
    assert thickness == 25e3


@pytest.mark.parametrize("text, message", [
    ("C = 1.0\nrho = 2700\nH = 0", "rheology"),
    ("wet quartz\nC = 1.0\nH = 0", "missing rho"),
    ("wet quartz\nC = 1.0\nrho = 2700\nH = 0\nviscosity = 1e21", "unknown property"),
    ("granite\nC = 1.0\nrho = 2700\nH = 0", "Unknown rheology"),
])
def test_parse_description_errors(text, message):
    with pytest.raises(SVGInputError, match=message):
        parse_description(text, "test")


# Examples ######################################################################

@pytest.mark.parametrize("name", SVG_SCENARIOS)
def test_example_domain(name):
    domain = read_domain(str(svg_path(name)))
    assert domain is not None, "the example must have a shape labelled 'setup'"
    assert domain["Lx"] > 0 and domain["Lz"] > 0 and domain["Nx"] > 1 and domain["Nz"] > 1
    regime, moving_boundary = SVG_SCENARIOS[name]
    assert domain["experiment"]["regime"] == regime
    assert domain["experiment"].get("moving_boundary", "right") == moving_boundary


@pytest.mark.parametrize("name", SVG_SCENARIOS)
def test_example_units_have_reference_thickness(name):
    '''Every unit of the examples gives h, except the bottom one (asthenosphere), which fills the rest of the domain.'''
    _, units = read_units(str(svg_path(name)))
    _, layers = _layers(svg_path(name))
    bottom = layers[0].layer_label
    without_h = [label for label, unit in units.items() if unit["thickness"] is None and label != bottom]
    assert without_h == []
    assert units[bottom]["thickness"] is None


@pytest.mark.parametrize("name", SVG_SCENARIOS)
def test_example_layers(name):
    domain, layers = _layers(svg_path(name))
    assert layers[0].layer_label == "asthenosphere"
    assert layers[-1].rheology_name == "air"
    bases = np.array([layer.interface for layer in layers])
    assert np.all(np.diff(bases, axis=0) <= 0), "the layers are stacked from the bottom to the top"
    assert np.all(bases[1:] >= 0) and np.all(bases[1:] <= domain["Lz"])


def test_h_sets_the_thickness_of_a_layer(tmp_path):
    '''Changing h of the upper crust of the rifting example changes its thickness, but not the drawing.'''
    text = svg_path("rifting").read_text()
    assert "h = 25 km" in text
    svg = tmp_path / "rifting.svg"
    svg.write_text(text.replace("h = 25 km", "h = 30 km", 1))
    _, layers = _layers(svg)
    labels = [layer.layer_label for layer in layers]
    i = labels.index("upper_crust")
    thickness = layers[i].interface - layers[i + 1].interface
    assert np.median(thickness) == pytest.approx(30e3)


def test_h_of_the_bottom_unit_is_an_error(tmp_path):
    text = svg_path("rifting").read_text()
    svg = tmp_path / "rifting.svg"
    svg.write_text(re.sub(r"(rho = 3378\.0\nH = 0\.0)", r"\1\nh = 100 km", text, count=1))
    with pytest.raises(SVGInputError, match="bottom layer"):
        _layers(svg)


# Builder and interfaces.txt ######################################################

def test_layers_to_builder(tmp_path):
    domain, layers = _layers(svg_path("rifting"))
    builder = layers_to_builder(layers, domain["Lx"], domain["Lz"], Nz=31, path=str(tmp_path))
    interfaces = builder.DTree.interfaces.ds
    assert len(interfaces.data_vars) == len(layers)
    ids = sorted(interfaces[name].attrs["id"] for name in interfaces.data_vars)
    assert ids == [-1] + list(range(len(layers) - 1))
    seed = [name for name in interfaces.data_vars if name.endswith("weak_seed")]
    assert interfaces[seed[0]].attrs["weakening_seed"] == 1.0  # initial strain in the weak seed
    others = [name for name in interfaces.data_vars if not name.endswith("weak_seed")]
    assert all(interfaces[name].attrs["weakening_seed"] == -1.0 for name in others)


def test_write_interfaces_does_not_touch_other_files(tmp_path):
    domain, layers = _layers(svg_path("rifting"))
    (tmp_path / "interfaces.txt").write_text("keep me")
    write_interfaces(layers, domain["Lx"], domain["Lz"], str(tmp_path / "other.txt"))
    assert (tmp_path / "interfaces.txt").read_text() == "keep me"
    lines = (tmp_path / "other.txt").read_text().splitlines()
    assert lines[0].split()[0] == "C" and len(lines) == 13 + 161
