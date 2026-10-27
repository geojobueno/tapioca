"""Tests of the SVG scenarios (tapioca.inputs.svg.SVGScenario) and of the command line tools."""

import sys

import numpy as np
import pytest

import tapioca as tp
from tapioca.inputs import MandyocBuilder, TemperatureFieldBuilder, VelocityFieldBuilder
from tapioca.inputs.svg import SVGInputError, SVGScenario
from tapioca.inputs.svg import cli, reader

from conftest import SVG_SCENARIOS, build_scenario, quiet, svg_path


@pytest.fixture(scope="module", params=list(SVG_SCENARIOS))
def written(request, tmp_path_factory):
    '''Each example on the coarse grid, written with write_all: (name, scenario, output folder).'''
    name = request.param
    scenario = build_scenario(name)
    folder = tmp_path_factory.mktemp(name)
    with quiet():
        scenario.write_all(str(folder), plots=False, system="linux", ncores=4)
    return name, scenario, folder


def _read_params(folder):
    return tp.evaluate_params(params_str=(folder / "param.txt").read_text())


# Input files ######################################################################

def test_input_files(written):
    name, scenario, folder = written
    expected = ["interfaces.txt", "param.txt", "input_temperature_0.txt", f"{name}.svg",
                "run_linux.sh", "0_organize_outputs.sh", "zipper.sh", f"infos_{folder.name}.txt"]
    if scenario.velocity_from_ascii:
        expected += ["input_velocity_0.txt", "vel_bc.txt", "velz_bc.txt"]
    if scenario.variable_bcv:
        expected.append("scale_bcv.txt")
    for fname in expected:
        assert (folder / fname).is_file(), fname
    assert (folder / "input_velocity_0.txt").exists() == (SVG_SCENARIOS[name][0] != "quiescence")


def test_param_txt_matches_the_scenario(written):
    _, scenario, folder = written
    params = _read_params(folder)
    assert (params["nx"], params["nz"]) == (scenario.Nx, scenario.Nz)
    assert params["n_interfaces"] == len(scenario.layers) - 1
    assert params["velocity_from_ascii"] == str(scenario.velocity_from_ascii)
    assert params["magmatism_Ts0"] == scenario.magmatism_Ts0


def test_interfaces_txt(written):
    _, scenario, folder = written
    lines = (folder / "interfaces.txt").read_text().splitlines()
    n_params = len(tp.MATERIAL_PARAMETERS)
    assert sorted(line.split()[0] for line in lines[:n_params]) == sorted(tp.MATERIAL_PARAMETERS)
    values = {line.split()[0]: [float(v) for v in line.split()[1:]] for line in lines[:n_params]}
    assert values["rho"] == [layer.density for layer in scenario.layers]
    assert values["C"] == [layer.effective_viscosity_scale_factor for layer in scenario.layers]
    seed = [i for i, layer in enumerate(scenario.layers) if layer.layer_label == scenario.seed_label]
    assert all(values["weakening_seed"][i] == (1.0 if i in seed else -1.0) for i in range(len(scenario.layers)))
    depths = np.loadtxt(lines[n_params:])
    assert depths.shape == (scenario.Nx, len(scenario.layers) - 1)
    expected = -np.array([layer.interface for layer in scenario.layers[1:]]).T
    np.testing.assert_allclose(depths, expected, rtol=0, atol=0.05)  # written with one decimal (%.1f)


def test_initial_temperature(written):
    _, scenario, folder = written
    T = np.loadtxt(folder / "input_temperature_0.txt")
    assert T.shape == (scenario.Nx * scenario.Nz,)
    T = T.reshape(scenario.Nz, scenario.Nx)
    np.testing.assert_array_equal(T, scenario.T)
    assert T.min() == 0.0 and T.max() < scenario.TP + 300
    air = scenario.Z < scenario.thickness_sa[None, :]
    assert np.all(T[air] == 0.0)


def test_velocity_field(written):
    name, scenario, folder = written
    regime, moving_boundary = SVG_SCENARIOS[name]
    if regime == "quiescence":
        assert scenario.VX is None and scenario.velocity_builder is None
        return
    with quiet():
        flux = scenario.velocity_builder.integrate_normal_components()
    assert abs(flux) < 1e-15, "the inflow must equal the outflow"
    v = np.loadtxt(folder / "input_velocity_0.txt").reshape(scenario.Nz, scenario.Nx, 2)
    np.testing.assert_array_equal(v[..., 0], scenario.VX)
    np.testing.assert_array_equal(v[..., 1], scenario.VZ)
    left, right = scenario.VX[:, 0], scenario.VX[:, -1]
    interior = scenario.VX[:, 1:-1]
    assert np.all(interior == 0), "velocity only at the boundaries"
    lithosphere = scenario.Z[:, 0] < scenario.thickness_sa[0] + scenario.h_v_const
    v_cm = scenario.velocity / (100 * scenario.velocity_builder.unifac * 365 * 24 * 3600)  # cm/yr -> m/s of the SVG
    if moving_boundary == "symmetric":
        np.testing.assert_allclose(right, -left)
        assert right[lithosphere & (scenario.Z[:, 0] > scenario.thickness_sa[0])].max() == pytest.approx(v_cm / 2)
    elif regime == "compression":
        fixed = right if moving_boundary == "left" else left
        assert np.all(fixed == 0), "the opposite boundary is fixed"
    else:
        moving = right if moving_boundary == "right" else left
        assert abs(moving).max() > 0


@pytest.mark.parametrize("quiescence_time, n_changes", [(0, 3), (300, 4)])
def test_scale_bcv_of_accordion(tmp_path, quiescence_time, n_changes):
    scenario = build_scenario("accordion", quiescence_time=quiescence_time)
    with quiet():
        scenario.write_all(str(tmp_path), plots=False, scripts=False)
    values = (tmp_path / "scale_bcv.txt").read_text().split()
    assert int(values[0]) == n_changes and len(values) == 1 + 2 * n_changes


# Settings ##########################################################################

def test_builders_of_the_scenario():
    scenario = build_scenario("rifting")
    assert isinstance(scenario.builder, MandyocBuilder)
    with quiet():
        scenario.build_temperature()
        scenario.build_velocity()
    assert isinstance(scenario.temperature_builder, TemperatureFieldBuilder)
    assert isinstance(scenario.velocity_builder, VelocityFieldBuilder)
    assert scenario.yield_strength_envelope(10).shape == (scenario.Nz,)
    for name in ["rho", "temperature", "vx", "vz", "YSE"]:
        assert name in scenario.builder.DTree.fields


def test_velocity_changed_by_script_is_written(tmp_path):
    scenario = build_scenario("rifting")
    with quiet():
        scenario.build_velocity()
        scenario.velocity_builder.set_region("top", [0, scenario.Lx], vz=1e-12, mode="add")
        scenario.builder.create_velocity_field(scenario.velocity_builder)
        scenario.write_all(str(tmp_path), plots=False, scripts=False)
    velz = np.loadtxt(tmp_path / "velz_bc.txt")
    assert velz == pytest.approx(scenario.builder.DTree.fields.vz.isel(z=-1).values)


@pytest.mark.parametrize("moving_boundary", ["left", "right", "symmetric"])
def test_moving_boundary_argument_overrides_the_svg(moving_boundary):
    scenario = build_scenario("rifting", moving_boundary=moving_boundary)
    assert scenario.moving_boundary == moving_boundary and scenario.source["moving_boundary"] == "argument"


def test_symmetric_compression_is_an_error():
    with pytest.raises(SVGInputError, match="symmetric"):
        build_scenario("compression", moving_boundary="symmetric")


def test_param_txt_keyword_arguments(tmp_path):
    scenario = build_scenario("quiescence")
    with quiet():
        scenario.write_all(str(tmp_path), plots=False, scripts=False, particles_per_element=100)
    assert _read_params(tmp_path)["particles_per_element"] == 100
    with pytest.raises(SVGInputError, match="Unknown parameter"):
        scenario.param_text(not_a_parameter=1)


def test_magmatism_extraction_adds_a_basalt_layer():
    scenario = build_scenario("rifting", magmatism="on", magmatism_extraction="on")
    assert scenario.layers[-2].layer_label == "basalt"
    assert scenario.magmatic_layer == len(scenario.layers) - 2


def test_missing_svg_is_an_error(tmp_path):
    with pytest.raises(SVGInputError, match="not found"):
        SVGScenario(str(tmp_path / "missing.svg"))


def test_figures(tmp_path):
    scenario = build_scenario("rifting_symmetric")
    with quiet():
        scenario.write_all(str(tmp_path), plots=True, scripts=False)
    for fname in ["numerical_setup.png", "initial_temperature.png"]:
        assert (tmp_path / fname).stat().st_size > 10_000


# Command line ######################################################################

def test_cli_tapioca_svg(tmp_path):
    with quiet():
        cli.main([str(svg_path("rifting")), "-o", str(tmp_path), "--dx", "10", "--dz", "10", "--no-plots",
                  "--no-scripts", "--regime", "accordion", "--quiescence-time", "10"])
    assert _read_params(tmp_path)["variable_bcv"] == "True"
    assert (tmp_path / "scale_bcv.txt").is_file()


def test_cli_tapioca_svg_error_message(tmp_path):
    with pytest.raises(SystemExit, match="Error"):
        cli.main([str(tmp_path / "missing.svg"), "-o", str(tmp_path)])


def test_cli_tapioca_svg_interfaces(tmp_path, monkeypatch):
    out = tmp_path / "layers.txt"
    monkeypatch.setattr(sys, "argv", ["tapioca-svg-interfaces", str(svg_path("rifting")), "--dx", "10", "-o", str(out)])
    with quiet():
        reader.main()
    lines = out.read_text().splitlines()
    assert len(lines) == len(tp.MATERIAL_PARAMETERS) + 161
