"""Tests of the generic builders (tapioca.inputs: MandyocBuilder, VelocityFieldBuilder and TemperatureFieldBuilder)."""

import os

import numpy as np
import pytest

import tapioca as tp
from tapioca.inputs import MandyocBuilder, TemperatureFieldBuilder, VelocityFieldBuilder
from tapioca.inputs.svg import AIR, DRY_OLIVINE, WET_OLIVINE, WET_QUARTZ

from conftest import quiet

Lx, Lz, Nx, Nz = 1000e3, 300e3, 101, 31  # 10 km x 10 km


@pytest.fixture
def builder(tmp_path):
    '''
    Air (40 km), crust (35 km), lithospheric mantle (thinner on the right) and asthenosphere.
    Each interface is the top of the layer whose material it carries.
    '''
    b = MandyocBuilder(str(tmp_path / "scenario"), Nx=Nx, Nz=Nz, Lx=Lx, Lz=Lz, verbose=False)
    b.create_interface(-1, 0.0, "air", **AIR.params)
    b.create_interface(0, np.where(b.x < Lx / 2, -160e3, -120e3), "asthenosphere", **WET_OLIVINE.params)
    b.create_interface(1, -75e3, "mantle", **DRY_OLIVINE.params)
    b.create_interface(2, -40e3, "crust", **WET_QUARTZ.params)
    b.create_materials_fields()
    return b


# MandyocBuilder ################################################################

def test_builder_grid(builder):
    assert builder.x[0] == 0 and builder.x[-1] == Lx and len(builder.x) == Nx
    assert builder.z[0] == -Lz and builder.z[-1] == 0 and len(builder.z) == Nz


def test_builder_creates_its_folder_only_on_export(builder):
    assert not os.path.exists(builder.path)
    builder.export_interfaces(export="none")
    assert os.path.isfile(os.path.join(builder.path, "interfaces.txt"))


def test_interface_needs_a_density(builder):
    with pytest.raises(ValueError, match="density"):
        builder.create_interface(3, -50e3, "no_density", C=1.0)


def test_material_fields_follow_the_interfaces(builder):
    rho = builder.DTree.fields.rho
    left, right = rho.isel(x=0), rho.isel(x=-1)
    at = lambda column, z: float(column.sel(z=z))
    assert at(left, -10e3) == AIR.params["rho"]            # sticky air
    assert at(left, -60e3) == WET_QUARTZ.params["rho"]     # crust
    assert at(left, -150e3) == DRY_OLIVINE.params["rho"]   # lithospheric mantle on the left
    assert at(right, -150e3) == WET_OLIVINE.params["rho"]  # asthenosphere below the thinner mantle on the right
    assert at(left, -250e3) == WET_OLIVINE.params["rho"]   # asthenosphere
    for key in tp.MATERIAL_PARAMETERS:
        assert key in builder.DTree.fields and not (builder.DTree.fields[key] == -999).any()


def test_export_interfaces_with_all_the_parameters(builder):
    builder.export_interfaces(export="none")
    lines = open(f"{builder.path}/interfaces.txt").read().splitlines()
    params = [line.split()[0] for line in lines[:len(tp.MATERIAL_PARAMETERS)]]
    assert params == ["C", "rho", "H", "A", "n", "Q", "V", "k", "weakening_seed", "cohesion_min", "cohesion_max",
                      "friction_angle_min", "friction_angle_max"]
    assert all(len(line.split()) == 1 + 4 for line in lines[:len(params)])  # one value per layer
    data = np.loadtxt(lines[len(params):])
    assert data.shape == (Nx, 3)  # one column per interface
    np.testing.assert_array_equal(data[:, 0], np.where(builder.x < Lx / 2, -160e3, -120e3))  # top of the asthenosphere
    np.testing.assert_array_equal(data[:, 2], -40e3)  # top of the crust


def test_params_keep_the_grid_of_the_builder(builder):
    builder.set_params(params_str="nx = 5\ntime_max = 1.0e6\nsolver = direct")
    assert builder.params["nx"] == Nx and builder.params["n_interfaces"] == 3
    assert builder.params["time_max"] == 1.0e6
    builder.export_params()
    text = open(f"{builder.path}/param.txt").read()
    assert "solver" in text and f"nx" in text and "dimensions" in text


def test_export_fields(builder):
    builder.create_temperature_field(_basic_temperature(builder))
    builder.export_field("temperature")
    T = np.loadtxt(f"{builder.path}/input_temperature_0.txt")
    assert T.shape == (Nx * Nz,)


# VelocityFieldBuilder ############################################################

def test_velocity_builder_conserves_the_flux(builder):
    with quiet():
        v = VelocityFieldBuilder(builder, units="cm/y")
        v.set_region("right", [-120e3, -40e3], vx=1.0)
        assert v.integrate_normal_components() > 0
        v.conservate_veloc("bot")
        assert abs(v.integrate_normal_components()) < 1e-12
        builder.create_velocity_field(v)
    vx = builder.DTree.fields.vx
    assert float(vx.sel(x=Lx, z=-80e3)) == pytest.approx(0.01 / tp.SEC_PER_YEAR)  # 1 cm/yr in m/s
    assert float(abs(vx.isel(x=slice(1, -1), z=slice(1, -1))).max()) == 0  # only the boundaries


def test_velocity_builder_profiles(builder):
    profile = np.linspace(-1e-10, 1e-10, Nz)
    with quiet():
        v = VelocityFieldBuilder(builder, units="m/s").set_profile("left", vx=profile)
        builder.create_velocity_field(v)
    np.testing.assert_array_equal(builder.DTree.fields.vx.isel(x=0).values, profile)


def test_linear_velocity(builder):
    with quiet():
        v = VelocityFieldBuilder(builder, units="m/s")
        v.linear_velocity("left", "vx", clims=[-Lz, 0], vlims=[1.0, 0.0])
    np.testing.assert_allclose(v.velocs["left"].vx.values, -builder.z / Lz, atol=1e-12)


# TemperatureFieldBuilder ##########################################################

def _basic_temperature(builder):
    t = TemperatureFieldBuilder(builder)
    t.apply_basic_temperature(lithosphere_thickness=120e3, t_pot=1350.0)
    return t


def test_basic_temperature(builder):
    T = _basic_temperature(builder).temperature
    assert float(T.min()) == 0.0
    assert float(T.sel(z=0)[0]) == 0.0  # top of the sticky air
    assert float(T.max()) < 1600.0


def test_conductive_columns_follow_the_lithosphere(builder):
    t = TemperatureFieldBuilder(builder)
    with quiet():
        t.solve_conductive_columns(np.where(builder.x < Lx / 2, 120e3, 80e3), time_max=20e6, t_pot=1400.0)
    T = t.temperature.sel(z=-100e3)  # 60 km below the surface
    assert float(T.isel(x=-1)) > float(T.isel(x=0))
    builder.create_temperature_field(t)
    assert builder.DTree.fields.temperature.attrs["t_potential"] == 1400.0


def test_heat_diffusion_1D(builder):
    t = _basic_temperature(builder)
    with quiet():
        t.solve_heat_diffusion1D(time_max=1e6, dt_years=1e4, x_inx=0)
    assert np.isfinite(t.temperature.values).all()


def test_yield_strength_envelope(builder):
    builder.create_temperature_field(_basic_temperature(builder))
    builder.calculate_YSE(sr=1e-15)
    fields = builder.DTree.fields
    for name in ["litho_P", "plastic_YSE", "viscous_YSE", "YSE", "YSE_soft"]:
        assert name in fields
    assert (fields.litho_P >= 0).all()
    assert (fields.YSE <= fields.plastic_YSE).all() and (fields.YSE <= fields.viscous_YSE).all()
    assert (fields.YSE_soft <= fields.YSE + 1e-6).all()  # softening lowers the strength
