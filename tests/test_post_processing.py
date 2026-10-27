"""Tests of the outputs of Mandyoc (tapioca.MandyocScen) and of the post-processing (DataTree accessor postproc),
with the scenario of examples/continental_rift."""

import numpy as np
import pytest

import tapioca as tp

from conftest import CONTINENTAL_RIFT, quiet


@pytest.fixture
def scenario():
    '''A fresh scenario (density and particles), for the tests that change it.'''
    with quiet():
        return tp.MandyocScen(str(CONTINENTAL_RIFT), variables="density", load_particles=True)


# Loading ###########################################################################

def test_metadata(continental_rift):
    s = continental_rift
    assert (s.Nx, s.Nz) == (121, 31)
    assert (s.XMIN, s.XMAX, s.ZMIN, s.ZMAX) == (0.0, 1.2e6, -3.0e5, 0.0)
    assert s.TMIN == 0.0 and s.TMAX == pytest.approx(10.0, abs=0.01)
    assert s.params["nx"] == s.Nx and s.params["nz"] == s.Nz
    assert s.name == "continental_rift"


def test_data_tree(continental_rift):
    tree = continental_rift.DTree
    mesh = tree["/mesh/original"]
    for var in ["density", "temperature", "vx", "vz", "viscosity", "strain_rate"]:
        assert var in mesh.data_vars
        assert mesh[var].dims == ("time", "z", "x")
    assert mesh.sizes == {"time": 21, "z": 31, "x": 121}
    assert "lithology" in tree["/mesh/upscaled"].data_vars
    assert tree["/surface/topography"]["surface"].dims == ("time", "x")
    particles = tree["/particles/original"]
    assert particles.sizes["id"] == 18000 and {"x", "z", "lithology"} <= set(particles.variables)


def test_physical_ranges(continental_rift):
    mesh = continental_rift.DTree["/mesh/original"].ds.isel(time=0).compute()
    assert float(mesh.temperature.min()) >= 0 and float(mesh.temperature.max()) < 2000
    assert float(mesh.density.min()) >= 0 and float(mesh.density.max()) < 3500
    assert float(mesh.viscosity.min()) >= 0 and float(mesh.viscosity.max()) <= 1.0e26


def test_empty_variables_are_an_error():
    with pytest.raises(ValueError, match="cannot be empty"):
        tp.MandyocScen(str(CONTINENTAL_RIFT), variables=[])


def test_missing_variable_is_an_error(tmp_path):
    (tmp_path / "param.txt").write_text((CONTINENTAL_RIFT / "param.txt").read_text())
    with pytest.raises(FileNotFoundError):
        tp.MandyocScen(str(tmp_path), variables="density")


def test_limits_select_part_of_the_scenario():
    with quiet():
        s = tp.MandyocScen(str(CONTINENTAL_RIFT), variables="temperature",
                           xlimits=[2e5, 8e5], zlimits=[-1e5, 0], tlimits=[1, 5])
    mesh = s.DTree["/mesh/original"]
    assert float(mesh.x.min()) >= 2e5 and float(mesh.x.max()) <= 8e5
    assert float(mesh.z.min()) >= -1e5
    assert float(mesh.time.min()) >= 1 and float(mesh.time.max()) <= 5


def test_correct_z_coordinate(scenario):
    z_top = float(scenario.DTree["/mesh/original"].z.max())
    assert scenario.correctZcoord() is True
    assert float(scenario.DTree["/mesh/original"].z.max()) == z_top + scenario.thick_air
    assert scenario.zlimits[1] == z_top + scenario.thick_air
    assert scenario.correctZcoord() is False  # only once


# Post-processing ####################################################################

def test_deviatoric_stress_tensor(continental_rift):
    with quiet():
        continental_rift.DTree.postproc.DeviatoricStressTensor()
    mesh = continental_rift.DTree["/mesh/original"]
    for var in ["tau_xx", "tau_zz", "tau_xz", "tau_J2"]:
        assert var in mesh.data_vars
    snapshot = mesh.ds.isel(time=-1).compute()
    np.testing.assert_allclose(snapshot.tau_xx, -snapshot.tau_zz)  # incompressible flow
    assert float(snapshot.tau_J2.min()) >= 0


# Particles ###########################################################################

def test_select_particles_by_lithology_with_default_names(scenario):
    scenario.selectParticles_bylithology([1, 2])
    selected = scenario.DTree["/particles/selected"]
    assert 0 < selected.sizes["id"] < scenario.DTree["/particles/original"].sizes["id"]
    assert set(np.unique(selected.lithology.isel(time=0).values)) <= {1.0, 2.0}


def test_select_particles_by_coordinates(scenario):
    scenario.selectParticles_bycoords(xlim=[4e5, 8e5], selection_name="center")
    x = scenario.DTree["/particles/center"].x.isel(time=0).values
    assert x.size > 0 and x.min() >= 4e5 and x.max() <= 8e5


def test_chained_selections(scenario):
    scenario.selectParticles_bycoords(xlim=[4e5, 8e5], selection_name="center")
    scenario.selectParticles_bylithology([1], selected_name="center", selection_name="center_1")
    n_center = scenario.DTree["/particles/center"].sizes["id"]
    n_center_1 = scenario.DTree["/particles/center_1"].sizes["id"]
    assert 0 < n_center_1 < n_center
