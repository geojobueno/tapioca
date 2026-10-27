"""Tests of the variables (tapioca._variables) and of the auxiliary functions (tapioca._aux_functions)."""

import numpy as np
import pytest

import tapioca as tp
from tapioca._aux_functions import _conductive_columns, evaluate_params, read_params

from conftest import CONTINENTAL_RIFT


# Variables ##################################################################

def test_material_parameters_have_units_and_names():
    assert tp.MATERIAL_PARAMETERS == list(tp.DEFAULT_MATERIAL)
    for key in tp.MATERIAL_PARAMETERS:
        assert key in tp.PARAMETERS_UNITS
        assert key in tp.INTERFACES_PARAMETERS


@pytest.mark.parametrize("name", list(tp.DEFAULT_MATERIALS))
def test_default_materials_have_every_parameter(name):
    assert set(tp.DEFAULT_MATERIALS[name]) == set(tp.DEFAULT_MATERIAL)


def test_seconds_per_year():
    assert tp.SEC_PER_YEAR == pytest.approx(365.25 * 24 * 3600)


# param.txt ###################################################################

def test_read_params_of_example():
    params = read_params(str(CONTINENTAL_RIFT / "param.txt"))
    assert params["nx"] == 121 and params["nz"] == 31
    assert params["lx"] == pytest.approx(1.2e6)
    for key, value in params.items():
        assert isinstance(value, tp.PARAMS_TYPES[key])


def test_evaluate_params_from_text_and_keywords():
    text = """
    # comment line
    nx = 11        # inline comment
    time_max = 1.0e6
    solver = direct
    """
    assert evaluate_params(params_str=text) == {"nx": 11, "time_max": 1.0e6, "solver": "direct"}
    assert evaluate_params(nx="11", magmatism_Ts0="1080") == {"nx": 11, "magmatism_Ts0": 1080.0}


def test_evaluate_params_rejects_unknown_parameters():
    with pytest.raises(KeyError):
        evaluate_params(not_a_parameter=1)


# Conductive model ##############################################################

def _columns(thickness_litho, H=0.0, num_steps=20000, dt_years=5000.0):
    depth = np.linspace(300e3, 0, 151)  # from the bottom, as in Mandyoc
    Nx = len(thickness_litho)
    T, n_columns = _conductive_columns(depth, np.full(Nx, 40e3), np.asarray(thickness_litho, dtype=float),
                                       np.full((len(depth), Nx), H), dt=dt_years * tp.SEC_PER_YEAR,
                                       num_steps=num_steps, t_pot=1400.0)
    return depth, T, n_columns


def test_conductive_columns_solves_each_distinct_column_once():
    depth, T, n_columns = _columns([120e3, 120e3, 80e3, 120e3])
    assert n_columns == 2
    assert T.shape == (len(depth), 4)
    np.testing.assert_array_equal(T[:, 0], T[:, 1])
    np.testing.assert_array_equal(T[:, 0], T[:, 3])


def test_conductive_columns_boundary_conditions():
    depth, T, _ = _columns([120e3, 80e3])
    air = depth <= 40e3
    assert np.all(T[air] == 0.0)  # sticky air at 0 oC
    assert np.all(np.diff(T[::-1], axis=0) >= -1e-9)  # temperature increases with depth
    # adiabat below the lithosphere (t_pot exp(-g alpha z / c)), unchanged by the diffusion
    deep = depth >= 40e3 + 200e3
    adiabat = 1400.0 / np.exp(-10.0 * 3.28e-5 * (depth[deep] - 40e3) / 1250.0)
    np.testing.assert_allclose(T[deep, 0], adiabat)


def test_conductive_columns_thinner_lithosphere_is_hotter():
    depth, T, _ = _columns([120e3, 80e3])
    i = np.argmin(abs(depth - (40e3 + 60e3)))  # 60 km below the surface
    assert T[i, 1] > T[i, 0]


def test_conductive_columns_radiogenic_heat_warms_the_lithosphere():
    depth, T0, _ = _columns([120e3])
    _, TH, _ = _columns([120e3], H=1.0e-9)
    litho = (depth > 40e3) & (depth < 40e3 + 120e3)
    assert np.all(TH[litho] >= T0[litho]) and np.any(TH[litho] > T0[litho])
