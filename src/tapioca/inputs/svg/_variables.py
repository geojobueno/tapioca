'''
Variables of the SVG scenarios (tapioca.inputs.svg.SVGScenario).
'''

__all__ = ["DEFAULT_PARAMS"]

DEFAULT_PARAMS:dict = {
    # Simulation options
    "multigrid": 1,
    "solver": "direct",
    "denok": 1e-14,
    "particles_per_element": 40,
    "surface_particles_per_element": 40,
    "particles_perturb_factor": 0.7,
    "rtol": 1.0e-7,
    "RK4": "Euler",
    "Xi_min": 1.0e-5,
    "random_initial_strain": 0.3,
    "pressure_const": -1.0,
    "initial_dynamic_range": True,
    "periodic_boundary": False,
    "high_kappa_in_asthenosphere": False,
    "K_fluvial": 2.0e-7,
    "m_fluvial": 1.0,
    "sea_level": 0.0,
    "basal_heat": 0.0,
    # Surface processes
    "sp_surface_tracking": True,
    "sp_surface_processes": False,
    "plot_sediment": False,
    "a2l": True,
    "free_surface_stab": True,
    "theta_FSSA": 0.5,
    # Time constraints
    "step_max": 800000,
    "dt_max": 10000.0,
    "step_print": 50,
    "sub_division_time_step": 0.5,
    "initial_print_step": 0,
    "initial_print_max_time": 1.0e6,
    # Viscosity
    "viscosity_reference": 1.0e26,
    "viscosity_max": 1.0e25,
    "viscosity_min": 1.0e18,
    "viscosity_per_element": "constant",
    "viscosity_mean_method": "arithmetic",
    "viscosity_dependence": "pressure",
    # External ASCII inputs/outputs
    "interfaces_from_ascii": True,
    "temperature_from_ascii": True,
    "binary_output": False,
    "sticky_blanket_air": True,
    "print_step_files": True,
    "checkered": False,
    "geoq": "on",
    "geoq_fac": 100.0,
    # Physical parameters
    "temperature_difference": 1500.0,
    "thermal_expansion_coefficient": 3.28e-5,
    "thermal_diffusivity_coefficient": 1.0e-6,
    "gravity_acceleration": 10.0,
    "density_mantle": 3300.0,
    "external_heat": 0.0e-12,
    "heat_capacity": 1250.0,
    "non_linear_method": "on",
    "adiabatic_component": "on",
    "radiogenic_component": "on",
    "export_lithology": True,
    # Velocity boundary conditions
    "top_normal_velocity": "fixed",
    "top_tangential_velocity": "free",
    "bot_normal_velocity": "fixed",
    "bot_tangential_velocity": "free",
    "left_normal_velocity": "fixed",
    "left_tangential_velocity": "fixed",
    "right_normal_velocity": "fixed",
    "right_tangential_velocity": "fixed",
    "surface_velocity": 0.0e-2,
    "multi_velocity": False,
    # Temperature boundary conditions
    "top_temperature": "fixed",
    "bot_temperature": "fixed",
    "left_temperature": "fixed",
    "right_temperature": "fixed",
    "rheology_model": 19,
    "T_initial": 3,
}
"""dict: Parameters of param.txt of the SVG scenarios that do not depend on the scenario (any of them can be
changed with keyword arguments of SVGScenario.write_all or SVGScenario.parameters)."""
