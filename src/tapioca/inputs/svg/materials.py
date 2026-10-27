'''
Materials of Mandyoc scenarios: layers (MandyocLayer) and their rheologies (LithologicalUnit).

The rheologies are instances of LithologicalUnit built from tapioca.DEFAULT_MATERIALS
(WET_OLIVINE, DRY_OLIVINE, WET_QUARTZ, ...), so adding a material to that dictionary makes it
available here and in the SVG descriptions.
'''

import numpy as np

from ..._variables import DEFAULT_MATERIAL, DEFAULT_MATERIALS

#CLASSES
class MandyocLayer:
    def __init__(self, layer_label, rheology,
                 density, effective_viscosity_scale_factor=1.0, radiogenic_heat_production=0.0,
                 base_depth=0.0e3, Nx=None):
        """
        This class creates a layer with the given properties

        Parameters
        ----------
        layer_label: str
            Name of the layer
        rheology: LithologicalUnit or str
            Rheology of the layer, e.g. WET_OLIVINE or its name ("wet olivine", see get_rheology)
        density: float
            Density of the layer [kg/m3]
        interface: NoneType or np.array
            Interface of the layer.
            If None, the interface will be setted after the creation of the Layer.
            If np.array, the interface is defined by the given array.
        effective_viscosity_scale_factor: float
            Scale factor for the effective vistocisty
        radiogenic_heat_production: float
            Radiogenic heat production of the layer [W/kg]
        base_depth: float
            Depth of the layer base [m]
        Nx: int
            Number of points in x direction
        """

        self.layer_label = layer_label
        self.rheology = rheology if(isinstance(rheology, LithologicalUnit)) else get_rheology(rheology)
        self.density = density
        self.effective_viscosity_scale_factor = effective_viscosity_scale_factor
        self.radiogenic_heat_production = radiogenic_heat_production
        self.rheology_name = self.rheology.name
        self.pre_exponential_constant = self.rheology.pre_exponential_constant
        self.power_law_exponent = self.rheology.power_law_exponent
        self.activation_energy = self.rheology.activation_energy
        self.activation_volume = self.rheology.activation_volume
        self.base_depth = base_depth
        self.Nx = Nx

        self.interface = np.ones(Nx)*base_depth

    def material(self):
        '''
        Parameters of the layer in interfaces.txt (the keys of tapioca.DEFAULT_MATERIAL), as given to
        MandyocBuilder.create_interface: those of the rheology, replaced by the ones of the layer (C, rho, H,
        A, n, Q, V and any other key set as an attribute, e.g. cohesion_max given in the SVG description).
        '''
        params = dict(self.rheology.params)
        params.update(C=self.effective_viscosity_scale_factor, rho=self.density, H=self.radiogenic_heat_production,
                      A=self.pre_exponential_constant, n=self.power_law_exponent, Q=self.activation_energy,
                      V=self.activation_volume)
        params.update({key: getattr(self, key) for key in DEFAULT_MATERIAL if hasattr(self, key)})
        return params

class LithologicalUnit:
    """
    Generic material: a name and the parameters of interfaces.txt (the keys of tapioca.DEFAULT_MATERIAL:
    rho, C, H, A, n, Q, V, k, cohesion_min, ...). Parameters not given take the DEFAULT_MATERIAL values.

    Example: LithologicalUnit("wet_olivine", A=1.393e-14, n=3.0, Q=429.0e3, V=14.5e-6)

    name: str
        Name of the material, e.g. wet_olivine
    params: keyword arguments
        Parameters of the material
    """
    def __init__(self, name, **params):
        unknown = set(params) - set(DEFAULT_MATERIAL)
        if unknown:
            raise ValueError(f"Unknown parameter(s) of {name}: {', '.join(sorted(unknown))}. Use {', '.join(DEFAULT_MATERIAL)}.")
        self.name = name
        self.params = {**DEFAULT_MATERIAL, **params}

    # flow law with the names used by MandyocLayer
    @property
    def pre_exponential_constant(self):
        return self.params["A"]

    @property
    def power_law_exponent(self):
        return self.params["n"]

    @property
    def activation_energy(self):
        return self.params["Q"]

    @property
    def activation_volume(self):
        return self.params["V"]

    def __repr__(self):
        return f"LithologicalUnit({self.name!r}, {', '.join(f'{k}={v}' for k, v in self.params.items())})"


#Default rheologies, one instance of LithologicalUnit per material of DEFAULT_MATERIALS (WET_OLIVINE, AIR, ...)
RHEOLOGIES = {key.lower(): LithologicalUnit(key.lower(), **params) for key, params in DEFAULT_MATERIALS.items()}
globals().update({unit.name.upper(): unit for unit in RHEOLOGIES.values()})

__all__ = ["MandyocLayer", "LithologicalUnit", "RHEOLOGIES", "get_rheology"] + [name.upper() for name in RHEOLOGIES]

def get_rheology(name):
    '''
    Return the rheology (LithologicalUnit) from a free-form name.
    "wet olivine", "wet_olivine", "Wet Olivine" and "WetOlivine" all return WET_OLIVINE.
    '''
    key = name.replace(" ", "").replace("_", "").replace("-", "").lower()
    for rheology in RHEOLOGIES.values():
        if rheology.name.replace("_", "") == key:
            return rheology
    options = ", ".join(RHEOLOGIES)
    raise ValueError(f"Unknown rheology '{name}'. Options: {options}")
