'''
Builder of the input files of a Mandyoc scenario drawn in an SVG file.

The geometry and the materials come from the SVG (see tapioca.inputs.svg.reader). The shape
labelled "setup" gives the domain, the resolution and, optionally, the numerical
experiment:

    Lx = 1600 km
    Lz = 400 km
    dx = 2 km
    dz = 2 km
    regime = accordion                         (rifting, compression, accordion, quiescence)
    velocity = 1 cm/yr
    moving_boundary = right                    (left, right or symmetric)
    quiescence_time = 300 Myr
    time_max = 500 Myr
    magmatism = off
    magmatism_extraction = off
    magmatism_Ts0 = 1080
    preset_initial_temperature = ../preliminary
    temperature_solver = 2D                    (2D or columns)
    min_thickness = 1 km

Each setting may also be given to SVGScenario (or on the command line, tapioca-svg),
which overrides the SVG; without it, the default is used.

Example
-------
    from tapioca.inputs.svg import SVGScenario
    scenario = SVGScenario("rifting.svg", regime="accordion", quiescence_time=300)
    scenario.write_all("my_scenario")

The scenario is built on the generic builders of tapioca: scenario.builder is a MandyocBuilder with the
interfaces and the material fields of the SVG, and build_temperature and build_velocity use a
TemperatureFieldBuilder and a VelocityFieldBuilder. The input files are exported by scenario.builder,
so interfaces.txt has the 13 parameters of tapioca.DEFAULT_MATERIAL.
'''

import os, re, glob, shutil, platform

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..generic_input_classes import TemperatureFieldBuilder, VelocityFieldBuilder
from .materials import MandyocLayer, BASALT
from .reader import SVGInputError, SETUP_LABEL, MIN_THICKNESS, read_domain, layers_from_svg, layers_to_builder, points_from_spacing

__all__ = ["SVGScenario", "REGIMES"]

## This variables should be in a dedicate file to auxiliary variables

REGIMES = ['rifting', 'compression', 'accordion', 'quiescence']
SEC_PER_YEAR = 365 * 24 * 3600
MANTLE_RHEOLOGIES = ['wet_olivine', 'dry_olivine']
SYSTEMS = ['linux', 'mac']

#Script of tapioca that moves the outputs of Mandyoc to one folder per variable (copied to the scenario)
ORGANIZE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'postrunning', '0_organize_outputs.sh')

#Run script of Mandyoc (@KEY@ are replaced in write_scripts). PETSC_DIR, PETSC_ARCH and MANDYOC
#set in the environment override the values of the script.
RUN_SCRIPT = '''#!/usr/bin/env bash
# Runs Mandyoc and moves the outputs to one folder per variable (0_organize_outputs.sh). Created by tapioca.
# PETSC_DIR, PETSC_ARCH and MANDYOC set in the environment override the values below.

PETSC_DIR=${PETSC_DIR:-@PETSC_DIR@}
PETSC_ARCH=${PETSC_ARCH-@PETSC_ARCH@}
MANDYOC=${MANDYOC:-@MANDYOC@}
NUMBER_OF_CORES=@NCORES@
MANDYOC_OPTIONS="@MANDYOC_OPTIONS@"

touch FD.out
@LAUNCHER@$PETSC_DIR/$PETSC_ARCH/bin/mpirun -n $NUMBER_OF_CORES $MANDYOC $MANDYOC_OPTIONS | tee FD.out

# Moves the outputs to one folder per variable
bash 0_organize_outputs.sh

# Zips the inputs and the outputs (skips the files already in the zip)
# bash zipper.sh
'''

#caffeinate keeps the Mac awake while Mandyoc runs
RUN_LAUNCHER = {'linux': '', 'mac': 'caffeinate -i '}

#Zips the scenario in <folder>.zip, skipping the files already in the zip. The outputs are in the
#folders created by 0_organize_outputs.sh.
ZIP_SCRIPT = '''#!/usr/bin/env bash
# Zips the files of Mandyoc in $DIRNAME.zip, skipping the files already in the zip. Created by tapioca.
set -eo pipefail

DIRNAME=${DIRNAME:-$(basename "$PWD")}
ZIP="${DIRNAME:?DIRNAME is not set}.zip"

# Patterns: files or folders (the folders are zipped with all their files)
patterns=(
"interfaces.txt"
"param.txt"
"input*_0.txt"
"vel_bc.txt"
"velz_bc.txt"
"scale_bcv.txt"
"infos_*.txt"
"*.svg"
"run*.sh"
"bc_velocity_*.txt"
"sp_surface_global_*.txt"
"FD.out"
"density"
"heat"
"pressure"
"surface"
"lithos"
"strain"
"strain_rate"
"temperature"
"time"
"velocity"
"viscosity"
"thermal_diffusivity"
"steps"
"Phi"
"dPhi"
"X_depletion"
"*.bin*.txt"
"bc*-1.txt"
"*.log"
# "_*.nc"
)

export LC_ALL=C   # same order for comm
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Files that match the patterns; in the folders, all their files
# (find avoids the "argument list too long" error)
for pat in "${patterns[@]}"; do
    find . -maxdepth 1 -name "$pat" ! -name "$ZIP" -exec find {} -type f \\;
done | sed 's|^\\./||' | sort -u > "$tmp/all"

# Files already in the zip
if [[ -f "$ZIP" ]]; then
    zipinfo -1 "$ZIP" | sort -u > "$tmp/zipped"
    grow=-g   # appends to the zip without rewriting it
else
    : > "$tmp/zipped"
    grow=
fi

# Only the files that are not in the zip yet
comm -23 "$tmp/all" "$tmp/zipped" > "$tmp/new"
n=$(wc -l < "$tmp/new" | tr -d ' ')

if [[ "$n" -eq 0 ]]; then
    echo "No new files for $ZIP"
else
    echo "Adding $n file(s) to $ZIP"
    zip -q $grow "$ZIP" -@ < "$tmp/new"
fi
'''

###############################################################################
# Auxiliary functions
###############################################################################

def find_nearest(array, value):
    '''Return the index in array nearest to a given value.'''
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def layer_indices(layers, label):
    '''Return the indices (position in the stack) of the layers with the given label.'''
    return [i for i, layer in enumerate(layers) if layer.layer_label == label]

def slot_thickness(layers, i, ix):
    '''Thickness [m] of the i-th layer of the stack at column(s) ix.'''
    top = layers[i+1].interface[ix] if i < len(layers)-1 else 0.0
    return layers[i].interface[ix] - top

def layer_thickness(layers, label, ix):
    '''Total thickness [m] of the layers with the given label at column(s) ix.'''
    return sum(slot_thickness(layers, i, ix) for i in layer_indices(layers, label))

## In mandyoc, the default is negative downwards. This function should be corrected
def layer_grid(layers, z):
    '''
    Index of the layer at each node as a (Nz, Nx) array.
    z is the depth of the nodes (positive downwards). Nodes on an interface belong to the layer below.
    '''
    bases = np.array([layer.interface for layer in layers[1:]]) #(n_layers-1, Nx)
    return (z[:, None, None] < bases[None, :, :]).sum(axis=1)

def parse_number_with_unit(text, units, name):
    '''Read a number with an optional unit, e.g. "1 cm/yr", and return it in the unit of factor 1 of units ({unit: factor}).'''
    match = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([A-Za-z/]*)\s*", str(text))
    if(match is None or (match.group(2) and match.group(2) not in units)):
        raise SVGInputError(f"Could not read {name} = {text}. Units: {', '.join(units)}")
    try:
        number = float(match.group(1))
    except ValueError:
        raise SVGInputError(f"Could not read {name} = {text}.")
    return number * units[match.group(2)] if(match.group(2)) else number

## This function can be replaced by read_param in aux_functions
def _read_param_grid(folder):
    '''Nx, Nz, Lx, Lz from the first lines of the param.txt in folder.'''
    with open(os.path.join(folder, "param.txt"), "r") as f:
        Nx = int(f.readline().split()[-1])
        Nz = int(f.readline().split()[-1])
        Lx = float(f.readline().split()[-1])
        Lz = float(f.readline().split()[-1])
    return Nx, Nz, Lx, Lz

## This function can be replaced by read_data in aux_functions
def _read_step_data(prop, step, Nz, Nx, folder):
    '''Field prop_step.txt in folder as a (Nz, Nx) array.'''
    data = np.loadtxt(os.path.join(folder, f"{prop}_{step}.txt"), skiprows=2, unpack=True, comments="P")
    return np.reshape(data, (Nz, Nx))


## This function should be in _aux_functions. The correct solution is to call the 
## `MandyocScenario` to load the temperature field and then extract the desired variables.
## In this way, is not necessary to load all params and check all those settings.
def read_preset_temperature(path):
    '''
    Horizontal mean of the final temperature field of a preliminary scenario.

    Parameters
    ----------
    path: str
        Folder of the preliminary scenario (with _output_temperature.nc, or with param.txt and temperature_*.txt)
        or the _output_temperature.nc file itself.

    Returns
    -------
    depth: depth of the nodes from the top of the domain [m] (increasing)
    profile: horizontal mean of the temperature [oC]
    thickness_sa: thickness of the sticky air [m]
    thickness_litho: thickness of the lithosphere [m] (None if the interfaces.txt of the scenario is not found)
    source: description of the field used
    '''
    folder = os.path.dirname(path) if(os.path.isfile(path)) else path
    nc_file = path if(os.path.isfile(path)) else os.path.join(folder, '_output_temperature.nc')
    ## this function looks for "_output_temperature.nc", but the default name for netcdf is: temperature.nc (meshes2netcdf.jl)
    ## use the `MandyocScenario` to read the results avoid this error

    if(os.path.isfile(nc_file)):
        dataset = xr.open_dataset(nc_file)
        temperature = dataset.temperature.isel(time=-1) #(x, z), final step
        depth = -dataset.z.values
        profile = temperature.mean(dim='x').values
        source = f"{nc_file} (time = {float(dataset.time[-1]):g} Myr)"
    else:
        if(not os.path.isfile(os.path.join(folder, 'param.txt'))):
            raise SVGInputError(f"preset_initial_temperature: no _output_temperature.nc nor param.txt in {folder}")
        Nx_aux, Nz_aux, Lx_aux, Lz_aux = _read_param_grid(folder)
        steps = [int(m.group(1)) for m in (re.fullmatch(r'temperature_(\d+)\.txt', os.path.basename(f))
                 for f in glob.glob(os.path.join(folder, 'temperature_*.txt'))) if m]
        if(len(steps) == 0):
            raise SVGInputError(f"preset_initial_temperature: no temperature_*.txt in {folder}")
        step_final = max(steps) #step of final thermal structure
        Datai = _read_step_data('temperature', step_final, Nz_aux, Nx_aux, folder) #(Nz, Nx), from the bottom to the top
        depth = np.linspace(Lz_aux, 0, Nz_aux)
        profile = np.mean(Datai, axis=1)
        source = f"{folder}/temperature_{step_final}.txt"

    order = np.argsort(depth)
    depth, profile = depth[order], profile[order]

    #thickness of the sticky air (last interface) and of the lithosphere (first interface = top of the asthenosphere)
    #from the interfaces.txt of the preliminary scenario (most common values along x). Without it, the sticky air is
    #the depth where the temperature starts to increase and the lithosphere is unknown (None).
    thickness_sa, thickness_litho = None, None
    for fname in [os.path.join(folder, 'interfaces.txt')] + sorted(glob.glob(os.path.join(folder, 'inputs*', 'interfaces.txt'))):
        if(os.path.isfile(fname)):
            try:
                interfaces_preset = -np.loadtxt(fname, skiprows=7, ndmin=2)
            except ValueError:
                continue
            def most_common(values):
                v, c = np.unique(np.round(values, -2), return_counts=True)
                return v[np.argmax(c)]
            thickness_sa = most_common(interfaces_preset[:, -1])
            thickness_litho = most_common(interfaces_preset[:, 0]) - thickness_sa
            source += f", interfaces from {fname}"
            break
    if(thickness_sa is None):
        thickness_sa = depth[max(np.argmax(profile > 1.0) - 1, 0)]

    return depth, profile, thickness_sa, thickness_litho, source

## Probably the current `TemperatureFieldBuilder` can be called to adapt the preset temperature. 
## the surface and LAB interfaces inside the builder could remove the complicated loop inside this function
def adapt_preset_temperature(depth_ref, T_ref, sa_ref, litho_ref, z, thickness_sa, thickness_litho, thickness_mantle):
    '''
    Initial temperature field (Nz, Nx) from the horizontal mean profile of a preliminary scenario,
    adapted to the lithosphere of each column:

    - columns with the reference lithosphere thickness litho_ref: the profile itself (as a function of the
      depth below the surface);
    - other columns (keels, thinned or oceanic lithosphere, ...): the profile in the crust, a linear geotherm
      in the lithospheric mantle from the Moho down to the base of the lithosphere, and the asthenospheric
      profile below it. Where the lithosphere is thinner than litho_ref, the asthenospheric profile is
      extended upwards with a linear fit (adiabat) of the profile below litho_ref + 50 km.

    Parameters
    ----------
    depth_ref, T_ref: np.array
        Depth from the top [m] and temperature [oC] of the profile
    sa_ref: float
        Sticky air thickness of the profile [m]
    litho_ref: float
        Lithosphere thickness of the preliminary scenario [m]
    z: np.array
        Depth of the nodes [m]
    thickness_sa, thickness_litho, thickness_mantle: np.array
        Thicknesses along x [m]
    '''
    d_ref = depth_ref - sa_ref #depth below the surface in the preliminary scenario
    cond = (d_ref >= litho_ref + 50.0e3) & (d_ref <= d_ref.max() - 50.0e3)
    a, b = np.polyfit(d_ref[cond], T_ref[cond], 1) #asthenospheric profile

    def T_profile(d):
        '''Profile at depth d below the surface (extended below the preliminary domain with the fit)'''
        return np.where(d > d_ref.max(), a*d + b, np.interp(d, d_ref, T_ref))

    columns = np.round(np.vstack([thickness_sa, thickness_litho, thickness_mantle]), 1)
    columns_unique, idx_columns = np.unique(columns, axis=1, return_inverse=True)
    T_unique = np.zeros((len(z), columns_unique.shape[1]))
    for k, (sa, litho, mantle) in enumerate(columns_unique.T):
        d = z - sa #depth below the surface
        if(abs(litho - litho_ref) < 1.0e3): #reference lithosphere
            Tk = T_profile(d)
        else:
            moho = litho - mantle
            T_below = T_profile(d) if(litho >= litho_ref) else np.maximum(T_profile(d), a*d + b) #asthenosphere
            T_lab = T_profile(litho) if(litho >= litho_ref) else max(float(T_profile(litho)), a*litho + b)
            T_moho = T_profile(moho)
            Tk = np.where(d <= moho, T_profile(d), T_below)
            in_mantle = (d > moho) & (d < litho)
            if(litho > moho):
                Tk[in_mantle] = T_moho + (T_lab - T_moho) * (d[in_mantle] - moho) / (litho - moho)
        Tk[d <= 0] = 0.0 #sticky air
        T_unique[:, k] = Tk
    T = T_unique[:, idx_columns.ravel()]
    T[T <= 1.0e-7] = 0.0 #dealing with <=0 values inherited from interpolation
    return T

## Is it necessary? If yes, this function should be moved to auxilary functions
def _write_clean(fname, text):
    '''Write text without indentation and empty lines (as the param.txt of Mandyoc).'''
    with open(fname, "w") as f:
        for line in text.split("\n"):
            line = line.strip()
            if len(line):
                f.write(" ".join(line.split()) + "\n")

###############################################################################
# Scenario builder
###############################################################################

class SVGScenario:
    '''
    Mandyoc scenario drawn in an SVG file.

    The geometry, the materials and the domain come from the SVG (see tapioca.inputs.svg.reader). The settings
    of the numerical experiment are taken from the arguments, then from the description of the setup
    shape of the SVG and finally from the defaults below.

    Parameters
    ----------
    svg_file : str
        SVG file with the scenario.
    dx, dz : float, optional
        Horizontal and vertical resolution [km] (default: from the SVG).
    regime : str, optional
        'rifting' (extension), 'compression' (convergence), 'accordion' (extension, convergence,
        quiescence and extension again, as in M300) or 'quiescence' (no velocity at the boundaries).
        Default: rifting.
    velocity : float, optional
        Velocity at the moving boundary [cm/yr] (with symmetric, the sum of the velocities of both boundaries).
        Default: 1 cm/yr.
    moving_boundary : str, optional
        'left', 'right' or 'symmetric' (rifting and accordion: both boundaries move with velocity/2 and the
        reference frame is at the centre of the domain). Default: right.
    quiescence_time : float, optional
        Duration of the quiescence [Myr] in the accordion and quiescence regimes. Default: 0 Myr.
    time_max : float, optional
        Duration of the simulation [Myr]. Default: 60 Myr in rifting and compression, the end of the
        second extension in accordion (which time_max sets when given) and quiescence_time (or 1000 Myr)
        in quiescence.
    magmatism, magmatism_extraction : str, optional
        'on' or 'off'. Default: off. With both on, a basalt layer of zero thickness is added below the
        sticky air to receive the extracted melt (magmatic_layer).
    magmatism_Ts0 : float, optional
        magmatism_Ts0 of param.txt [oC]. Default: 1080.
    preset_initial_temperature : str, optional
        Folder (or _output_temperature.nc) of a preliminary scenario: the horizontal mean of its final
        temperature, adapted to the lithosphere of the SVG, is the initial temperature. A relative path in the
        SVG is relative to the folder of the SVG. Default: none (conductive model).
    temperature_solver : str, optional
        Steady-state conduction in the lithosphere of the conductive model: '2D' (the whole lithosphere at once,
        so heat also flows sideways where the layers change along x) or 'columns' (each column alone).
        Default: 2D.
    asthenosphere_label, seed_label : str
        Labels of the asthenosphere and of the weak seed in the SVG.
    sampling : float
        Vertical sampling used to locate the interfaces in the SVG [m].
    min_thickness : float, optional
        Pieces of a unit thinner than min_thickness [km] in every column, when the unit is thicker
        elsewhere, are removed (slivers left by overlapping shapes; 0 keeps them). Default: from the setup
        shape of the SVG, or 1 km.
    verbose : bool
        Print the steps.

    Attributes
    ----------
    layers : list of MandyocLayer (bottom to top); layer.interface is the depth of the layer base [m]
    x, z : coordinates of the nodes [m] (z is the depth, from the bottom to the top of the domain)
    T : initial temperature (Nz, Nx), after build_temperature()
    VX, VZ : initial velocity (Nz, Nx), after build_velocity()
    builder : MandyocBuilder (tapioca.inputs) with the interfaces and the fields of the scenario; the
              input files are exported by it, so it can be changed by script before write_all
    temperature_builder, velocity_builder : TemperatureFieldBuilder and VelocityFieldBuilder of the scenario,
              after build_temperature() and build_velocity()
    '''

    #Initial temperature (the thermal diffusivity k of each layer comes from its description or rheology)
    TP = 1400.0 #mantle potential temperature [oC]
    ccapacity = 1250.0 #heat capacity [J/kg/K]

    #Accordion regime [Myr]
    ti_convergence = 25.0
    ti_quiescence = 75.0
    dt_rifting2 = 60.0

    #Velocity boundary conditions
    fac_air = 10.0e3 #no velocity in the sticky air above thickness_sa - fac_air [m]
    h_v_reference = 'lithosphere' #or 'lithospheric_mantle': reference thickness of the region with constant velocity

    #Plots: profiles of temperature and YSE
    min_profile_width = 0.05 #fraction of Lx
    profile_colors = ['tab:orange', 'tab:purple', 'tab:green', 'tab:blue', 'r', 'k'] #the thickest lithosphere is the last one

    ## because some parameters are useful only when a main option is "on", 
    ## many of this parameters could be kwargs, such as "magmatism_extraction", "quiescence_time", "magmatism_Ts0"
    def __init__(self, svg_file, dx=None, dz=None, regime=None, velocity=None, moving_boundary=None,
                 quiescence_time=None, time_max=None, magmatism=None, magmatism_extraction=None,
                 magmatism_Ts0=None, preset_initial_temperature=None, temperature_solver=None,
                 asthenosphere_label='asthenosphere', seed_label='weak_seed', sampling=100.0, min_thickness=None,
                 verbose=True):

        self.verbose = verbose
        self.svg_file = svg_file
        if(not os.path.isfile(svg_file)):
            raise SVGInputError(f"SVG file not found: {svg_file}")

        domain = read_domain(svg_file)
        if(domain is None):
            raise SVGInputError(f"No shape labelled '{SETUP_LABEL}' in {svg_file}. Draw a rectangle over the model domain, "
                                f"label it '{SETUP_LABEL}' and write Lx, Lz, dx and dz in its description.")

        #domain and resolution
        self.Lx, self.Lz = domain['Lx'], domain['Lz']
        self.Nx = points_from_spacing(self.Lx, dx * 1.0e3, 'dx') if(dx) else domain['Nx']
        self.Nz = points_from_spacing(self.Lz, dz * 1.0e3, 'dz') if(dz) else domain['Nz']
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.z = np.linspace(self.Lz, 0, self.Nz)
        self.X, self.Z = np.meshgrid(self.x, self.z)
        self.dx = self.Lx / (self.Nx - 1)
        self.dz = self.Lz / (self.Nz - 1)

        #numerical experiment: argument > SVG > default
        self._experiment = domain.get('experiment', {})
        self.source = {}
        self._read_settings(regime, velocity, moving_boundary, quiescence_time, time_max,
                            magmatism, magmatism_extraction, magmatism_Ts0, preset_initial_temperature,
                            temperature_solver)

        #layers
        self.asthenosphere_label = asthenosphere_label
        self.seed_label = seed_label
        self.min_thickness = min_thickness * 1.0e3 if(min_thickness is not None) else domain.get('min_thickness', MIN_THICKNESS)
        self.layers = layers_from_svg(svg_file, self.Lx, self.Lz, self.Nx, dz=sampling, domain_id=domain['id'],
                                      min_thickness=self.min_thickness, verbose=verbose)
        if(self.layers[-1].rheology_name != 'air'):
            raise SVGInputError(f"The top layer of {svg_file} must be the sticky air, with 'air' in the first line of its "
                                f"description (found '{self.layers[-1].layer_label}' with {self.layers[-1].rheology_name}).")
        if(len(layer_indices(self.layers, asthenosphere_label)) == 0):
            raise SVGInputError(f"No shape labelled '{asthenosphere_label}' in {svg_file}.")

        #Basalt layer with zero thickness just below the sticky air: it can not be drawn in the SVG and
        #receives the melt extracted by Mandyoc (magmatic_layer in param.txt)
        if(self.magmatism == 'on' and self.magmatism_extraction == 'on'):
            basalt = MandyocLayer('basalt', BASALT,
                                  density=2900.0,
                                  effective_viscosity_scale_factor=1.0,
                                  radiogenic_heat_production=9.0e-11,
                                  Nx=self.Nx)
            basalt.interface = np.copy(self.layers[-1].interface) #base of basalt = base of sticky air
            basalt.base_depth = basalt.interface.max()
            basalt.title = 'Basalt'
            basalt.color = 'xkcd:dark grey blue'
            self.layers.insert(len(self.layers)-1, basalt)
        
        ## It's necessary to create a convention for the layers ID, because there are the water and sediments layer 
        self.magmatic_layer = len(self.layers) - 2 #layer below the sticky air

        #index of the layer at each node, sticky air and lithosphere along x
        self.grid_layers = layer_grid(self.layers, self.z)
        self.thickness_sa = self.layers[-1].interface #base of the sticky air
        self.surface_id = len(self.layers) - 2 #interface ID of the surface: top of the layer below the sticky air
        self.idx_ast = max(layer_indices(self.layers, asthenosphere_label))
        self.thickness_litho = self.layers[self.idx_ast+1].interface - self.thickness_sa #surface to the top of the asthenosphere
        self.thickness_mantle = np.zeros(self.Nx) #lithospheric mantle (olivine rheologies above the asthenosphere)
        for i in range(self.idx_ast + 1, len(self.layers)):
            if(self.layers[i].rheology_name in MANTLE_RHEOLOGIES):
                self.thickness_mantle += slot_thickness(self.layers, i, slice(None))

        #position of the vertical profile of the scenario infos and the profiles of the plots
        self.x_profile = self.Lx / 4.0
        self.idx_profile = find_nearest(self.x, self.x_profile)
        self.profiles = self._find_profiles()

        #generic builder of tapioca (tapioca.inputs.MandyocBuilder) with the interfaces and the material fields of
        #the SVG; the temperature, the velocity and the YSE are built on it and the input files are exported by it
        self.builder = layers_to_builder(self.layers, self.Lx, self.Lz, Nz=self.Nz, seed_label=seed_label, verbose=False)
        self.builder.create_materials_fields()
        self.temperature_builder, self.velocity_builder = None, None

        self.T, self.VX, self.VZ = None, None, None

    def _print(self, message):
        if(self.verbose):
            print(message)

    ###########################################################################
    # Settings
    ###########################################################################

    def _setting(self, name, value, default):
        '''value (argument) > SVG > default'''
        if(value is not None):
            self.source[name] = 'argument'
            return value
        if(name in self._experiment):
            self.source[name] = 'SVG'
            return self._experiment[name]
        self.source[name] = 'default'
        return default

    def _on_off(self, name, value):
        value = str(self._setting(name, value, 'off')).strip().lower()
        if(value not in ['on', 'off']):
            raise SVGInputError(f"{name} must be on or off (found '{value}').")
        return value

    def _read_settings(self, regime, velocity, moving_boundary, quiescence_time, time_max,
                       magmatism, magmatism_extraction, magmatism_Ts0, preset_initial_temperature,
                       temperature_solver):
        time_units = {'Myr': 1.0, 'My': 1.0, 'kyr': 1.0e-3, 'yr': 1.0e-6}

        self.regime = str(self._setting('regime', regime, 'rifting')).strip().lower()
        if(self.regime not in REGIMES):
            raise SVGInputError(f"Unknown regime '{self.regime}'. Use {', '.join(REGIMES)}.")

        self.velocity = self._setting('velocity', velocity, 1.0) #cm/yr
        if(isinstance(self.velocity, str)):
            self.velocity = parse_number_with_unit(self.velocity, {'cm/yr': 1.0, 'cm/y': 1.0, 'mm/yr': 0.1, 'mm/y': 0.1}, 'velocity')

        self.moving_boundary = str(self._setting('moving_boundary', moving_boundary, 'right')).strip().lower()
        if(self.moving_boundary not in ['left', 'right', 'symmetric']):
            raise SVGInputError(f"Unknown moving_boundary '{self.moving_boundary}'. Use left, right or symmetric.")
        if(self.moving_boundary == 'symmetric' and self.regime == 'compression'):
            raise SVGInputError("moving_boundary = symmetric is only available in the rifting and accordion regimes.")

        time_max = self._setting('time_max', time_max, None) #Myr
        if(isinstance(time_max, str)):
            time_max = parse_number_with_unit(time_max, time_units, 'time_max')

        self.quiescence_time = self._setting('quiescence_time', quiescence_time, 0.0) #Myr
        if(isinstance(self.quiescence_time, str)):
            self.quiescence_time = parse_number_with_unit(self.quiescence_time, time_units, 'quiescence_time')
        if(self.quiescence_time < 0):
            raise SVGInputError(f"quiescence_time must not be negative ({self.quiescence_time:g} Myr).")
        if(self.regime in ['rifting', 'compression'] and self.source['quiescence_time'] != 'default'):
            print(f"Warning: quiescence_time is not used in the {self.regime} regime.")

        self.magmatism = self._on_off('magmatism', magmatism)
        self.magmatism_extraction = self._on_off('magmatism_extraction', magmatism_extraction)
        self.magmatism_Ts0 = self._setting('magmatism_Ts0', magmatism_Ts0, 1080.0) #oC
        if(isinstance(self.magmatism_Ts0, str)):
            self.magmatism_Ts0 = parse_number_with_unit(self.magmatism_Ts0, {'oC': 1.0, 'C': 1.0}, 'magmatism_Ts0')
        self.magmatism_Ts0 = float(self.magmatism_Ts0)
        if(self.magmatism_extraction == 'on' and self.magmatism == 'off'):
            print("Warning: magmatism_extraction is on but magmatism is off.")

        preset = self._setting('preset_initial_temperature', preset_initial_temperature, None)
        if(preset is not None):
            preset = os.path.expanduser(str(preset).strip())
            if(self.source['preset_initial_temperature'] == 'SVG' and not os.path.isabs(preset)):
                preset = os.path.join(os.path.dirname(os.path.abspath(self.svg_file)), preset)
            preset = os.path.normpath(preset)
            if(not os.path.exists(preset)):
                raise SVGInputError(f"preset_initial_temperature not found: {preset}")
        self.preset_initial_temperature = preset

        solver = str(self._setting('temperature_solver', temperature_solver, '2D')).strip()
        self.temperature_solver = {'2d': '2D', 'columns': 'columns', '1d': 'columns'}.get(solver.lower())
        if(self.temperature_solver is None):
            raise SVGInputError(f"Unknown temperature_solver '{solver}'. Use 2D or columns.")

        #Velocity and time:
        #   rifting: extension, the moving boundary moves outwards with velocity
        #   compression: convergence, the moving boundary pushes the lithosphere inwards with velocity
        #   accordion (as in M300): the boundary velocity of the rifting regime is scaled in time by scale_bcv.txt
        #       0 - ti_convergence: extension
        #       ti_convergence - ti_quiescence: convergence (velocity x -1)
        #       ti_quiescence - tf_quiescence: quiescence (velocity x 0.01)
        #       tf_quiescence - time_max: extension again (dt_rifting2)
        #   quiescence: no velocity imposed at the boundaries (velocity_from_ascii = False)
        self.velocity_from_ascii = (self.regime != 'quiescence')
        self.variable_bcv = (self.regime == 'accordion')
        self.dt_quiescence = self.quiescence_time
        if(self.variable_bcv):
            self.tf_quiescence = self.ti_quiescence + self.dt_quiescence
            if(time_max is not None): #time_max sets the duration of the second extension
                self.dt_rifting2 = time_max - self.tf_quiescence
                if(self.dt_rifting2 <= 0):
                    raise SVGInputError(f"time_max = {time_max:g} Myr ends before the second extension starts ({self.tf_quiescence:g} Myr).")
            self.time_max = (self.tf_quiescence + self.dt_rifting2) * 1.0e6
        elif(time_max is not None):
            self.time_max = time_max * 1.0e6
        elif(self.regime == 'quiescence'):
            self.time_max = self.dt_quiescence * 1.0e6 if(self.dt_quiescence > 0) else 1000.0e6
        else: #rifting and compression
            self.time_max = 60.0e6

        self._print(f"Numerical experiment: {self.regime} ({self.source['regime']}), velocity = {self.velocity:g} cm/yr "
                    f"({self.source['velocity']}), moving boundary: {self.moving_boundary} ({self.source['moving_boundary']})")
        self._print(f"Magmatism: {self.magmatism} ({self.source['magmatism']}), extraction: {self.magmatism_extraction} "
                    f"({self.source['magmatism_extraction']}), Ts0 = {self.magmatism_Ts0:g} oC ({self.source['magmatism_Ts0']})")
        if(self.preset_initial_temperature is not None):
            self._print(f"Initial temperature from the preliminary scenario {self.preset_initial_temperature} "
                        f"({self.source['preset_initial_temperature']})")

    ###########################################################################
    # Profiles for the plots
    ###########################################################################

    ## I think its a very complex function for a simple task, all data is stored inside the fields dataset
    ## All these profiles could be obtained from the: MandyocBuilder.DTree.fields.ds
    def _find_profiles(self):
        '''
        Temperature and YSE profiles of the plots: one for each lithospheric domain (e.g. oceanic,
        non-cratonic, cratonic), i.e. each region with the same lithosphere thickness and the same unit at its
        base over at least min_profile_width (outside the weak seed). Each profile is at the centre of the widest
        region and is labelled with the title of the lithospheric mantle unit above the asthenosphere.
        Return a list of (column, label, color).
        '''
        layers, Nx = self.layers, self.Nx
        litho_km = np.round(self.thickness_litho / 1.0e3, 1)
        has_seed = np.zeros(Nx, dtype=bool)
        for i in layer_indices(layers, self.seed_label):
            has_seed |= slot_thickness(layers, i, slice(None)) > 0
        base_unit = np.array([next((layers[i].layer_label for i in range(self.idx_ast + 1, len(layers))
                                    if slot_thickness(layers, i, ix) > 0), '') for ix in range(Nx)])
        litho_key = np.array([f"{litho_km[ix]}|{base_unit[ix]}" if(not has_seed[ix]) else '' for ix in range(Nx)])
        edges = np.r_[0, np.nonzero(litho_key[1:] != litho_key[:-1])[0] + 1, Nx]
        regions = {} #(lithosphere thickness, unit at its base): (first column, last column + 1)
        for begin, end in zip(edges[:-1], edges[1:]):
            key = (litho_km[begin], base_unit[begin])
            if(litho_key[begin] != '' and (end - begin) * self.dx >= self.min_profile_width * self.Lx):
                if((key not in regions) or (end - begin > np.diff(regions[key])[0])):
                    regions[key] = (begin, end)

        profiles = []
        colors = self.profile_colors[-len(regions):] if(len(regions) <= len(self.profile_colors)) else self.profile_colors
        for k, ((thickness, unit), (begin, end)) in enumerate(sorted(regions.items())):
            ix = (begin + end - 1) // 2
            i_mantle = next((i for i in range(self.idx_ast + 1, len(layers)) if slot_thickness(layers, i, ix) > 0), self.idx_ast + 1)
            name = layers[i_mantle].title.replace(' lith. mantle', '').replace(' lithospheric mantle', '')
            profiles.append((ix, f"{name} ({thickness:.0f} km)", colors[k % len(colors)]))
        if(len(profiles) == 0):
            profiles = [(self.idx_profile, f"x = {self.x_profile/1.0e3:.0f} km", 'r')]
        return profiles

    ###########################################################################
    # Initial temperature
    ###########################################################################

    def build_temperature(self):
        '''
        Initial temperature (Nz, Nx) built with a TemperatureFieldBuilder (self.temperature_builder), whose
        lithosphere goes from the surface (base of the sticky air) to the top of the asthenosphere (LAB):

        - without preset_initial_temperature: steady-state conduction in the lithosphere, with the
          conductivity and the radiogenic heat of each layer, 0 oC at the surface and TP at the LAB, and the
          mantle adiabat below the LAB (TemperatureFieldBuilder.apply_linear_temperature). temperature_solver
          '2D' (default) solves the whole lithosphere at once (solve_heat_2D), so lateral conduction follows the
          geometry of the layers; 'columns' solves each column alone (solve_heat_1D);
        - with preset_initial_temperature: the final thermal state of the preliminary scenario adapted to the
          lithosphere of each column.

        The field is also the temperature of the builder (self.builder.DTree.fields.temperature).
        '''
        self.temperature_builder = TemperatureFieldBuilder(self.builder, LAB_interface=self.idx_ast,
                                                           surface_interface=self.surface_id, c_cap=self.ccapacity)
        if(self.preset_initial_temperature is None):
            self.temperature_builder.apply_linear_temperature(t_pot=self.TP)
            if(self.temperature_solver == '2D'):
                self.temperature_builder.solve_heat_2D()
            else:
                self.temperature_builder.solve_heat_1D()
        else:
            depth, T_preset, sa_preset, litho_preset, self.preset_source = read_preset_temperature(self.preset_initial_temperature)
            #Without the interfaces.txt of the preliminary scenario, its lithosphere is the most common one of this scenario
            if(litho_preset is None):
                values, counts = np.unique(np.round(self.thickness_litho / 1.0e3), return_counts=True)
                litho_preset = values[np.argmax(counts)] * 1.0e3
                print(f"Warning: interfaces.txt of the preliminary scenario not found; its lithosphere is assumed to be "
                      f"{litho_preset/1.0e3:.0f} km thick (the most common in {self.svg_file}).")
            self.thickness_litho_preset = litho_preset
            self.temperature_builder.t_pot = self.TP
            self.temperature_builder.temperature.values = adapt_preset_temperature(depth, T_preset, sa_preset, litho_preset, self.z,
                                                                                   self.thickness_sa, self.thickness_litho,
                                                                                   self.thickness_mantle)
            self._print(f"Preset of initial temperature: {self.preset_source}")
            self._print(f"\tsticky air of the preliminary scenario: {sa_preset/1.0e3:.0f} km; reference lithosphere: "
                        f"{litho_preset/1.0e3:.0f} km (other lithosphere thicknesses are adapted)")
        self.builder.create_temperature_field(self.temperature_builder)
        self.T = self.builder.DTree.fields.temperature.values #(Nz, Nx)
        return self.T

    ###########################################################################
    # Velocity boundary conditions
    ###########################################################################

    def build_velocity(self):
        '''
        Initial velocity (Nz, Nx) with the boundary conditions of the regime (None in the quiescence regime).
        Constant velocity down to h_v_const below the surface (reference thickness + 20 km, where the lithosphere
        is thickest in rifting and accordion, as in M300, and at the moving boundary in compression) and the return
        flow below it, so that the integral of the inflow is zero. In compression the return flow leaves through the
        moving boundary and the opposite boundary is fixed (VX = 0). With moving_boundary = symmetric (rifting and
        accordion), both boundaries move outwards with velocity/2, each with its own return flow.
        The profiles of the boundaries are given to a VelocityFieldBuilder (self.velocity_builder), which creates the
        velocity fields of the builder (self.builder.DTree.fields.vx and vz).
        '''
        if(not self.velocity_from_ascii):
            self.VX, self.VZ = None, None
            return None

        X, Z, Lx, Lz, Nx = self.X, self.Z, self.Lx, self.Lz, self.Nx
        idx_bc = 0 if(self.moving_boundary == 'left') else Nx - 1 #column of the moving boundary
        thickness_sa_bc = self.thickness_sa[idx_bc]

        if(self.h_v_reference == 'lithosphere'):
            thickness_ref_x = self.thickness_litho
        else:
            thickness_ref_x = self.thickness_mantle
        thickness_ref_bc = thickness_ref_x[idx_bc] if(self.regime == 'compression') else thickness_ref_x.max()

        vL = (0.5*self.velocity/100) / SEC_PER_YEAR  # m/s
        self.h_v_const = thickness_ref_bc + 20.0e3 #thickness with constant velocity
        ha = Lz - thickness_sa_bc - self.h_v_const
        vR = 2 * vL * (self.h_v_const + self.fac_air + ha) / ha #to ensure the integral equals zero

        VX = np.zeros_like(X)
        if(self.regime == 'compression'):
            #Convergence with the opposite boundary fixed (VX = 0): the moving boundary pushes the lithosphere
            #inwards (2*vL = velocity) down to h_v_const and all the inflow leaves through the same boundary in the
            #return flow below it. The return flow takes the inflow of both boundaries of the extensional profile,
            #so vR is doubled to keep the integral equal to zero.
            vR = 2 * vR
            x_bc = 0 if(self.moving_boundary == 'left') else Lx
            cond = X == x_bc
            VX[cond] = 2 * vL
            cond = (Z > self.h_v_const + thickness_sa_bc) & (X == x_bc)
            VX[cond] -= vR * (Z[cond] - self.h_v_const - thickness_sa_bc) / ha
            cond = Z <= thickness_sa_bc - self.fac_air
            VX[cond] = 0
            if(self.moving_boundary == 'right'):
                VX = -VX #the right boundary moves to the left
        elif(self.moving_boundary == 'symmetric'):
            #Extension with the reference frame at the centre of the domain: each boundary moves outwards with
            #vL (2*vL = velocity) down to h_v_const and has its own return flow below it (inflow), with ha and vR
            #of its own sticky air thickness, so that the integral is zero on each boundary.
            for x_bc, idx, sign in ((0, 0, -1), (Lx, Nx - 1, +1)):
                sa_bc = self.thickness_sa[idx]
                ha_bc = Lz - sa_bc - self.h_v_const
                vR_bc = 2 * vL * (self.h_v_const + self.fac_air + ha_bc) / ha_bc
                column = X == x_bc
                VX[column] = vL
                cond = column & (Z > self.h_v_const + sa_bc)
                VX[cond] -= vR_bc * (Z[cond] - self.h_v_const - sa_bc) / ha_bc
                VX[column & (Z <= sa_bc - self.fac_air)] = 0
                VX[column] *= sign
        else:
            cond = (Z > self.h_v_const + thickness_sa_bc) & (X == 0)
            VX[cond] = vR * (Z[cond] - self.h_v_const - thickness_sa_bc) / ha
            cond = (Z > self.h_v_const + thickness_sa_bc) & (X == Lx)
            VX[cond] = -vR * (Z[cond] - self.h_v_const - thickness_sa_bc) / ha
            cond = X == Lx
            VX[cond] += +2 * vL
            cond = Z <= thickness_sa_bc - self.fac_air
            VX[cond] = 0

            #The profile above is extensional, with the right boundary moving outwards (2*vL = velocity).
            #With the left moving boundary it is mirrored (in accordion, scale_bcv.txt reverses it in time):
            if(self.moving_boundary == 'left'):
                VX = -VX[:, ::-1] #extension with the left boundary moving outwards (to the left)

        ## there are methods inside VelocityFieldBuilder to integrate the velocity field
        ## and create a conservate field (with a tolerance)
          
        #vertical velocity at the top to balance the inflow of the discrete boundaries
        v0 = VX[(X == 0)]
        vf = VX[(X == Lx)]
        sv0 = np.sum(v0[1:-1]) + (v0[0] + v0[-1]) / 2.0
        svf = np.sum(vf[1:-1]) + (vf[0] + vf[-1]) / 2.0
        vv = -(svf - sv0) * self.dz / Lx
        VZ = np.zeros_like(X)
        VZ[Z == 0] = vv

        ## This code is redundat:
        ## A 2D field is created, the profiles are passed to the VelocityFieldBuilder
        ## and then the 2D fields are recreated again.

        #boundary profiles in the generic velocity builder, which creates the fields of the scenario builder
        v = VelocityFieldBuilder(self.builder, units='m/s')
        v.set_profile('left', vx=VX[:, 0], vz=VZ[:, 0])
        v.set_profile('right', vx=VX[:, -1], vz=VZ[:, -1])
        v.set_profile('bot', vx=VX[0, :], vz=VZ[0, :])
        v.set_profile('top', vx=VX[-1, :], vz=VZ[-1, :])
        self.builder.create_velocity_field(v)
        self.velocity_builder = v

        self.vL = vL
        self.VX = self.builder.DTree.fields.vx.values
        self.VZ = self.builder.DTree.fields.vz.values
        return self.VX, self.VZ

    def scale_bcv(self):
        '''Content of scale_bcv.txt in the accordion regime (Mandyoc multiplies the boundary velocity by each factor at the given time).'''
        if(not self.variable_bcv):
            return None
        if(self.dt_quiescence > 0):
            return f""" 4
                    {self.ti_convergence} -1.0
                    {self.ti_quiescence} 0.01
                    {self.tf_quiescence} -100.0
                    {self.time_max/1.0e6} 0.01
                    """
        #no quiescence: convergence followed directly by extension
        return f""" 3
                {self.ti_convergence} -1.0
                {self.ti_quiescence} -1.0
                {self.time_max/1.0e6} 0.01
                """

    ###########################################################################
    # param.txt
    ###########################################################################

    ## A good approach is to use the "set_params" method of MandyocBuilder
    ## I think it's complicated to assume default parameters for velocity, r_tol, particles, etc
    ## The users should set it for their own scenarios.
    ## also, all params (and some future params) are in `PARAMS_TYPES` in ..._variables.py
    def param_text(self, **params):
        '''
        Content of param.txt. Any parameter of the file can be changed with a keyword argument,
        e.g. param_text(particles_per_element=100, rheology_model=9).
        '''
        text = f"""
        nx = {self.Nx}
        nz = {self.Nz}
        lx = {self.Lx}
        lz = {self.Lz}
        # Simulation options
        multigrid                           = 1             # ok -> soon to be on the command line only
        solver                              = direct        # default is direct [direct/iterative]
        denok                               = 1e-14       # default is 1.0E-4
        particles_per_element               = 40          # default is 81
        surface_particles_per_element       = 40           # default is 2
        particles_perturb_factor            = 0.7           # default is 0.5 [values are between 0 and 1]
        rtol                                = 1.0e-7        # the absolute size of the residual norm (relevant only for iterative methods), default is 1.0E-5
        RK4                                 = Euler         # default is Euler [Euler/Runge-Kutta]
        Xi_min                              = 1.0e-5       # default is 1.0E-14
        random_initial_strain               = 0.3           # default is 0.0
        pressure_const                      = -1.0          # default is -1.0 (not used) - useful only in horizontal 2D models
        initial_dynamic_range               = True         # default is False [True/False]
        periodic_boundary                   = False         # default is False [True/False]
        high_kappa_in_asthenosphere         = False         # default is False [True/False]
        K_fluvial                           = 2.0e-7        # default is 2.0E-7
        m_fluvial                           = 1.0           # default is 1.0
        sea_level                           = 0.0           # default is 0.0
        basal_heat                          = 0.0          # default is -1.0
        # Surface processes
        sp_surface_tracking                 = True         # default is False [True/False]
        sp_surface_processes                = False        # default is False [True/False]
        plot_sediment                       = False         # default is False [True/False]
        a2l                                 = True          # default is True [True/False]
        free_surface_stab                   = True         # default is True [True/False]
        theta_FSSA                          = 0.5          # default is 0.5 (only relevant when free_surface_stab = True)
        # Time constrains
        step_max                            = 800000        # Maximum time-step of the simulation
        time_max                            = {self.time_max}    # Maximum time of the simulation [years]
        dt_max                              = 10000.0      # Maximum time between steps of the simulation [years]
        step_print                          = 50  # Make file every <step_print>
        sub_division_time_step              = 0.5           # default is 1.0
        initial_print_step                  = 0             # default is 0
        initial_print_max_time              = 1.0e6         # default is 1.0E6 [years]
        # Viscosity
        viscosity_reference                 = 1.0e26        # Reference viscosity [Pa.s]
        viscosity_max                       = 1.0e25        # Maximum viscosity [Pa.s]
        viscosity_min                       = 1.0e18        # Minimum viscosity [Pa.s]
        viscosity_per_element               = constant      # default is variable [constant/variable]
        viscosity_mean_method               = arithmetic      # default is harmonic [harmonic/arithmetic]
        viscosity_dependence                = pressure      # default is depth [pressure/depth]
        # External ASCII inputs/outputs
        interfaces_from_ascii               = True          # default is False [True/False]
        n_interfaces                        = {len(self.layers)-1}           # Number of interfaces int the interfaces.txt file
        variable_bcv                        = {self.variable_bcv} #False         # default is False [True/False]
        temperature_from_ascii              = True         # default is False [True/False]
        velocity_from_ascii                 = {self.velocity_from_ascii} #False      # default is False [True/False]
        binary_output                       = False         # default is False [True/False]
        sticky_blanket_air                  = True         # default is False [True/False]
        # precipitation_profile_from_ascii    = False         # default is False [True/False]
        # climate_change_from_ascii           = False         # default is False [True/False]
        print_step_files                    = True          # default is True [True/False]
        checkered                           = False         # Print one element in the print_step_files (default is False [True/False])
        geoq                                = on            # ok
        geoq_fac                            = 100.0           # ok
        # Physical parameters
        temperature_difference              = 1500.         # ok
        thermal_expansion_coefficient       = 3.28e-5       # ok
        thermal_diffusivity_coefficient     = 1.0e-6 #0.75e-6       #default is 1.0e-6        # ok
        gravity_acceleration                = 10.0          # ok
        density_mantle                      = 3300.         # ok
        external_heat                       = 0.0e-12       # ok
        heat_capacity                       = 1250.         # ok #default is 1250
        non_linear_method                   = on            # ok
        adiabatic_component                 = on            # ok
        radiogenic_component                = on            # ok
        magmatism                           = {self.magmatism}           # ok
        magmatism_extraction = {self.magmatism_extraction}
        magmatism_Ts0 = {self.magmatism_Ts0} #oC
        export_lithology = True
        magmatic_layer = {self.magmatic_layer}
        # Velocity boundary conditions
        top_normal_velocity                 = fixed         # ok
        top_tangential_velocity             = free          # ok
        bot_normal_velocity                 = fixed         # ok
        bot_tangential_velocity             = free          # ok
        left_normal_velocity                = fixed         # ok
        left_tangential_velocity            = fixed          # ok
        right_normal_velocity               = fixed         # ok
        right_tangential_velocity           = fixed         # ok
        surface_velocity                    = 0.0e-2        # ok
        multi_velocity                      = False         # default is False [True/False]
        # Temperature boundary conditions
        top_temperature                     = fixed         # ok
        bot_temperature                     = fixed         # ok
        left_temperature                    = fixed         # ok
        right_temperature                   = fixed         # ok
        rheology_model                      = 19             # ok

        T_initial                           = 3             # ok
        """
        lines = []
        used = set()
        for line in text.split("\n"):
            key = line.split("=")[0].strip()
            if("=" in line and not line.strip().startswith("#") and key in params):
                line = f"{key} = {params[key]}"
                used.add(key)
            lines.append(line)
        unknown = set(params) - used
        if(unknown):
            raise SVGInputError(f"Unknown parameter(s) of param.txt: {', '.join(sorted(unknown))}")
        return "\n".join(lines)

    ###########################################################################
    # Writing the files
    ###########################################################################

    ## The base text of the scripts should be in a dedicated file (like aux_variables), not here
    def write_scripts(self, output_dir='.', system=None, ncores=None, mandyoc_options='',
                      petsc_dir=None, petsc_arch=None, mandyoc=None):
        '''
        Write the scripts of the scenario in output_dir:
          - run_linux.sh or run_mac.sh: runs Mandyoc with mpirun of PETSc and then moves the outputs to one
            folder per variable with 0_organize_outputs.sh (on the Mac, caffeinate keeps it awake);
          - 0_organize_outputs.sh: copy of src/postrunning/0_organize_outputs.sh;
          - zipper.sh: zips the scenario in <folder>.zip, skipping the files already in the zip.

        Parameters
        ----------
        system : str, optional
            'linux' or 'mac'. Default: the system that runs tapioca.
        ncores : int, optional
            Number of MPI processes. Default: number of cores of this computer.
        mandyoc_options : str, optional
            Command line options of Mandyoc (e.g. '-seed 0,5,8 -strain_seed 0.0,1.0,1.0').
        petsc_dir, petsc_arch, mandyoc : str, optional
            PETSc folder and architecture (mpirun is $PETSC_DIR/$PETSC_ARCH/bin/mpirun) and the Mandyoc
            executable. Default: PETSC_DIR and PETSC_ARCH of the environment, or $HOME/opt/petsc and the
            only arch-* folder in it with mpirun (arch-label-optimized if there is none or more than one),
            and $HOME/opt/mandyoc/bin/mandyoc. Values set in the environment when the script runs override them.
        '''
        this_system = {'Linux': 'linux', 'Darwin': 'mac'}.get(platform.system())
        if(system is None):
            if(this_system is None):
                raise SVGInputError(f"Run scripts are written for {' or '.join(SYSTEMS)}; give system "
                                    f"(this computer is {platform.system()}).")
            system = this_system
        if(system not in SYSTEMS):
            raise SVGInputError(f"Unknown system '{system}'; use {' or '.join(SYSTEMS)}.")
        if(not os.path.isfile(ORGANIZE_SCRIPT)):
            raise SVGInputError(f"0_organize_outputs.sh not found in {os.path.dirname(os.path.abspath(ORGANIZE_SCRIPT))}.")

        if(petsc_dir is None):
            petsc_dir = os.environ.get('PETSC_DIR', '$HOME/opt/petsc')
        if(petsc_arch is None):
            petsc_arch = os.environ.get('PETSC_ARCH')
        if(petsc_arch is None):
            #the arch of $HOME/opt/petsc, only if the script is for this computer
            archs = glob.glob(os.path.join(os.path.expanduser('~'), 'opt', 'petsc', 'arch-*', 'bin', 'mpirun'))
            petsc_arch = (archs[0].split(os.sep)[-3] if(system == this_system and len(archs) == 1)
                          else 'arch-label-optimized')
        if(mandyoc is None):
            mandyoc = '$HOME/opt/mandyoc/bin/mandyoc'
        if(ncores is None):
            ncores = os.cpu_count() or 1

        os.makedirs(output_dir, exist_ok=True)
        path = lambda fname: os.path.join(output_dir, fname)

        run = RUN_SCRIPT
        for key, value in [('PETSC_DIR', petsc_dir), ('PETSC_ARCH', petsc_arch), ('MANDYOC', mandyoc),
                           ('NCORES', int(ncores)), ('MANDYOC_OPTIONS', mandyoc_options),
                           ('LAUNCHER', RUN_LAUNCHER[system])]:
            run = run.replace(f'@{key}@', str(value))
        scripts = {f'run_{system}.sh': run, 'zipper.sh': ZIP_SCRIPT}
        with open(ORGANIZE_SCRIPT) as f:
            scripts['0_organize_outputs.sh'] = f.read()

        for fname, text in scripts.items():
            with open(path(fname), 'w') as f:
                f.write(text)
            os.chmod(path(fname), 0o755)
        if(self.verbose):
            print(f"Scripts: {', '.join(scripts)} ({system}, {int(ncores)} cores)")

    def write_all(self, output_dir='.', plots=True, copy_svg=True, scripts=True, system=None, ncores=None,
                  mandyoc_options='', **params):
        '''
        Write the input files of Mandyoc in output_dir: interfaces.txt, param.txt, input_temperature_0.txt,
        input_velocity_0.txt, vel_bc.txt, velz_bc.txt and scale_bcv.txt (accordion), the scenario infos
        (infos_<folder>.txt), a copy of the SVG and the figures numerical_setup.png and initial_temperature.png.
        With scripts, also the run script of Mandyoc for system ('linux' or 'mac'), 0_organize_outputs.sh and
        zipper.sh (see write_scripts).
        Keyword arguments change parameters of param.txt (see param_text).
        '''
        os.makedirs(output_dir, exist_ok=True)
        path = lambda fname: os.path.join(output_dir, fname)

        if(copy_svg):
            target = path(os.path.basename(self.svg_file))
            if(not os.path.exists(target) or not os.path.samefile(self.svg_file, target)):
                shutil.copy(self.svg_file, target)

        builder = self.builder
        builder.path = output_dir
        builder.export_interfaces(export='none')
        builder.set_params(params_str=self.param_text(**params))
        builder.export_params()

        if(self.T is None):
            self.build_temperature()
        builder.export_field('temperature', header="T1\nT2\nT3\nT4")

        if(self.velocity_from_ascii):
            if(self.VX is None):
                self.build_velocity()
            #fields of the builder, which may have been changed by script after build_velocity
            self.VX, self.VZ = builder.DTree.fields.vx.values, builder.DTree.fields.vz.values
            X = self.X
            v0 = self.VX[(X == 0)]
            vf = self.VX[(X == self.Lx)]
            np.savetxt(path("vel_bc.txt"), np.array([v0, vf]).T)
            np.savetxt(path("velz_bc.txt"), self.VZ[(self.z == 0)].T)
            builder.export_field('velocity', header="v1\nv2\nv3\nv4")
            if(self.variable_bcv):
                _write_clean(path("scale_bcv.txt"), self.scale_bcv())

        if(scripts):
            self.write_scripts(output_dir, system=system, ncores=ncores, mandyoc_options=mandyoc_options)

        if(plots):
            self.plot_temperature(path('initial_temperature.png'))
            self.plot_setup(path('numerical_setup.png'))

        name = os.path.basename(os.path.abspath(output_dir))
        infos = self.infos(name)
        np.savetxt(path(f'infos_{name}.txt'), infos, fmt="%s")
        if(self.verbose):
            for line in infos:
                print(line)

    ###########################################################################
    # Scenario infos
    ###########################################################################

    ## This is a useful function that could be inside the MandyocBuilder class
    def infos(self, name=''):
        '''Description of the scenario (list of lines).'''
        layers, ix = self.layers, self.idx_profile
        lines = ['SCENARIO INFOS:', ' ', f'Name: {name}', ' ', 'Domain parameters:',
                 f"\tLx: {self.Lx*1.0e-3} km", f"\tLz: {self.Lz*1.0e-3} km", f"\tNx: {self.Nx}", f"\tNz: {self.Nz}",
                 f"Resolution dx x dz: {1.0e-3*self.dx} x {1.0e-3*self.dz} km2", ' ',
                 f'Time limit: {self.time_max/1.0e6} Myr']
        if(self.variable_bcv):
            lines += [f"Extension: 0 - {self.ti_convergence} Myr",
                      f"Convergence: {self.ti_convergence} - {self.ti_quiescence} Myr",
                      f"Time of quiescence: {self.dt_quiescence} Myr ({self.ti_quiescence} - {self.tf_quiescence} Myr)",
                      f"Second extension: {self.dt_rifting2} Myr ({self.tf_quiescence} - {self.time_max/1.0e6} Myr)", ' ']
        lines.append(f'Layers (bottom to top) from {self.svg_file}, thickness at x = {self.x_profile/1.0e3:.0f} km:')
        for i, layer in enumerate(layers):
            lines.append(f"\t{i}: {layer.layer_label} ({layer.rheology_name}): h = {slot_thickness(layers, i, ix)/1.0e3:.1f} km, "
                         f"C = {layer.effective_viscosity_scale_factor}, rho = {layer.density} kg/m3, H = {layer.radiogenic_heat_production:.4g} W/kg")
        lines.append(f"\tlithosphere: {self.thickness_litho[ix]/1.0e3:.1f} km")
        lines.append(f"\tmagmatic layer: {self.magmatic_layer} ({layers[self.magmatic_layer].layer_label})")
        lines.append("Temperature and YSE profiles in numerical_setup.png:")
        for ixp, label, color in self.profiles:
            lines.append(f"\t{label}: x = {self.x[ixp]/1.0e3:.0f} km")
        lines += [' ', f"Numerical experiment: {self.regime}", f"Velocity field: {self.velocity_from_ascii}"]
        if(self.velocity_from_ascii and self.VX is not None):
            lines.append(f"inital velocity: {self.vL*SEC_PER_YEAR*2*100} cm/yr")
            lines.append(f"\tmoving boundary: {self.moving_boundary}, constant velocity down to {self.h_v_const/1.0e3:.0f} km below the surface")
        lines += [f"Variable velocity field: {self.variable_bcv}", 'Initial temperature field setup:',
                  f"\tPreset of initial temperature field: {self.preset_initial_temperature is not None}"]
        if(self.preset_initial_temperature is None):
            lines += [f"\tSteady-state conduction in the lithosphere ({self.temperature_solver} solver)",
                      f"\tMantle potential temperature (at the LAB): {self.TP} oC"]
        elif(hasattr(self, 'preset_source')):
            lines += [f"\tPreliminary scenario: {self.preset_source}",
                      f"\tReference lithosphere of the preliminary scenario: {self.thickness_litho_preset/1.0e3:.0f} km"]
        lines += [' ', f'magmatism: {self.magmatism}, extraction: {self.magmatism_extraction}, Ts0: {self.magmatism_Ts0:g} oC']
        lines.append('Settings (argument, SVG or default): ' + ', '.join(f"{k}: {v}" for k, v in self.source.items()))
        return lines

    ###########################################################################
    # Plots
    ###########################################################################

    def yield_strength_envelope(self, ix, sr=1.0e-15):
        '''YSE [Pa] along the vertical profile at column ix (strain rate sr [1/s]), from MandyocBuilder.calculate_YSE.'''
        if(self.T is None):
            self.build_temperature()
        fields = self.builder.DTree.fields
        if('YSE' not in fields or fields.YSE.attrs.get('strain_rate') != sr):
            self.builder.calculate_YSE(sr=sr)
        return self.builder.DTree.fields.YSE.isel(x=ix).values

    ## It's a good function to be inside the TemperatureFieldBuilder
    ## This function should get **kwargs to set the plot
    ## TemperatureFieldBuilder has all interfaces encapsulated
    def plot_temperature(self, fname='initial_temperature.png'):
        '''Initial temperature field with the 600, 1000 and 1300 oC isotherms and the layer interfaces.'''
        T = self.T if(self.T is not None) else self.build_temperature()
        X, Z, x = self.X, self.Z, self.x
        sa = self.thickness_sa[self.idx_profile]

        fig, ax0 = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(16, 6))
        im = ax0.contourf(X / 1.0e3, (-Z + sa) / 1.0e3, T, levels=np.arange(0, np.max(T) + 100, 100), cmap='inferno')
        isotherms = ax0.contour(X / 1.0e3, (-Z + sa) / 1.0e3, T, levels=[600, 1000, 1300], colors='w', linewidths=1.0, linestyles='--')
        ax0.clabel(isotherms, fmt='%d °C', fontsize=10)
        for layer in self.layers[1:]:
            ax0.plot(x / 1.0e3, (-layer.interface + sa) / 1.0e3, color='k', lw=0.6)
        ax0.set_xlim(0, self.Lx / 1.0e3)
        ax0.set_ylim(-self.Lz/1000 + sa/1000, 0 + sa/1000)
        ax0.tick_params(labelsize=12)
        ax0.set_xlabel("Distance [km]", fontsize=14)
        ax0.set_ylabel("Depth [km]", fontsize=14)
        ax0.set_title(f"Initial temperature field ({os.path.basename(self.svg_file)})", fontsize=14)
        cbar = fig.colorbar(im, orientation='vertical', ax=ax0, pad=0.01)
        cbar.set_label("Temperature [°C]", fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    def plot_setup(self, fname='numerical_setup.png'):
        '''
        Numerical setup: layers with the colours of the SVG, legend with the properties, weak seed,
        temperature and YSE profiles of each lithospheric domain, friction angle and velocity boundary conditions.
        '''
        T = self.T if(self.T is not None) else self.build_temperature()
        if(self.velocity_from_ascii and self.VX is None):
            self.build_velocity()
        with plt.rc_context({'xtick.labelsize': 18, 'ytick.labelsize': 18}):
            self._plot_setup(fname, T)

    def _plot_setup(self, fname, T):
        layers, x, z, Lx, Lz, Nx = self.layers, self.x, self.z, self.Lx, self.Lz, self.Nx
        sa = self.thickness_sa[self.idx_profile]
        dx, dz = self.dx, self.dz

        fig, axs = plt.subplots(1, 1, figsize=(14, 6))
        ylimplot = [-Lz/1000 + sa/1000, 0 + sa/1000]

        for i, layer in enumerate(layers):
            top = layers[i+1].interface if i < len(layers)-1 else np.zeros(Nx)
            axs.fill_between(x/1000, (-layer.interface + sa)/1000, (-top + sa)/1000, color=layer.color, lw=0)

        axs.set_xlim(0, Lx/1000)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_ylim(ylimplot)
        axs.set_xlabel(f"Distance = {Lx/1000:.0f} km\nResolution = {dx/1000:.0f} km", fontsize=16)
        axs.set_ylabel(f"Depth = {Lz/1000:.0f} km\nResolution = {dz/1000:.0f} km", fontsize=16)

        #legend: one entry per label, from top to bottom; thickness h:
        #   bodies (units that do not reach both sides of the domain, e.g. a basin): maximum thickness (depocentre)
        #   bottom unit (asthenosphere): range of thickness along x
        #   other layers: most common thickness along x (their reference thickness)
        legend_layers = {}
        for layer in layers[::-1]:
            if(layer.layer_label != self.seed_label):
                legend_layers.setdefault(layer.layer_label, layer)

        def legend_thickness(label):
            th = layer_thickness(layers, label, slice(None)) / 1.0E3
            if(th[0] <= 0 or th[-1] <= 0):
                return f"{th.max():.0f} km"
            if(label == layers[0].layer_label and np.round(th.min()) != np.round(th.max())):
                return f"{th.min():.0f}-{th.max():.0f} km"
            values, counts = np.unique(np.round(th[th > 0], 1), return_counts=True)
            return f"{values[np.argmax(counts)]:.0f} km" if(len(values)) else "0 km"

        legend_elements = []
        for label, layer in legend_layers.items():
            legend_label = (f"{layer.title}\n{layer.effective_viscosity_scale_factor:g} x {layer.rheology_name.replace('_', ' ')}\n"
                            + fr"$\rho$ = {layer.density:.0f} kg/m³" + f"\n$h$ = {legend_thickness(label)}")
            legend_elements.append(Patch(facecolor=layer.color, edgecolor='black', label=legend_label))
        fig.subplots_adjust(bottom=0.25)
        axs.legend(handles=legend_elements, ncol=len(legend_elements), loc='lower center', bbox_to_anchor=(0.5, -0.45),
                   frameon=False, title='Layers properties', title_fontsize=15, fontsize=8, columnspacing=1.0)


        ## Idea: get the weak seed from the MandyocBuilder.DTree.interfaces, where weakening_seed > 0
        #weak seed (one annotation per body, i.e. per group of contiguous columns)
        idx_seed = layer_indices(layers, self.seed_label)
        if(len(idx_seed) > 0):
            seed_bases = np.array([layers[i].interface for i in idx_seed])
            seed_tops = np.array([layers[i+1].interface if i < len(layers)-1 else np.zeros(Nx) for i in idx_seed])
            seed_present = seed_bases > seed_tops
            seed_top = np.where(seed_present, seed_tops, np.inf).min(axis=0)
            seed_base = np.where(seed_present, seed_bases, -np.inf).max(axis=0)
            seed_cols = seed_present.any(axis=0)
            seed_title = layers[idx_seed[0]].title.lower()
            edges = np.diff(np.r_[0, seed_cols.astype(int), 0])
            for begin, end in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
                cols = slice(begin, end)
                x_seed = np.mean(x[cols])
                width_seed = (end - begin) * dx
                thickness_seed = np.max(seed_base[cols] - seed_top[cols])
                axs.text(x_seed/1000, -(np.max(seed_base[cols]) - sa)/1000 - 2,
                         f'{seed_title}\n{width_seed/1000:.0f}x{thickness_seed/1000:.0f} km²',
                         color='k', fontsize=12, ha='center', va='top')

        #temperature profiles
        axt = axs.inset_axes((0.205, 0, 0.18, 1))
        for ix, label, color in self.profiles:
            axt.plot(T[:, ix], (-z + sa) / 1.0e3, "-", color=color, label=label)
            axs.plot(x[ix] / 1.0e3, sa / 1.0e3 * 0.15, 'v', color=color, ms=6) #position of the profile
        axt.legend(loc='lower left', fontsize=8, framealpha=0.9)
        axt.grid(visible=True, axis='x', which='both', ls='--', color='red', alpha=0.3)
        axt.set_ylim(ylimplot)
        axt.set_yticks([])
        axt.set_xticks(np.linspace(0, 1800, 7))
        axt.patch.set_alpha(0)
        axt.xaxis.set_ticks_position('top')
        axt.tick_params(labelsize=8)
        axt.tick_params(axis='x', colors='k')
        axt.spines['left'].set_visible(False)
        axt.spines['right'].set_visible(False)
        axt.set_title('Temperature [°C]', color='k')

        #yield strength envelopes
        axsg = axs.inset_axes((0.605, 0, 0.13, 1))
        for ix, label, color in self.profiles:
            axsg.plot(self.yield_strength_envelope(ix)/1e9, -(z - sa)/1e3, '-', color=color, label=label)
        axsg.grid(visible=True, axis='x', which='both', ls='--', color='gray', alpha=0.8)
        axsg.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axsg.set_xlim(-0.1, 1.1)
        axsg.set_ylim(ylimplot)
        axsg.set_yticks([])
        axsg.patch.set_alpha(0)
        axsg.xaxis.set_ticks_position('top')
        axsg.spines['left'].set_visible(False)
        axsg.spines['right'].set_visible(False)
        axsg.set_title(r'$\sigma_{YSE}$ [GPa]')
        axsg.tick_params(labelsize=8)

        #effective friction angle
        axsf = axs.inset_axes((0.800, 0, 0.10, 0.30))
        axsf.patch.set_alpha(0.4)
        fisize = 10
        axsf.plot(np.array([0, 0.25, 0.75, 1.0]), np.array([15, 15, 2, 2]), 'k-')
        axsf.set_xlim([0.10, 0.9])
        axsf.set_ylim([0, 17])
        axsf.set_xticks([0.25, 0.5, 0.75])
        axsf.set_xticklabels([0.05, ' ', 1.05])
        axsf.tick_params('x', top=True, labeltop=False)
        axsf.xaxis.set_ticks_position('bottom')
        axsf.set_xlabel(r"$\varepsilon$", fontsize=fisize)
        axsf.set_yticks([2, 15])
        axsf.set_yticklabels(['2°', '15°'])
        axsf.set_ylabel(r'$\Phi_{\mathrm{eff}}$', fontsize=fisize)
        axsf.tick_params(labelsize=fisize)
        axsfC = axsf.twinx()
        axsfC.patch.set_visible(False) #the background of the twin axes would hide the curve
        axsfC.set_ylim([0, 17])
        axsfC.set_yticks([2, 15])
        axsfC.set_yticklabels([4, 20])
        axsfC.set_ylabel('Cohesion [MPa]', fontsize=fisize, rotation=90)
        axsfC.tick_params(labelsize=fisize)

        #velocity boundary conditions: insets centred on the edges, so v = 0 is on the boundary
        if(self.velocity_from_ascii):
            scale = 100.0 * SEC_PER_YEAR #m/s to cm/yr
            fac = 0.96
            for column, origin, title in ((-1, fac, f'v = {self.velocity} [cm/y]'), (0, -(1-fac), None)):
                vmax = np.round(max(abs(self.VX[:, column] * scale)), 0)
                vmax = vmax if(vmax > 0) else 1.0
                axv = axs.inset_axes((origin, 0, (1-fac)*2, 1))
                #only the hatch is drawn: the hatch takes the edge color (with alpha=0 it is also invisible in matplotlib >= 3.11)
                axv.fill_betweenx((-z + sa) / 1.0e3, self.VX[:, column] * scale, 0, facecolor='none', edgecolor='k', linewidth=0, hatch='---')
                axv.set_ylim(ylimplot)
                axv.set_yticks([])
                axv.set_xticks([])
                axv.patch.set_alpha(0)
                if(title):
                    axv.set_title(title, fontsize=8)
                axv.set_xlim([-vmax, vmax])
                for side in ['left', 'right', 'bottom'] + (['top'] if(title is None) else []):
                    axv.spines[side].set_visible(False)
            #the hatches of the left boundary go beyond the domain where VX < 0 (clipped by the inset): the y label
            #(its right side is at the given x) is 0.5% of the width to the left of them
            vmin_left = max(-self.VX[:, 0].min() * scale, 0.0)
            hatch_left = (1 - fac) * min(vmin_left / vmax, 1.0)
            axs.yaxis.set_label_coords(-hatch_left - 0.005, 0.5)

        fig.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close(fig)
