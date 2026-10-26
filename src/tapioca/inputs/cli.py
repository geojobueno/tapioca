'''
Command line interface of the scenario builder:

    tapioca-svg scenario.svg [-o output_dir] [--dx 1] [--dz 1] [--regime rifting] [--velocity 1]
                [--moving-boundary right] [--quiescence-time 300] [--time-max 500]
                [--magmatism off] [--magmatism-extraction off] [--magmatism-Ts0 1080]
                [--preset-initial-temperature path/to/preliminary] [--no-plots]
                [--system linux] [--ncores 20] [--mandyoc-options="-seed 0,5,8 -strain_seed 0.0,1.0,1.0"] [--no-scripts]

The options override the settings of the SVG (description of the shape labelled "domain").
'''

import argparse
import sys

from .svg import SVGInputError
from .scenario import SVGScenario, REGIMES, SYSTEMS

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(prog='tapioca-svg',
                                     description='Create the Mandyoc input files of a scenario drawn in an SVG file.')
    parser.add_argument('svg_file', help='SVG file with the scenario')
    parser.add_argument('-o', '--output-dir', default='.', help='folder of the input files (default: current folder)')
    parser.add_argument('--dx', type=float, help='horizontal resolution [km] (default: from the domain shape of the SVG)')
    parser.add_argument('--dz', type=float, help='vertical resolution [km] (default: from the domain shape of the SVG)')
    parser.add_argument('--regime', choices=REGIMES, help='tectonic regime of the numerical experiment (default: from the SVG, or rifting)')
    parser.add_argument('--velocity', type=float, help='velocity at the boundary [cm/yr] (default: from the SVG, or 1 cm/yr)')
    parser.add_argument('--moving-boundary', choices=['left', 'right'], help='boundary that moves (default: from the SVG, or right)')
    parser.add_argument('--quiescence-time', type=float, help='duration of the quiescence [Myr] in the accordion and quiescence '
                                                              'regimes (default: from the SVG, or 0 Myr)')
    parser.add_argument('--time-max', type=float, help='duration of the simulation [Myr] (default: from the SVG, or 60 Myr in rifting '
                                                       'and compression, the end of the second extension in accordion and the '
                                                       'quiescence time (or 1000 Myr) in quiescence)')
    parser.add_argument('--magmatism', choices=['on', 'off'], help='melting in Mandyoc (default: from the SVG, or off)')
    parser.add_argument('--magmatism-extraction', choices=['on', 'off'],
                        help='extraction of the melt to a basalt layer below the sticky air (default: from the SVG, or off)')
    parser.add_argument('--magmatism-Ts0', type=float, help='magmatism_Ts0 in param.txt [oC] (default: from the SVG, or 1080)')
    parser.add_argument('--preset-initial-temperature', help='folder (or _output_temperature.nc) of a preliminary scenario: the '
                                                             'horizontal mean of its final temperature, adapted to the lithosphere '
                                                             'of the SVG, is the initial temperature (default: from the SVG, or none)')
    parser.add_argument('--no-plots', action='store_true', help='do not create numerical_setup.png and initial_temperature.png')
    parser.add_argument('--system', choices=SYSTEMS, help='system of the run script of Mandyoc (default: the system that runs tapioca)')
    parser.add_argument('--ncores', type=int, help='number of MPI processes in the run script (default: number of cores of this computer)')
    parser.add_argument('--mandyoc-options', default='', help='command line options of Mandyoc in the run script')
    parser.add_argument('--no-scripts', action='store_true', help='do not create the run script, 0_organize_outputs.sh and zipper.sh')
    args = parser.parse_args(argv)

    try:
        scenario = SVGScenario(args.svg_file, dx=args.dx, dz=args.dz, regime=args.regime, velocity=args.velocity,
                               moving_boundary=args.moving_boundary, quiescence_time=args.quiescence_time,
                               time_max=args.time_max, magmatism=args.magmatism,
                               magmatism_extraction=args.magmatism_extraction, magmatism_Ts0=args.magmatism_Ts0,
                               preset_initial_temperature=args.preset_initial_temperature)
        scenario.write_all(args.output_dir, plots=not args.no_plots, scripts=not args.no_scripts, system=args.system,
                           ncores=args.ncores, mandyoc_options=args.mandyoc_options)
    except SVGInputError as err:
        sys.exit(f"Error: {err}")


if __name__ == '__main__':
    main()
