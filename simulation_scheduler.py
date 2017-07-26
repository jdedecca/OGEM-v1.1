## Copyright 2017 João Gorenstein Dedecca

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = """
Joao Gorenstein Dedecca
"""

__copyright__ = """
Copyright 2017 João Gorenstein Dedecca, GNU GPL 3
"""

import sys, os
from OGEM import Load_Parameters
from simulation import main

def execute_run(data):
    """ Calls each run for a given run schedule.
        Separate function to allow multiprocessing.
    """

    main(data)

def schedule():
    """" Creates output files, sets up multiprocessing and calls runs """

    # All runs should be placed in Simulations file and each folder must start with OGEM
    # The simulation_scheduler shell command arguments specify the simulations to run.
    # If no arguments are passed the simulation_scheduler runs all folders in Simulations.
    if len(sys.argv) == 1:
        runs = []
        for (dirpath, dirnames, filenames) in os.walk("input"):
            runs.extend(dirnames)
    else:
        runs = sys.argv[1:]
    print(runs)

    data = []
    for run in runs:
        if run[:4] == "OGEM": # Run schedules must start with "OGEM".
            parameters = Load_Parameters(run)
            data.append(run)

    for rn, run in enumerate(runs):
        if run[:4] == "OGEM":  # Run schedules must start with "OGEM".
            sys.stdout = sys.__stdout__
            print("Running", str(data[rn]))  # Indicate current run schedule.
            execute_run(data[rn])

if __name__ == '__main__':
    schedule()

