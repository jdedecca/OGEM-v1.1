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

import os, sys, logging, shutil
from OGEM import Period_Run, Network_Setup, Load_Parameters, Load_Data
import pandas as pd
import numpy as np
import pypsa
from pyutilib.services import TempfileManager

class Logger(object):
    """" The Logger class allows for parallel output to the console and a log.dat file.
        Thanks to Triptych in http://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python
    """

    def __init__(self,simulation_name):
        self.terminal = sys.__stdout__
        self.log = open(os.path.join(os.getcwd(),"network",simulation_name,"log.dat"), "a",1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main(run_name):
    """ Executes a single run of the exploratory model """
    # Make some output nicer.
    pd.set_option("precision", 2)
    pd.set_option("max_rows", 50)
    pd.set_option("max_columns", 70)
    pd.set_option('display.width', 200)
    pd.options.display.float_format = '{:.2e}'.format
    np.set_printoptions(precision=3)
    logging.getLogger().setLevel(logging.WARNING) # Increase the logging thresholder for PyPSA

    parameters = Load_Parameters(run_name)

    # Increment path number to find a non-existing folder
    path = os.path.join(os.getcwd(), r"network",parameters['simulation_name'])
    while os.path.exists(path):
        parameters['simulation_name'] = parameters['simulation_name'][:-1] + str(int(parameters['simulation_name'][-1]) + 1)
        path = os.path.join(os.getcwd(), r"network", parameters['simulation_name'])

    print("Running {}".format(parameters["simulation_name"]))

    TempfileManager.tempdir = path #Indicates temp directory, e.g. for solver files
    os.mkdir(path)

    for script in ['OGEM.py','simulation.py']:
        shutil.copy(script,path) #Back-up code version

    for period in range(parameters['periods']):
        path = os.path.join(os.getcwd(), r"network", parameters['simulation_name'],"p"+str(period))
        os.mkdir(path)

    if sys.gettrace() is None: # Only duplicate stream to log file if not in debugging mode.
        sys.stdout = Logger(parameters['simulation_name'])

    input_panel = Load_Data(run_name) # Loads setup data
    # The pathway dataframes constain the transmission and offshore wind expansion capacity for all period
    expansion_pathway = pd.DataFrame()
    generator_pathway = pd.DataFrame()

    # Network setup
    branches_dataframe, network = Network_Setup(0, input_panel)

    # Main loop for expansion simulation.
    def Single_Period():
        parameters["period"] = period
        selected_expansions,selected_generators = Period_Run(period, network, branches_dataframe, input_panel)
        return selected_expansions.loc[:,["s_nom_opt"]].rename(columns = {"s_nom_opt":"period"+str(period)}),selected_generators.loc[:,["p_nom_opt"]].rename(columns = {"p_nom_opt":"period"+str(period)})

    for period in range(parameters["periods"]):
        selected_expansions, selected_generators = Single_Period()
        expansion_pathway = expansion_pathway.join(selected_expansions,how='outer').fillna(0)
        generator_pathway = generator_pathway.join(selected_generators,how = 'outer').fillna(0)

    expansion_pathway.to_csv(path_or_buf=os.path.join(os.getcwd(), r"network",parameters['simulation_name'],"expansion_pathway.csv"))
    np.save(os.path.join(os.getcwd(),r"network",parameters['simulation_name'],'parameters.npy'),parameters)
    print(generator_pathway)
    print(expansion_pathway)

    if sys.gettrace() == None:
        sys.stdout.log.close() #Close log.dat file if not in debugging mode.

if __name__=='__main__':
    main()