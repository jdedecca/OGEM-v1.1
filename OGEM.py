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

import os, sys, csv, warnings, gc, datetime,psutil
import numpy as np
import pandas as pd
import pypsa
from pyclustering.cluster.kmedoids import kmedoids
import multiprocessing

def Period_Run(period, network, network_branches, input_panel):
    """ Runs one expansion period.
        First updates all period-related network and time series data.
    """

    print("Period {} run at {:%H:%M}...".format(period,datetime.datetime.now()))

    Network_Update(period, network, network_branches, input_panel)

    Time_Series_Update(period, network, input_panel)

    CBA_data = Expansions_Simulation(period, network_branches, network, input_panel)

    return CBA_data["optimal expansions"], CBA_data["optimal generators"]

def Model_Check(network):
    """ Check for clear network inconsistencies after constructing the reduced network but before solver run.
    """

    # CSP and PSP technologies are storage units, while Demand is load.
    if any(network.generators.carrier.isin(["Demand","CSP","PSP"])):
        print("Warning: Wrong carrier present in generators")
        sys.exit()

    # The reduced network should not have 0 p_nom generators.
    if any((~ network.generators["p_nom_extendable"]) & (network.generators["p_nom"] == 0)):
        print("Warning: Non-extendable generators with 0 P_nom present")
        sys.exit()

    # Any NA marginal cost, transmission capacity or time-variant value will crash the solver
    if network.generators["marginal_cost"].isnull().values.any() or network.storage_units["marginal_cost"].isnull().values.any():
        print("NA marginal cost")
        sys.exit()

    for gen,value in network.generators_t.items():
        if value.isnull().sum().sum() > 0:
            print("Warning: NA items in generators_t {}".format(gen))
            sys.exit()

    for load, value in network.loads_t.items():
        if value.isnull().sum().sum() > 0:
            print("Warning: NA items in loads_t {}".format(load))
            sys.exit()

    if any(network.lines.s_nom.isnull()):
        print("Warning: Null lines s_nom")
        sys.exit()

    if any(network.links.p_nom.isnull()):
        print("Warning: Null links p_nom")
        sys.exit()


def Results_Check(network):
    """ Check output after solve (LP and MILP).
    """
    if network.buses_t.marginal_price.isnull().values.any():
        print("Warning: Non-valid bus prices")
        print(network.buses_t.marginal_price.loc[:,network.buses_t.marginal_price.min()<0])

    if network.buses_t.marginal_price.max().max() == 0:
        print("Warning: Null bus prices")

    # Negative marginal prices are possible due to flow constraints but unlikely.
    # Clip negative prices and warn users of ocurrence.
    if network.buses_t.marginal_price.min().min() < -1e6:
        print("Warning: Negative bus prices. Clipping prices to $0")
        print(network.buses_t.marginal_price.loc[:,network.buses_t.marginal_price.min() < 0])

    print("Clipping negative prices...")
    network.buses_t.marginal_price = network.buses_t.marginal_price.clip(lower = 0)


def MILP_OPF(period,network,OGEM_options,base_welfare=None):
    """ Run optimal power flow for current period.
        Only MILP investment runs should have base_welfare.
    """

    print("OPF call at {:%H:%M}...".format(datetime.datetime.now()))

    """ CPLEX simulation parameters. Adapt as necessary for solver. See the CPLEX parameters reference for further detail.
        Pyomo uses the CPLEX interactive solver parameters, substituting spaces with underscores

        Base and final (linear) parameters:
        lpmethod: set to 4 to use the barrier method, which is faster for the OGEM system size.
        barrier_crossover: Disable barrier cross-over to find basis solution after barrier solve to reduce solve time. A basis solution is not necessary as the solution is not used for a warmstart.
        simplex_tolerances_optimality and _feasibility: Main tolerances in case simplex algorithm is used.
        threads: limit threads to 8 (for HPCs) or the available on the machine, leaving one for other user applications.
        barrier_algorithm: 0, set to default (solver chooses).
        barrier_startalg: 1, set to default (dual is 0).

        Expansion (mixed-integer linear) parameters:
        simplex_tolerances_optimality and _feasibility: tolerances for relaxed subproblems of b&c algorithm if used.
        mip_tolerances_mipgap: 0.5%, main MILP tolerance parameter.
        mip_tolerances_absmipgap: absolute MILP tolerance parameter, not used in practice.
        mip_strategy_subalgorithm: Linear algorithm for relaxed subproblems, 4 = barrier.
        barrier_convergetol: Tolerance for barrier subproblem algorithm.
    """

    solver_name = 'cplex'

    # Adapt number of maximum parallel cores according to HPC cluster
    max_cores = 16 if multiprocessing.cpu_count() == 16 else 8 if multiprocessing.cpu_count() == 32 else 8
    print('Max cores:', multiprocessing.cpu_count())
    if (OGEM_options["reference"] == "base"):
        if solver_name == 'cplex':
            solver_options = {'workdir': os.path.join(os.getcwd(),'network',parameters["simulation_name"]),'simplex_tolerances_optimality': 1e-4, 'simplex_tolerances_feasibility': 1e-4,'barrier_convergetol':1e-4, 'lpmethod': 4,'barrier_crossover': parameters["barrier_crossover"], 'threads': min(multiprocessing.cpu_count()-1, max_cores),'barrier_display':1,'randomseed':201607292,'barrier_algorithm': 0,'barrier_startalg': 1}
        else:
            print("Solver parameters not found, exiting...")
            sys.exit()
    else:
        if solver_name == 'cplex':
            solver_options = {'simplex_tolerances_optimality': 1e-4, 'simplex_tolerances_feasibility': 1e-4,'mip_tolerances_mipgap': 5e-3, 'mip_tolerances_absmipgap': 1e-5, 'barrier_convergetol':1e-4,'workdir': os.path.join(os.getcwd(),'network',parameters["simulation_name"]), 'mip_strategy_subalgorithm': 4,'lpmethod': 4,'barrier_crossover': parameters["barrier_crossover"],'threads': min(multiprocessing.cpu_count()-1, max_cores),'mip_display':2,'randomseed':201607292,'preprocessing_aggregator':0,'emphasis_mip':1}
        else:
            print("Solver parameters not found, exiting...")
            sys.exit()

    # If True, keep solver .lp and .sol files for debugging if necessary
    if OGEM_options["reference"] == "expansion":
        keep_files = False
    else:
        keep_files = False

    # Remove all non-extendable null capacity generators, storage units, branches and buses before solve.
    reduced_network = network.copy()

    for line in reduced_network.lines.loc[(~reduced_network.lines["s_nom_extendable"]) & (reduced_network.lines["s_nom"] == 0)].index:
        reduced_network.remove("Line",line)
    for link in reduced_network.links.loc[(~reduced_network.links["p_nom_extendable"]) & (reduced_network.links["p_nom"] == 0)].index:
        reduced_network.remove("Link",link)
    for gen in reduced_network.generators.loc[(~reduced_network.generators["p_nom_extendable"]) & (reduced_network.generators["p_nom"] == 0)].index:
        reduced_network.remove("Generator", gen)
    for su in reduced_network.storage_units.loc[(~reduced_network.storage_units["p_nom_extendable"]) & (reduced_network.storage_units["p_nom"] == 0)].index:
        reduced_network.remove("StorageUnit",su)

    buses = np.unique(np.concatenate([reduced_network.buses[(reduced_network.buses.terminal_type == 'owf')|(reduced_network.buses.carrier == 'DC')].index,reduced_network.generators.bus.append(reduced_network.storage_units.bus)])) # Remove North Africa  and Eastern Europe buses if there is no capacity
    for bus in reduced_network.buses.index:
        if bus not in buses:
            for line in reduced_network.lines.loc[reduced_network.lines.index.str.contains(bus)].index:
                reduced_network.remove("Line", line)
            for link in reduced_network.links.loc[reduced_network.links.index.str.contains(bus)].index:
                reduced_network.remove("Link", link)
            reduced_network.remove("Bus",bus)

    Model_Check(reduced_network)

    reduced_network.lopf(snapshots=reduced_network.snapshots, solver_name=solver_name, keep_files = keep_files,OGEM_options=OGEM_options,base_welfare=base_welfare,parameters=parameters,solver_options=solver_options)

    Results_Check(reduced_network)

    return reduced_network

def MILP_Network(period, network):
    """ Expansion portfolio creation following assigned rule.
    Creates portfolio of candidates and then calls function to determine the transmission capacity of each candidate.
    """

    milp_network = network.copy() # Keep network intact
    milp_network.lines["base_branch"] = milp_network.lines.index # Since we duplicate branches below this keeps track of the original one
    milp_network.links["base_branch"] = milp_network.links.index

    # Extendable branches should belong to North Sea systems and be an interconnector or a farm-to-shore or farm-to-farm connector
    extendable_branches = parameters["branch_parameters"]["branch_class"].isin(parameters["extendable_branch_classes"])\
                          & parameters["branch_parameters"][["system0","system1"]].isin(parameters["cooperative_systems"]).all(axis = 1)

    for b,branch in parameters["branch_parameters"][extendable_branches].iterrows():
        branch_data = {"p_nom_extendable":True,"s_nom_extendable":True,"p_min_pu" : -1.0} # Update branch parameters since during import PyPSA only imports the standard parameters for each component
        branch_data.update(branch.to_dict())
        if not ((parameters["cooperation_limit"] is 0) and (branch["cooperative"] == True)): # A cooperation limit == 0 prohibits any cooperative branch, so skip this step for cooperative branches
            for n,capacity in enumerate(parameters["IC_cap"]): # Create one branch for every IC_cap element
                name = b + "-" + str(capacity)
                import_parameters = ["branch_type", "branch_class", "cooperative", "cooperative", "system0", "system1"]

                if branch["branch_type"] in ["AC","DC"]:
                        milp_network.add("Line",name,**branch_data)
                        milp_network.lines.loc[name,import_parameters + "marginal_cost"] = branch[import_parameters + "marginal_cost", ] # These parameters are not standard to lines and need to be imported manually
                        milp_network.lines.loc[name, []] = branch[[]] # System data is also not standard

                        # Set the minimum for each IC_cap element
                        if n == 0:
                            milp_network.lines.loc[name,"s_nom_min"] = parameters['min_cap']
                        else:
                            milp_network.lines.loc[name,"s_nom_min"] = parameters["IC_cap"][n-1]
                        milp_network.lines.loc[name,"s_nom_max"] = capacity

                        # Set electrical parameters
                        milp_network.lines.loc[name, "base_branch"] = b
                        milp_network.lines.loc[name, "r"] = branch["r"] * 1 / (capacity-milp_network.lines.loc[name,"s_nom_min"]) * 2 # Original value source: 0.0087 ohms/km for a single conductor (EuropaCable, s.d.), 0.015 ohms/km for a ~1000MW HVDC line (Pinto, 2014)
                        milp_network.lines.loc[name, "x"] = branch["x"] * 1 / (capacity-milp_network.lines.loc[name,"s_nom_min"]) * 2 # Original value source: 0.181 H/km per conductor (EuropaCable, s.d.)

                elif branch["branch_type"] == "ptp":
                        milp_network.add("Link",name,**branch_data)
                        milp_network.links.loc[name,import_parameters] = branch[import_parameters] # These parameters are not standard to lines and need to be imported manually

                        milp_network.links.loc[name, "p_nom_max"] = capacity
                        milp_network.links.loc[name, "base_branch"] = b

                        # Set the minimum for each IC_cap element
                        if n == 0:
                            milp_network.links.loc[name, "p_nom_min"] = parameters['min_cap']
                        else:
                            milp_network.links.loc[name, "p_nom_min"] = parameters["IC_cap"][n - 1]
    # Add converters to the investment portfolio
    for b,branch in parameters["branch_parameters"][parameters["branch_parameters"]["branch_type"].isin(["converter"])].iterrows():
         branch_data = {"p_nom_extendable":True ,"p_nom_min":0,"p_min_pu":-1.0}
         branch_data.update(branch.to_dict())
         group = branch["branch_type"] + str(b)
         milp_network.add("Link",group,**branch_data)
         milp_network.links.loc[group,"branch_type"] = branch["branch_type"]
         milp_network.links.loc[group,"base_branch"] = b
         milp_network.links.loc[group, ["system0","system1"]] = branch[["system0","system1"]]

    # Extendable wind farms are those with positive capital cost. This excludes farms which already existed in the first period.
    extendable_farms_i = milp_network.generators.loc[(milp_network.generators["carrier"] == "Wind - North Sea")&(milp_network.generators["capital_cost"] > 0)].index
    milp_network.generators.loc[extendable_farms_i,"p_nom_extendable"] = True # This was set to False for the base optimization.
    return milp_network

def Output_Recovery(output, network, period, milp_network = False):
    """ Calculate operational data from solution and aggregate from snapshots to representative hour:
        Total operational cost
        Nodal prices and demand
        Generator dispatch and cost
        Lines flow and price difference
    """
    print("Output recovery at {:%H:%M}...".format(datetime.datetime.now()))

    index = pd.MultiIndex(levels=[[]],
                          labels=[[]],
                          names=[u'snapshot'])
    CBA_data = {}
    data = {}
    # Construct dataframes with adequate column names, while indices are snapshots.
    for key in ["bus_shadow","bus_demand","con_payments"]:
        data[key] = pd.DataFrame(columns=network.buses.index, index=index)

    for key in ["generator_gen","gen_cost","gen_surplus","gen_shadow"]:
        data[key] = pd.DataFrame(columns=network.generators.index.append(network.storage_units.index), index=index)

    for key in ["branch_flow"]:
        data[key] = pd.DataFrame(columns=output.lines.index.append(output.links.index), index=index)

    # Map time-varying data from components to operational data, especially for data combining two components (generators and storage units).
    indicators = {"bus_demand":{"p":["loads_t"]},"bus_shadow":{"marginal_price":["buses_t"]},"generator_gen":{"p":["generators_t","storage_units_t"]},"branch_flow": {"p0": ["lines_t", "links_t"]}}

    for indicator,att1 in indicators.items():
        for variable,components in att1.items():
            dataframe = pd.concat([getattr(output,component)[variable] for component in components],axis = 1)
            data[indicator] = data[indicator].append(dataframe)

    # Since we combined generators and storage units data, all indices and costs need to be coherent.
    gen_sto_buses = output.generators["bus"].append(output.storage_units["bus"])
    gen_marginal_cost = pd.concat([output.generators.marginal_cost,output.storage_units.marginal_cost], axis=0)

    data["bus_shadow"] = data["bus_shadow"].divide(output.snapshot_weightings, axis = 0, level = "snapshot") # Since the marginal cost at the LP/MILP objective function is weighed by the snapshot weighing we need to rescale the marginal prices

    data["gen_shadow"] = data["bus_shadow"].loc[:,gen_sto_buses]
    data["gen_shadow"].columns = output.generators.index.append(output.storage_units.index) # Rename columns from bus to generator names
    data["gen_cost"] = data["generator_gen"].clip(lower = 0) * gen_marginal_cost # Clipping because of possible storage units with positive marginal cost
    data["gen_surplus"] = data["generator_gen"].clip(lower = 0) * (data["gen_shadow"].sub(gen_marginal_cost)) + data["generator_gen"].clip(upper = 0) * data["gen_shadow"] # Generators and storage earn their dispatch value by their marginal profit, and pay their storage at the marginal price.
    data["con_payments"] = (data["bus_demand"] * data["bus_shadow"]).fillna(0) # Consumers pay the marginal price for their demand.

    buses0 = output.lines["bus0"].append(output.links["bus0"])
    buses1 = output.lines["bus1"].append(output.links["bus1"])
    data["bus0_shadow"] = data["bus_shadow"][buses0]
    data["bus1_shadow"] = data["bus_shadow"][buses1]
    data["bus0_shadow"].columns = output.lines.index.append(output.links.index)
    data["bus1_shadow"].columns = output.lines.index.append(output.links.index)
    data["branch_shadow_delta"] = data["bus1_shadow"] - data["bus0_shadow"]
    data["branch_congestion_rent"] = data["branch_shadow_delta"] * data["branch_flow"] # Flows are valued at the terminal price differences, and congestion rents can be negative (no absolute value).

    # Average operation data of all snapshots to obtain a representative hour. Since we rescaled marginal prices these can also be averaged.
    probability = output.snapshot_weightings/output.snapshot_weightings.sum()
    for k,v in data.items():
        data[k] = data[k].mul(probability,axis = 0).sum()

    if milp_network:
        branch_map = output.lines["base_branch"].append(output.links["base_branch"]).to_dict() # Map all branches (base and expandable) to their base branch
        for variable in ["branch_flow","branch_congestion_rent"]:
            data[variable] = data[variable].groupby(branch_map).sum() # Aggregates branch variables to base branches
        data["branch_shadow_delta"] = data["branch_shadow_delta"].groupby(branch_map).mean() # Prices for branches with a common base and thus the same buses should be the same.

    offshore_farms = output.generators[output.generators.carrier == 'Wind - North Sea'].index
    onshore_renewables = output.generators_t.p_max_pu.columns.drop(offshore_farms) # Separate producer surplus among offshore wind farm and onshore renewable and conventional producers

    CBA_data["generator gen"] = data["generator_gen"]
    CBA_data["generator cost"] = data["gen_cost"]
    CBA_data["generator surplus"] = data["gen_surplus"]
    CBA_data["nodal prices"] = data["bus_shadow"]
    CBA_data["nodal con payments"] = data["con_payments"]
    CBA_data["nodal cost"] = data["gen_cost"].groupby(gen_sto_buses).sum().fillna(0)
    CBA_data["nodal generation"] = data["generator_gen"].groupby(gen_sto_buses).sum().fillna(0)
    CBA_data["nodal producer surplus"] = data["gen_surplus"].groupby(gen_sto_buses).sum().fillna(0)
    CBA_data["nodal offshore producer surplus"] = CBA_data["generator surplus"].fillna(0).loc[offshore_farms].groupby(gen_sto_buses).sum().fillna(0)
    CBA_data["nodal renewable producer surplus"] = CBA_data["generator surplus"].fillna(0).loc[onshore_renewables].groupby(gen_sto_buses).sum().fillna(0)
    CBA_data["nodal conventional producer surplus"] = CBA_data["generator surplus"].fillna(0).drop(onshore_renewables.append(offshore_farms)).groupby(gen_sto_buses).sum().fillna(0)
    CBA_data["branch congestion rent"] = data['branch_congestion_rent'].fillna(0)

    if milp_network:
        if hasattr(output.model, 'pareto_welfare'):
            CBA_data["participating countries"] = pd.DataFrame.from_dict(
                {'participation': {c: output.model.country_participation[c].value for c in output.model.country_participation},'welfare_change': {c: output.model.pareto_welfare[c].lslack() for c in output.model.pareto_welfare}}) # The slack indicates how close a country was not to participate with the pareto welfare constraint.
        else:
            CBA_data["participating countries"] = pd.DataFrame.from_dict(
                {'participation': {c: output.model.country_participation[c].value for c in output.model.country_participation}})

        del output.model
        # Save optimal lines, links and generators.
        optimal_lines = output.lines[(output.lines["s_nom_extendable"] == True) & ((output.lines["s_nom_opt"] - output.lines["s_nom"])>1E-5)]
        optimal_links = output.links[(output.links["p_nom_extendable"] == True) & ((output.links["p_nom_opt"] - output.links["p_nom"])>1E-5)]
        optimal_generators = output.generators[(output.generators["p_nom_extendable"] == True) & (output.generators["p_nom_opt"] > output.generators["p_nom"])]

        CBA_data["optimal expansions"] = pd.concat(
            [optimal_lines, optimal_links.rename(columns={'p_nom': 's_nom', 'p_nom_max' : 's_nom_max', 'p_nom_extendable': 's_nom_extendable', 'p_nom_min' : 's_nom_min','p_nom_opt':'s_nom_opt'})]) # Rename p_ parameters of links before concatenating.

        if len(CBA_data["optimal expansions"]) > 0:
            print(CBA_data["optimal expansions"])

        CBA_data["optimal generators"] = optimal_generators
        if len(CBA_data["optimal generators"]) > 0:
            print(CBA_data["optimal generators"])

    # Clusters are used in the onshore distribution function to assign national offshore nodes to offshore components.
    cluster_network = output.copy(with_time=False)

    for line in cluster_network.lines.loc[(cluster_network.lines["s_nom_opt"] == 0) & (cluster_network.lines["s_nom_extendable"] == False)].index:
        cluster_network.remove("Line", line)
    for link in cluster_network.links.loc[(cluster_network.links["p_nom_opt"] == 0) & (cluster_network.links["p_nom_extendable"] == False)].index:
        cluster_network.remove("Link", link)

    cluster_network.determine_full_network_topology() # Custom PyPSA function used to include both lines and links in the topology determination.

    CBA_data["clusters"] = [sub_net.buses() for sub_net in cluster_network.sub_networks["obj"]]

    return CBA_data

def Onshore_Distribution(CBA_data, network):
    """ Redistribute welfare components according to selected algorithm, reallocating any offshore bus welfare to onshore """
    base_buses_i = network.buses[network.buses["base_bus"] == network.buses.index].index

    CBA_data["onshore distribution"] = pd.DataFrame(0, index=network.buses.index, columns=base_buses_i)

    if parameters["redistribution_mechanism"] == "no redistribution":
        # No redistribution is made - standard mechanism.

        for buses in CBA_data["clusters"]: # There should be one large cluster with most system buses.
            cluster_countries = buses.loc[buses.index.isin(base_buses_i) & (buses.terminal_type == "on"),'country']
            if len(cluster_countries): # Fill distribution matrix only if there are onshore base buses.
                for b, bus in buses[buses["terminal_type"].isin(["owf"])].iterrows(): # Offshore welfare components are shared among national base buses.
                    national_buses_i = cluster_countries.loc[cluster_countries==bus.country].index
                    CBA_data["onshore distribution"].loc[b, national_buses_i] = 1 / len(national_buses_i)
                for b, bus in buses[~buses.index.isin(base_buses_i) & (~buses["terminal_type"].isin(["owf"]))].iterrows():
                    CBA_data["onshore distribution"].loc[b, bus["base_bus"]] = 1 # Onshore converter buses welfare components go to respective base bus

        base_onshore_i = network.buses.loc[network.buses["terminal_type"].isin(["on"]) & (network.buses["base_bus"] == network.buses.index)].index

        for bus in base_onshore_i:
            if CBA_data["onshore distribution"].loc[bus, bus] == 0:
                CBA_data["onshore distribution"].loc[bus, bus] = 1  # An onshore bus must receive its own cost and benefits.

    else:
        print("Redistribution mechanism not recognized")
        sys.exit()

    if (CBA_data["onshore distribution"].sum(axis=1)>1.).any(): # The welfare components distribution of any bus must not be greater than 100%.
        print("Incorrect onshore distribution matrix sum")
        sys.exit()

    # Share welfare components using the onshore distribution matrix.
    CBA_data["nodal components"] = pd.DataFrame(index = base_buses_i)
    welfare_components = ["producer surplus","offshore producer surplus","renewable producer surplus","conventional producer surplus","benefit","congestion rent","cost","con payments"]
    for component in welfare_components:
        CBA_data["nodal components"].loc[:,component] =  CBA_data["nodal "+component].reindex(network.buses.index,fill_value=0).dot(CBA_data["onshore distribution"])

    if "nodal inv cost" in CBA_data.keys():
        investment_components = ["inv cost","gen inv cost","trans inv cost"]
        for component in investment_components:
            CBA_data["nodal components"].loc[:, component] = CBA_data["nodal " + component].reindex(network.buses.index, fill_value=0).dot(CBA_data["onshore distribution"])

def Congestion_Distribution(CBA_data, network, network_branches):
    """ Branch congestion rent distribution.
        This is shared equally (50/50) between the branch buses.
        Afterwards the onshore distribution reallocates any offshore congestion rent to onshore buses.
     """

    # The congestion distribution matrix maps branch congestion rent to buses.
    congestion_distribution = pd.DataFrame(0, index=CBA_data["branch congestion rent"].index, columns=network.buses.index)

    for branch in congestion_distribution.index:
        congestion_distribution.loc[branch, network.buses.loc[network_branches.loc[branch,"bus0"],"base_bus"]] += 0.5
        congestion_distribution.loc[branch, network.buses.loc[network_branches.loc[branch,"bus1"],"base_bus"]] += 0.5

    # Aggregate congestion rent per bus.
    CBA_data["nodal congestion rent"] = CBA_data["branch congestion rent"].dot(congestion_distribution)

    # Now all welfare components are allocated to buses, calculate total benefit. Since demand is inelastic this benefit makes sense only when compared with another system state, with consumer benefits inversely valued as a proxy to consumer surplus.
    CBA_data["nodal benefit"] = (CBA_data["nodal producer surplus"] + CBA_data["nodal congestion rent"] - CBA_data["nodal con payments"]).fillna(0)

def Cost_Calculation(CBA_data, network, network_branches):
    """ Calculates investment cost of transmission and generation """

    def Expansion_Cost_Calculation(unconnected_branches):
        """ Calculates the investment cost for any given combination of branches"""
        cost_vector = pd.Series(0, index=network.buses.index)
        expansion = expansions[~ unconnected_branches]

        # Transmission costs are split equally between the branch buses.
        cost_vector = cost_vector.add(
            expansion.groupby(["bus0"])["cost"].sum().groupby(network.buses["base_bus"]).sum() / 2, fill_value=0)
        cost_vector = cost_vector.add(
            expansion.groupby(["bus1"])["cost"].sum().groupby(network.buses["base_bus"]).sum() / 2, fill_value=0)

        return cost_vector

    """ Calculates total costs used in the welfare analysis """
    onshore_buses = network.buses["terminal_type"].isin(["on"]).tolist() # All system onshore buses.

    expansions = CBA_data["optimal expansions"]
    # Calculate candidate expansion cost, nodal cost and length.
    expansions.loc[:,"cost"] = expansions["capital_cost"] * (1+ np.pv(parameters['discount_rate'],parameters['high_lifetime'],-parameters['low_OPEX_%'])) * (expansions["s_nom_opt"]-expansions["s_nom"]) # Total investment costs include CAPEX and OPEX as a % of CAPEX
    CBA_data["cost"] = expansions["cost"].sum()

    CBA_data["nodal trans inv cost"] = Expansion_Cost_Calculation(pd.Series(False, index=expansions.index))
    CBA_data["length"] = expansions["length"].sum()

    generators = CBA_data["optimal generators"]
    generators.loc[:,"cost"] = generators["capital_cost"] * (1 + np.pv(parameters['discount_rate'],parameters['low_lifetime'],-parameters['high_OPEX_%'])) * (generators["p_nom_opt"] - generators["p_nom"]) # Total investment costs include CAPEX and OPEX as a % of CAPEX
    CBA_data["cost"]  += generators["cost"].sum()

    CBA_data["nodal gen inv cost"] = generators.groupby(["bus"])["cost"].sum()
    CBA_data["nodal inv cost"] = CBA_data["nodal trans inv cost"].add(CBA_data["nodal gen inv cost"],fill_value = 0)

def Snapshot_Selection(network, milp_network ,method = 'k-medoids',series='prices',n_components = 90):
    """ Selects a number of representative snapshots among all year snapshots using the k-medoids/agglomerative algorithms using marginal prices as the information """
    from sklearn.decomposition import PCA

    print("Snapshot selection at {:%H:%M}...".format(datetime.datetime.now()))
    print("Clustering series is", series)
    if series == 'NS prices':
        countries = ['gb', 'be', 'nl', 'de', 'dk', 'lu', 'no', 'se', 'fr', 'ie']
        clustering_series = network.buses_t.marginal_price.loc[:,network.buses.country.isin(countries)]

    elif series == 'prices':
        clustering_series = network.buses_t.marginal_price
        pca = PCA(n_components=n_components)
        pca.fit(clustering_series)
        pca_series = pca.transform(clustering_series)

    elif series == 'net load':
        net_load = network.loads_t.p_set
        net_load = net_load.sub((network.generators_t.p_max_pu * network.generators.p_nom[network.generators_t.p_max_pu.columns]).groupby(network.generators["bus"], axis=1).sum(),
                                fill_value=0)
        net_load = net_load.sub(network.storage_units_t.p.groupby(network.storage_units["bus"], axis=1).sum(), fill_value=0)
        net_load = net_load.fillna(0)
        clustering_series = net_load # Snapshot selection with net load = loads - renewable - fixed storage
        pca = PCA(n_components=n_components)
        pca.fit(clustering_series)
        pca_series = pca.transform(clustering_series)

    else:
        print("Warning: Clustering series type not found")
        sys.exit()

    print('Clustering method is', method)
    if method == 'k-medoids':
        # Snapshot selection with marginal prices
        initial_index =np.arange(0, parameters["snapshots"], parameters["snapshots"] / parameters["clusters"]).astype(int) # Distribute starting snapshots equally among snapshots.
        clusters = kmedoids(clustering_series.as_matrix(), initial_index, tolerance=0.05)
        clusters.process()
        medoids = [int(np.where(np.all(clustering_series == medoid, axis=1))[0][0]) for medoid in clusters.get_medoids()] # Find medoids list among all snapshots.
        k_labels = [c for ind in range(len(clustering_series)) for c, cluster in enumerate(clusters.get_clusters()) if ind in cluster]
        weightings = [len(cluster) for cluster in clusters.get_clusters()]
        milp_network.set_snapshots(network.snapshots[medoids])
        milp_network.snapshot_weightings.loc[:] = weightings # Weight of snapshots (medoids) is number of members in each cluster

    elif method == 'pca_k-medoids':
        # Snapshot selection with marginal prices
        initial_index =np.arange(0, parameters["snapshots"], parameters["snapshots"] / parameters["clusters"]).astype(int) # Distribute starting snapshots equally among snapshots.
        clusters = kmedoids(pca_series, initial_index, tolerance=0.25)
        clusters.process()
        medoids = [int(np.where(np.all(pca_series == medoid, axis=1))[0][0]) for medoid in clusters.get_medoids()] # Find medoids list among all snapshots.
        k_labels = [c for ind in range(len(pca_series)) for c, cluster in enumerate(clusters.get_clusters()) if ind in cluster]
        weightings = [len(cluster) for cluster in clusters.get_clusters()]
        milp_network.set_snapshots(network.snapshots[medoids])
        milp_network.snapshot_weightings.loc[:] = weightings # Weight of snapshots (medoids) is number of members in each cluster

    elif method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import pairwise_distances

        clustering = AgglomerativeClustering(linkage='ward', n_clusters=parameters["clusters"])
        clustering.fit(clustering_series.as_matrix())
        AC_medoids = []
        for cl_num in pd.unique(clustering.labels_):
            cluster = clustering_series.loc[clustering.labels_ == cl_num, :]
            cluster_index = clustering_series.loc[clustering.labels_ == cl_num, :].index
            AC_medoids.append(cluster.index[np.argmin(pairwise_distances(cluster).sum(axis=0))])
        weightings = pd.Series(clustering.labels_).value_counts().reindex(pd.unique(clustering.labels_))
        milp_network.set_snapshots(network.snapshot_weightings[AC_medoids].index)
        milp_network.snapshot_weightings.loc[:] = weightings.as_matrix() # Weight of snapshots (medoids) is number of members in each cluster

    elif method == 'pca_agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import pairwise_distances

        clustering = AgglomerativeClustering(linkage='ward', n_clusters=parameters["clusters"])
        clustering.fit(pca_series)
        AC_medoids = []
        for cl_num in pd.unique(clustering.labels_):
            cluster = pca_series[clustering.labels_ == cl_num, :]
            cluster_index = clustering_series.loc[clustering.labels_ == cl_num, :].index
            AC_medoids.append(cluster_index[np.argmin(pairwise_distances(cluster).sum(axis=0))])
        weightings = pd.Series(clustering.labels_).value_counts().reindex(pd.unique(clustering.labels_))
        milp_network.set_snapshots(network.snapshot_weightings[AC_medoids].index)
        milp_network.snapshot_weightings.loc[:] = weightings.as_matrix()  # Weight of snapshots (medoids) is number of members in each cluster
        milp_network.set_snapshots(network.snapshot_weightings[AC_medoids].index)
        milp_network.snapshot_weightings.loc[:] = weightings.as_matrix()  # Weight of snapshots (medoids) is number of members in each cluster

    else:
        print("Warning: Clustering method not found")
        sys.exit()

    milp_network.snapshot_weightings.loc[:] = milp_network.snapshot_weightings / milp_network.snapshot_weightings.sum()

    print("Selection finished at {:%H:%M}...".format(datetime.datetime.now()))

def Time_Series_Scaling(network,milp_network,load_method = 'mixed'):

    print('RES scaling method: Peak')
    factors = pd.Series()
    peak_RES_factor = milp_network.generators_t.p_max_pu.max().groupby(milp_network.generators.carrier).max() / network.generators_t.p_max_pu.max().groupby(network.generators.carrier).max()
    milp_network.generators_t.p_max_pu = milp_network.generators_t.p_max_pu.apply(lambda x: x / peak_RES_factor[milp_network.generators.loc[x.name, 'carrier']], axis=0)
    factors = factors.append(peak_RES_factor)

    print('Load scaling method:',load_method)

    if load_method == 'mean':
        load_factor = milp_network.loads_t.p_set.mean().mean() / network.loads_t.p_set.mean().mean()
    elif load_method == 'peak':
        load_factor = milp_network.loads_t.p_set.max().max() / network.loads_t.p_set.max().max()
    elif load_method == 'mixed':
        mean_load_factor = milp_network.loads_t.p_set.mean().mean() / network.loads_t.p_set.mean().mean()
        peak_load_factor = milp_network.loads_t.p_set.max().max() / network.loads_t.p_set.max().max()
        load_factor = (mean_load_factor+peak_load_factor)/2
        factors = factors.append(pd.Series(mean_load_factor, index=['mean_load_factor']))
        factors = factors.append(pd.Series(peak_load_factor, index=['peak_load_factor']))
        factors = factors.append(pd.Series(load_factor, index=['load_factor']))
    else:
        print("Warning: Load scaling method not found, exiting.")
        sys.exit()

    milp_network.loads_t.p_set = milp_network.loads_t.p_set / load_factor

    factors.name = 'Scaling factors'
    print(factors)

def Expansions_Simulation(period, network_branches, network, input_panel):
    """ Runs the base, expansion and final cases for the period network.
        1) Runs the base case, a full year linear system operation case. Calculates system operational data and welfare components.
        2) Selects representative snapshots with Snapshot_Selection(). Calculates system operational data and welfare components of the base case for those snapshots.
        3) Includes extendable transmission branches and offshore wind farms (potential investments) in the network.
        4) Runs the expansion case, a mixed-integer system operation and investment problem with the representative snapshots. Calculates system operational data and welfare components.
        5) Updates the network with the optimal transmission and offshore wind farms investments.
        6) Runs the final case, a full year linear system operation case used to compared the expanded system to the base case. Calculates system operational data and welfare components.
    """
    print("Base {} simulation at {:%H:%M}...".format(period,datetime.datetime.now()))

    # Simulation of base case and data calculation.
    OGEM_options = {"reference":"base", "pareto": False, "cooperation_limit": False} # The base case does never has pareto or cooperation constraints.

    base_output = MILP_OPF(period,network,OGEM_options=OGEM_options)  # Base case optimal power flow.
    Write_Network(base_output, period, OGEM_options["reference"])

    # Base case operational data and welfare component calculation.
    CBA_base = Output_Recovery(base_output, network, period)
    Congestion_Distribution(CBA_base, network, network_branches)
    Onshore_Distribution(CBA_base,network)
    Write_Output(CBA_base, period,'base')

    # Simulation of expansion case
    print("Expansion {} simulation at {:%H:%M}...".format(period, datetime.datetime.now()))
    OGEM_options = {"reference": "expansion", "pareto": parameters["pareto"], "cooperation_limit": parameters["cooperation_limit"]}
    milp_network = MILP_Network(period, network) # Include extendable transmission and generation in the network.

    Snapshot_Selection(base_output, milp_network) # Select snapshots for the MILP and to calculate welfare with reduced snapshots

    if parameters['north_sea_expansion']:
        # Fix the dispatch and storage of storage and energy-constrained generators since the snapshot selection loses the temporal connection of snapshots.
        inflow_generators = base_output.generators[base_output.generators.inflow > 0].index
        pypsa.io.import_series_from_dataframe(milp_network, base_output.generators_t.p.loc[:, inflow_generators] * (base_output.generators_t.p.loc[:, inflow_generators].abs() > 1e-4), "Generator", "p_set") # Dispatches under .1 MW are disconsidered
        pypsa.io.import_series_from_dataframe(milp_network, base_output.storage_units_t.p * (base_output.storage_units_t.p.abs()>1e-4), "StorageUnit", "p_set") # Dispatches under .1 MW are disconsidered
        if 'expansion_scaling' in parameters.keys():
            if parameters['expansion_scaling']:
                Time_Series_Scaling(network, milp_network)

        milp_network.scenarios.loc[:] = milp_network.snapshots # For future use in stochastic optimization.

        # Calculate the operation data and welfare components of the base case using the selected snapshots, to obtain the base welfare.
        base_output.set_snapshots(milp_network.snapshots)
        base_output.buses_t.marginal_price = base_output.buses_t.marginal_price.mul(milp_network.snapshot_weightings/base_output.snapshot_weightings,axis=0) # Calculation of base welfare requires correct marginal prices, which need to be weighed by the probability of the representative snapshots.
        base_output.snapshot_weightings = milp_network.snapshot_weightings

        # The reduced base CBA represents the CBA with reduced snapshots.
        CBA_reduced_base = Output_Recovery(base_output, network, period)
        Congestion_Distribution(CBA_reduced_base, network, network_branches)
        Onshore_Distribution(CBA_reduced_base,network)
        base_welfare = CBA_reduced_base["nodal benefit"].groupby(base_output.buses.country).sum() # The base welfare is the reference used by countries to decide on cooperation when the pareto welfare constraint is active.
        del CBA_reduced_base # To reduce memory usage.

    del base_output # To reduce memory usage.
    del CBA_base

    if parameters['north_sea_expansion']: # Determines whether to run the expansion case. A no-expansion simulation is used to compare the welfare components of expansion simulations.
        output = MILP_OPF(period,milp_network,OGEM_options=OGEM_options,base_welfare=base_welfare)
        Write_Network(output, period, OGEM_options["reference"])
        del milp_network # To reduce memory usage.

        # Calculate the operation data and welfare components of the expansion case. The expansion case is the only to have cost calculations.
        CBA_exp = Output_Recovery(output,network, period, milp_network=True)
        Cost_Calculation(CBA_exp, network, network_branches)
        Congestion_Distribution(CBA_exp, network, network_branches)
        Onshore_Distribution(CBA_exp, network)

        del output # To reduce memory usage.

    else:
        # Create empty optimal investment records in no expansion case.
        CBA_exp = {}
        CBA_exp["optimal expansions"] = network.lines.copy().drop(network.lines.index)
        CBA_exp["optimal generators"] = network.generators.copy().drop(network.generators.index)

    if parameters['north_sea_expansion']:
        Write_Output(CBA_exp, period, 'expansion')

    # Update link status, transmission capacity and attribute (reactance or resistance)
    for b,branch in CBA_exp["optimal expansions"].iterrows():
        base_branch = branch["base_branch"]
        attribute = "x" if branch["branch_type"] == "AC" else "r"
        if network_branches.loc[base_branch,"s_nom"] == 0:
            network_branches.loc[base_branch, attribute] = branch[attribute]
        else:
            network_branches.loc[base_branch, attribute] = (network_branches.loc[base_branch, attribute] * branch[attribute]) / (network_branches.loc[base_branch, attribute] + branch[attribute])

        network_branches.loc[base_branch, "s_nom"] += branch["s_nom_opt"]

    network_branches.loc[:,"s_nom_min"] = network_branches.loc[:,"s_nom"]

    # Set minimum capacity of lines and generators to new capacity value

    if len(CBA_exp["optimal generators"]) > 0:
        network.generators.loc[CBA_exp["optimal generators"].index, "p_nom"] = CBA_exp["optimal generators"]["p_nom_opt"]
        network.generators.loc[CBA_exp["optimal generators"].index, "p_nom_min"] = CBA_exp["optimal generators"]["p_nom_opt"]

    if parameters['north_sea_expansion']:
        print("Full expansion {} simulation at {:%H:%M}...".format(period, datetime.datetime.now()))

        # Update network with selected investments. This sets the extendability of generation to False.
        Network_Update(period, network, network_branches, input_panel)

        # Full hours run for comparison with base case
        OGEM_options = {"reference":"base", "pareto": False, "cooperation_limit": False}
        final_output = MILP_OPF(period,network,OGEM_options=OGEM_options)  # Base case OPF
        Write_Network(final_output, period, 'final')

        # Calculate the operation data and welfare components of the final case.
        CBA_final = Output_Recovery(final_output, network, period)
        Congestion_Distribution(CBA_final, network, network_branches)
        Onshore_Distribution(CBA_final,network)

        # Add investment cost to final results and write.
        CBA_final["nodal components"] = pd.concat([CBA_final["nodal components"],CBA_exp["nodal components"].loc[:,["inv cost","gen inv cost","trans inv cost"]]],axis=1)

        Write_Output(CBA_final, period,'final')
        del final_output
        del CBA_final

    return CBA_exp

def Write_Network(output_network, period, folder):
    """ Writes solution for debugging """

    path = os.path.join(os.getcwd(), r"network",parameters['simulation_name'], r"p" + str(period), folder)

    if not os.path.isdir(path):
        os.mkdir(path)

    np.save(os.path.join(path,'parameters.npy'), parameters)

    pypsa.io.export_to_csv_folder(output_network, path)

def Write_Output(CBA_data, period, folder):
    """ Writes welfare components and optimal investments, if any, for specific case """

    path = os.path.join(os.getcwd(), r"network", parameters['simulation_name'], r"p" + str(period), folder, parameters["simulation_name"] +".xls")

    writer = pd.ExcelWriter(path)

    CBA_data["nodal components"].to_excel(writer,sheet_name = r"nodal components")

    if folder == 'expansion':
        CBA_data["optimal expansions"].to_excel(writer,sheet_name = r"opt_exp") #.drop('obj',axis=1)
        CBA_data["optimal generators"].to_excel(writer,sheet_name = r"opt_gen")
        CBA_data["participating countries"].to_excel(writer, sheet_name=r"country_participation")

    writer.save()

def Network_Setup(period, input_panel):
    """ Creation of network elements from input data """

    print("Network setup at {:%H:%M}...".format(datetime.datetime.now()))

    network = pypsa.Network()

    # PyPSA snapshot data: name, weightings.
    network.set_snapshots(input_panel["snapshots"].iloc[parameters["initial_snapshot"]:parameters["initial_snapshot"]+parameters["snapshots"]].index) # For debugging it is possible to restrict snapshots to a given range
    network.snapshot_weightings.loc[:] = parameters["snapshots_probability"]

    # PyPSA bus data: number, x, y, terminal_type, current_type.
    pypsa.io.import_components_from_dataframe(network,input_panel["buses"], "Bus")

    # network_branches is used to update grid data for each candidate expansion and then transfer this to the PyPSA dataframe in the candidate iteration.

    def branch_class(branch):
        # Branch classes are interconnector, connector, farm-to-farm and onshore. It is used to select the extendable branches (investment portfolio).
        systems = [input_panel["buses"].loc[branch["bus0"],"system"],input_panel["buses"].loc[branch["bus1"],"system"]]
        systems.sort()
        systems = tuple(systems)
        return input_panel["branch_class"].loc[systems, "class"]

    def cooperative_branch(branch, bus0, bus1):
        # Cooperative branches are cross-border branches or connect wind farms to wind farms
        cooperative = branch["branch_class"] in parameters["cooperative_branch_classes"]
        cross_border_or_ns = (bus0["country"] != bus1["country"]) | ((bus0["system"] == "ns") & (bus1["system"] == "ns"))
        return cooperative & cross_border_or_ns
    
    # The branch_parameters dataframe lists the data for branch investment candidates
    input_panel["branch_parameters"]["system0"] = input_panel["buses"].loc[input_panel["branch_parameters"]["bus0"], "system"].tolist()
    input_panel["branch_parameters"]["system1"] = input_panel["buses"].loc[input_panel["branch_parameters"]["bus1"], "system"].tolist()

    # Transmission marginal costs prioritize nodal generation. Should be small without causing numerical difficulties to the problem.
    input_panel["branch_parameters"]['marginal_cost'] = parameters['trans_marginal_cost']
    input_panel["transmission"]['marginal_cost'] = parameters['trans_marginal_cost']

    for b,branch in input_panel["branch_parameters"].iterrows():
        input_panel["branch_parameters"].loc[b,"branch_class"] = branch_class(branch)
    for b, branch in input_panel["branch_parameters"].iterrows():
        input_panel["branch_parameters"].loc[b,"cooperative"] = cooperative_branch(branch, network.buses.loc[branch["bus0"]], network.buses.loc[branch["bus1"]])
    
    # The network_branches dataframe constains the actual branch portfolio, invested in or not
    network_branches = input_panel["branch_parameters"].copy()
    network_branches["s_nom"] = 0.0
    network_branches["s_nom_min"] = 0.0

    network_branches[["bus0", "bus1"]] = network_branches[["bus0", "bus1"]].astype(str) # PyPSA buses should be strings always
    extendable_branches = network_branches["branch_class"].isin(parameters["extendable_branch_classes"]) & network_branches[["system0", "system1"]].isin(
        parameters["cooperative_systems"]).all(axis=1)
    network_branches = network_branches[extendable_branches]

    # Create DC buses for buses of the Northern Seas system
    network.buses["base_bus"] = network.buses.index
    for b,bus in network.buses.loc[network_branches[["bus0", "bus1"]].stack().unique()].iterrows():
        for carrier in ["DC"]:
            name = b + "_" + carrier
            network.add("Bus",name,**bus.to_dict())
            # Adding bus_data manually for non-standard or non-existing parameters
            bus_data = {"base_bus": b, "terminal_type": bus.terminal_type, "x": bus.x, "y": bus.y,
                 "carrier": carrier, "country": bus.country,"system":bus.system,"v_nom":3. * 10 ** np.floor(np.log10(bus.v_nom))} # x and y are bus coordinates, terminal types are onshore or offshore
            for key,value in bus_data.items():
                network.buses.loc[name,key] = value

    # PyPSA line data: name, bus0, bus1, x, r, s_nom, s_nom_extendable, length.
    # PyPSA link data: name, bus0, bus1, p_nom, p_nom_extendable.

    # Append converters between AC and DC buses to network_branches
    for b in network_branches[["bus0", "bus1"]].stack().unique():
        for carrier in ["DC"]:
            bus1 = b + "_" + carrier
            index = ["conv" + bus1] # Converter name
            conv_data = {"bus0" : b, "bus1" : bus1, "s_nom": 0.0, "s_nom_min": 0.0,"branch_type" : "converter", "s_nom_extendable" : False,"system0":network.buses.loc[b,"system"],"system1":network.buses.loc[bus1,"system"],"p_min_pu" : -1.0,"marginal_cost":0.0,"capital_cost" : parameters["cc_"+carrier]+parameters["cb_"+carrier]}
            network_branches = network_branches.append(pd.DataFrame(conv_data,index=index))

    # Rename buses of DC branches to actual DC buses
    dc_branches = network_branches["branch_type"].isin(["DC"])
    network_branches.loc[dc_branches,"bus0"] = network_branches.loc[dc_branches,"bus0"] + "_" + network_branches.loc[dc_branches,"branch_type"]
    network_branches.loc[dc_branches,"bus1"] = network_branches.loc[dc_branches,"bus1"] + "_" + network_branches.loc[dc_branches,"branch_type"]

    # To calculate buses distances. Not used.
    def haversine(coord):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        Thanks to Michael Dunn in http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        """
        # convert decimal degrees to radians
        from math import radians, cos, sin, asin, sqrt
        lon1, lon2, lat1, lat2 = map(radians, coord.tolist())

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    if False:
        coordinates = pd.concat([network_branches.loc[:, ['bus0', 'bus1']].replace(network.buses.x), network_branches.loc[:, ['bus0', 'bus1']].replace(network.buses.y)], axis=1)
        coordinates.columns = ['x0', 'x1', 'y0', 'y1']
        network_branches.loc[:, "length"] = coordinates.apply(haversine,axis=1)

    # Update branch_parameters with new network_branches data before adding the exogenous branches of input_panel["transmission"]
    parameters["branch_parameters"] = network_branches.loc[:,["bus0", "bus1", "branch_type", "branch_class", "system0", "system1", "capital_cost","marginal_cost", "length", "r", "x","cooperative"]]

    # The input_panel["transmission"] contains the exogenous branches of the scenario
    for b, branch in input_panel["transmission"].iterrows():
        input_panel["transmission"].loc[b,"branch_class"] = branch_class(branch)
    network_branches = pd.concat([network_branches, input_panel["transmission"][~input_panel["transmission"].index.isin(network_branches.index)]])
    network_branches.update(input_panel["transmission"])
    network_branches.loc[:,"s_nom"] = network_branches.loc[:,"s_nom"+str(period)].fillna(0)
    network_branches["s_nom_extendable"] = False # All branches are non-extendable for the starting base case

    # Update system and cooperation parameters also for new exogenous branches
    network_branches["system0"] = network.buses.loc[network_branches["bus0"], "system"].tolist()
    network_branches["system1"] = network.buses.loc[network_branches["bus1"], "system"].tolist()
    for b, branch in network_branches.iterrows():
        network_branches.loc[b,"cooperative"] = cooperative_branch(branch, network.buses.loc[branch["bus0"]], network.buses.loc[branch["bus1"]])

    # Finally update pypsa with endogenous and exogenous branches
    pypsa.io.import_components_from_dataframe(network, network_branches[network_branches["branch_type"].isin(["AC","DC"])], "Line")
    pypsa.io.import_components_from_dataframe(network, network_branches[network_branches["branch_type"].isin(["ptp","converter"])].rename(columns={'s_nom': 'p_nom', 's_nom_max' : 'p_nom_max', 's_nom_extendable': 'p_nom_extendable', 's_nom_min' : 'p_nom_min'}),"Link")

    # PyPSA generator data: name, bus, p_nom, p_nom_extendable, carrier, marginal_cost, capital_cost.
    input_panel["generation"]["p_nom_extendable"] = False
    input_panel["generation"]["p_nom"] = input_panel["generation"]["p_nom"+str(period)] # Update p_nom for the period
    input_panel["generation"]["inflow"] = input_panel["generation"]["inflow_nom"+str(period)] * parameters["snapshots"] / 8760 # Update inflow, scaling for the number of snapshots which may vary during debugging.
    pypsa.io.import_components_from_dataframe(network, input_panel["generation"], "Generator")

    # PyPSA generator data: name, bus, p_nom, p_nom_extendable, carrier, marginal_cost, capital_cost.
    input_panel["extendable_generation"]["p_nom_extendable"] = False # All generators should be non-extendable for starting base case
    input_panel["extendable_generation"]["p_nom"] = 0
    input_panel["extendable_generation"]["p_nom_max"] = input_panel["extendable_generation"]["p_nom_max"+str(period)] # Update p_nom for the period
    pypsa.io.import_components_from_dataframe(network, input_panel["extendable_generation"], "Generator")


    # PyPSA RES generator time series: snapshots, generators
    pypsa.io.import_series_from_dataframe(network,input_panel["generation_series"],"Generator","p_max_pu")

    # PyPSA storage data: name, bus, p_nom, p_nom_extendable, carrier, state_of_charge_initial, efficiency_store, efficiency_dispatch, marginal_cost
    input_panel["storage"]["p_nom_extendable"] = False
    input_panel["storage"]["p_nom"] =input_panel["storage"]["p_nom"+str(period)]

    pypsa.io.import_components_from_dataframe(network, input_panel["storage"], "StorageUnit")
    #network.storage_units.loc[network.storage_units.efficiency_store>0,'marginal_cost'] = parameters['storage_marginal_cost'] # Add an epsilon cost to storage to avoid simultaneous storage and dispatch of storage units

    # PyPSA load data: bus, name
    columns = ["bus"]
    index = input_panel['demand'].index
    data = list(index)

    dataframe = pd.DataFrame(data, index=index, columns=columns)
    del dataframe.index.name

    pypsa.io.import_components_from_dataframe(network, dataframe, "Load")

    # To reduce memory usage.
    for item in ["buses","generation","storage","generation_series","branch_parameters","extendable_generation","branch_class"]:
        del input_panel[item]

    return network_branches, network

def Network_Update(period, network, network_branches, input_panel):

    """ Updates network data with period changes """
    # Update grid from network_branches to network with capacity, status, reactance and resistance.

    network_branches.update(input_panel["transmission"].loc[:, ['s_nom' + str(period)]].rename(columns={'s_nom' + str(period): 's_nom'})) # Update capacity for current period
    network_branches.loc[:,'s_nom_max'] = network_branches.loc[:,'s_nom']
    network_branches.loc[:,'s_nom_min'] = network_branches.loc[:,'s_nom']

    network.lines.update(network_branches)
    network.links.update(
        network_branches.rename(columns={'s_nom': 'p_nom', 's_nom_max': 'p_nom_max', 's_nom_extendable': 'p_nom_extendable', 's_nom_min': 'p_nom_min', 's_nom_opt': 'p_nom_opt'}))

    # Update time series for generation, storage
    network.generators.loc[:,"p_nom_extendable"] = False

    network.generators.loc[~(network.generators["carrier"] == "Wind - North Sea"),"p_nom"] = network.generators.loc[~(network.generators["carrier"] == "Wind - North Sea"),"p_nom"+str(period)].fillna(np.inf) # Update generation capacity, with infinite capacity for slack generators
    network.generators.loc[:,"inflow"] = network.generators.loc[:,"inflow_nom"+str(period)] * parameters["snapshots"] / 8760 # Update inflow, scaling for the number of snapshots
    network.generators.loc[:,"marginal_cost"] = network.generators.loc[:, "marginal_cost" + str(period)]

    # Hydro generators have minimum and maximum dispatch values. These may need to be altered for problem feasibility (a high minimum dispatch at all snapshots may be infeasible).
    hydro_generators = network.generators.carrier.isin(['Hydro','Hydro with reservoir','RoR'])
    minimum =network.generators.loc[hydro_generators, "inflow"] / parameters["snapshots"] / network.generators.loc[hydro_generators, "p_nom"] * 0.99
    network.generators.loc[hydro_generators,"p_min_pu"]  = np.minimum(minimum.fillna(0),parameters["storage_p_min_pu"])
    network.generators.loc[hydro_generators,"p_max_pu"]  = parameters["storage_p_max_pu"]
    network.generators.loc[:, "p_nom_max"] = network.generators.loc[:, "p_nom_max"+str(period)]

    network.storage_units.loc[:,"p_nom"] = network.storage_units.loc[:,"p_nom"+str(period)]
    network.storage_units.loc[:,"inflow"] = network.storage_units.loc[:,"inflow_nom"+str(period)]

def Time_Series_Update(period, network, input_panel):

    """ Updates times series for the current period """
    # PyPSA load time series: snapshots, loads
    demand_series = input_panel["demand_series"] * input_panel["demand"]["p_set"+str(period)].transpose()
    pypsa.io.import_series_from_dataframe(network,demand_series,"Load","p_set")

    # Inflow storage technologies need their inflow time series.
    # PyPSA hydro generator time series: snapshots, generators
    inflow_series = input_panel["inflow_series"] * network.storage_units.loc[input_panel["inflow_series"].columns, "p_nom" + str(period)] # Update inflow series for some storage, since it is dependent on capacity (e.g. concentrated solar power).
    pypsa.io.import_series_from_dataframe(network, inflow_series,"StorageUnit","inflow")

def Load_Parameters(run_name):
    """ Load analysis parameters to initialize run calls.
        The parameters are sent back to the simulation main function to start the periods iteration.
    """

    global parameters

    parameters_file = os.path.join("input", run_name, "parameters.csv")
    with open(parameters_file, mode='r') as file:
        reader = csv.reader(file)
        parameters = {rows[0]: float(rows[1]) if rows[2] == "float" else int(rows[1]) if rows[2] == "int" else rows[1] == "TRUE" if rows[2] == "bool" else rows[1] for rows in reader}

    for parameter in ["cooperative_systems","cooperative_branch_classes","extendable_branch_classes"]:
        parameters[parameter] = parameters[parameter].split() # These strings need to be split to list of strings

    # For the full hours run all snapshots are equiprobable
    parameters["snapshots_probability"] = np.array([1 / parameters["snapshots"] for sn in range(parameters["snapshots"])])

    # Convert the IC_cap string to a list of strings
    parameters["IC_cap"] = [float(cap) for cap in parameters["IC_cap"].split()]

    if parameters["cooperation_limit"] != "FALSE":
        parameters["cooperation_limit"] = int(parameters["cooperation_limit"])
    else:
        parameters["cooperation_limit"] = False

    return parameters

def Save_Parameters(parameters):
    """ Load analysis parameters to initialize run calls.
        The parameters are sent back to the simulation main function to start the periods iteration.
    """
    parameters_file = os.path.join("network", parameters['simulation_name'], "parameters.csv")
    with open(parameters_file,'wb') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(file)

def Load_Data(run_name):
    """ Load input data for a given run.
        The data is returned to the simulations main function to start the periods iteration.
    """

    input_panel = {}

    # Treat all input data with single-column indexes.
    for sheet in ["transmission","buses","generation","storage","inflow_series","generation_series","demand_series","branch_parameters","snapshots","extendable_generation","demand"]:
        input_path = os.path.join("input", run_name, sheet +".csv")
        input_panel[sheet] = pd.read_csv(input_path, index_col=0)
        input_panel[sheet] = input_panel[sheet].loc[input_panel[sheet].index.dropna()]
    input_panel["extendable_generation"]["bus"] = input_panel["extendable_generation"]["bus"].str.lower()

    # Treat all input data with double-column indexes.
    for sheet in ["branch_class"]:
        input_path = os.path.join("input", run_name, sheet +".csv")
        input_panel[sheet] = pd.read_csv(input_path, index_col=[0,1])
        input_panel[sheet] = input_panel[sheet].loc[input_panel[sheet].index.dropna()]
        input_panel[sheet].sort_index()

    # Limit time data to snapshots of interest to reduce memory usage.
    for sheet in ["generation_series","inflow_series","demand_series"]:
        input_panel[sheet] = input_panel[sheet].iloc[parameters["initial_snapshot"]:parameters["initial_snapshot"]+parameters["snapshots"],:]

    # All PyPSA indexes should be string.
    for sheet in ["transmission","buses","generation","storage","inflow_series","demand_series","branch_parameters","extendable_generation"]:
        if input_panel[sheet].index.dtype == np.int64:
            input_panel[sheet].index = input_panel[sheet].index.map(str)

    return input_panel
