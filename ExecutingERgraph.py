import numpy as np   
import pandas as pd
from collections import Counter
import datetime
import random
import matplotlib.pyplot as plt
import networkx as nx
import string
from functools import reduce
import itertools


density_list=[]
diameter_list=[]
clustering_list=[]
fattailedness_list=[]
prob_list=[]
seed_list=[]
total_iterations=100
n=20

for i in range(total_iterations):
    p=round(random.uniform(0, 1),2)
    seed=random.randint(1,50000)
    G = nx.erdos_renyi_graph(n=n,p=p,seed=seed)
    prob_list.append(p)
    xx=get_details2(G)
    density_list.append(xx[0])
    diameter_list.append(xx[1])
    clustering_list.append(xx[2])
    fattailedness_list.append(xx[3])
    seed_list.append(seed)

agent_and_strategy_pd2= pd.DataFrame({'density':density_list,'diameter':diameter_list,'clustering':clustering_list,'fattailedness':fattailedness_list,'prob':prob_list,'seed':seed_list})



agent_and_strategy_pd2.loc[((agent_and_strategy_pd2['prob']>0) & (agent_and_strategy_pd2['prob']<0.2))]

prob_1 = 0.14
graphseed=35852
simulation_function_neighbors2(random_seed = 851330,
                                 iteration_name = "case1_"+str(prob_1),
                                 path_to_save_output = "C:\\Users\\erdos_renyi_graph\\",
                                 num_strategies=2,
                                 agent_initial_state =[],
                                 strategy_share = [0.5,0.5],
                                 strategy_pair = {0: "X",1: "Y"},
                                 payoff_values = [[1,0.9],[1.1,1]],
#                                  network_name = folder_path,
                                 prob_edge_rewire = 0,
                                 grid_network_m = 4,
                                 grid_network_n = 5,
                                 num_agents = 20, 
                                 num_neighbors = 2,
                                 delta_to_add = 20,
                                 norms_agents_frequency = 0.7,
                                 norms_time_frequency = 0.5,
                                 min_time_period = 20,
                                 enable_delta_payoff = "No",
                                 num_of_trials = 50,
                                 fixed_agents = [],
                                 fixed_strategy_to_use = 0,
                                 function_to_use = "perturbed_response4",
                                 perturb_ratio = 0,
                               prob_1=prob_1,
                               graphseed=graphseed
                                  
                                 )
                                 
                                 
agent_and_strategy_pd2.loc[((agent_and_strategy_pd2['prob']>0.8) & (agent_and_strategy_pd2['prob']<0.99))]


prob_1 = 0.93
graphseed=45748
simulation_function_neighbors2(random_seed = 851330,
                                 iteration_name = "case2_"+str(prob_1),
                                 path_to_save_output = "C:\\Users\\erdos_renyi_graph\\",
                                 num_strategies=2,
                                 agent_initial_state =[],
                                 strategy_share = [0.5,0.5],
                                 strategy_pair = {0: "X",1: "Y"},
                                 payoff_values = [[1,1.1],[0.9,1]],
#                                  network_name = folder_path,
                                 prob_edge_rewire = 0,
                                 grid_network_m = 4,
                                 grid_network_n = 5,
                                 num_agents = 20, 
                                 num_neighbors = 2,
                                 delta_to_add = 20,
                                 norms_agents_frequency = 0.7,
                                 norms_time_frequency = 0.5,
                                 min_time_period = 20,
                                 enable_delta_payoff = "No",
                                 num_of_trials = 50,
                                 fixed_agents = [],
                                 fixed_strategy_to_use = 0,
                                 function_to_use = "perturbed_response4",
                                 perturb_ratio = 0,
                               prob_1=prob_1,
                               graphseed=graphseed
                                  
                                 )
                                 
 