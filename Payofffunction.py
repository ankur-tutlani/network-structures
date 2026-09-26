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



def perturbed_response4_v2(num_agents,agent_and_strategy_pd2,potential_edges,fixed_agents,fixed_strategy_to_use):
    agent_strategy_to_choose=[]

    if len(fixed_agents) > 0:

        for j in range(num_agents):
            if j in fixed_agents:
                agent_strategy_to_choose.append(fixed_strategy_to_use)

            else:

                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                data_to_check = agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)

                if data_to_check.loc[data_to_check['agent_no']==j]['payoff'].values[0] == max(data_to_check['payoff']):
                    agent_strategy_to_choose.append(data_to_check.loc[data_to_check['agent_no']==j]['strategy'].values[0])
                else:
                    xx = list(set(data_to_check.loc[(data_to_check['agent_no']!=j) & (data_to_check['payoff']==max(data_to_check['payoff']))]['strategy'].tolist()))
                    agent_strategy_to_choose.append(random.choices(population=xx,k=1)[0])

    else:
        
        for j in range(num_agents):
            try:
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                data_to_check = agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)

                if data_to_check.loc[data_to_check['agent_no']==j]['payoff'].values[0] == max(data_to_check['payoff']):
                    agent_strategy_to_choose.append(data_to_check.loc[data_to_check['agent_no']==j]['strategy'].values[0])
                else:
                    xx = list(set(data_to_check.loc[(data_to_check['agent_no']!=j) & (data_to_check['payoff']==max(data_to_check['payoff']))]['strategy'].tolist()))
                    agent_strategy_to_choose.append(random.choices(population=xx,k=1)[0])

            except:
                
                agent_strategy_to_choose.append(agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no']==j]['strategy'].values[0])
                
            

    return agent_strategy_to_choose