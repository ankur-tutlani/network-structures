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



def simulation_function_neighbors2(random_seed,
                                 iteration_name,
                                 path_to_save_output,
                                 num_strategies,
                                 agent_initial_state,
                                 strategy_share,
                                 strategy_pair,
                                 payoff_values,
                                 num_agents, 
                                 num_neighbors,
                                 norms_agents_frequency,
                                 norms_time_frequency,
                                 min_time_period,
                                 enable_delta_payoff,
                                 num_of_trials,
                                 fixed_agents,
                                 fixed_strategy_to_use,
                                 function_to_use,
                                 perturb_ratio,
                                   prob_1,
                                   graphseed
                                  
                                 ):
    random_seed = random_seed
    iteration_name = iteration_name
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path_to_save_output = path_to_save_output
    num_strategies = num_strategies
    agent_initial_state = agent_initial_state
    strategy_share = strategy_share
    thisdict = strategy_pair
    random.seed(random_seed)
    payoff_values= payoff_values
    num_agents = num_agents 
    num_neighbors = num_neighbors
    norms_agents_frequency = norms_agents_frequency
    norms_time_frequency = norms_time_frequency
    min_time_period = min_time_period
    enable_delta_payoff = enable_delta_payoff
    num_of_trials =  num_of_trials
    fixed_agents = fixed_agents
    fixed_strategy_to_use = fixed_strategy_to_use
    function_to_use =  function_to_use
    perturb_ratio = perturb_ratio
    
    
    
    payoff_matrix = np.zeros((num_strategies,num_strategies))
    thisdict2 = {value:key for (key,value) in thisdict.items()}
    for i in range(len(payoff_values)):
        payoff_matrix[i,]= payoff_values[i]
        
    list_to_fill = []
    for i in range(num_strategies):
        list_to_fill.append([i]*round(strategy_share[i]*num_agents))
        
    flat_list = [item for sublist in list_to_fill for item in sublist]
    if len(flat_list) == num_agents:
        pass
    #         print("ok")
    elif len(flat_list) < num_agents:
        delta = num_agents - len(flat_list)
        for kk in range(delta):
            flat_list.append(flat_list[-1])
    #         print("last elements added")
    else:
        delta = len(flat_list) - num_agents
        for kk in range(delta):
            flat_list.pop()
    #         print("last elements deleted")

    if len(agent_initial_state) > 0:
        samplelist1 = agent_initial_state
    else:
        samplelist1 = random.sample(flat_list,len(flat_list))
    
    
    random.seed(random_seed)
    G = nx.erdos_renyi_graph(n=num_agents,p=prob_1,seed=graphseed)
    fooxg=get_details2(G)
    
    
    
    nx.draw(G,with_labels=True)
    plt.savefig(path_to_save_output+"input_network_"+iteration_name+"_"+today+".png")
    plt.clf()
    
    potential_edges = list(G.edges)
    
    ### new added
    agents_in_edges = list(set([i[0] for i in potential_edges]+[i[1] for i in potential_edges]))
    agents_in_network = list(range(num_agents))
    delta_agents_hardcoded = [i for i in agents_in_network if i not in agents_in_edges]
    ### new added
    
    agents_list = list(range(num_agents))
    agent_and_strategy_pd = pd.DataFrame({'agent_no':agents_list,'strategy':samplelist1})
    intial_states_pd = pd.DataFrame([{'initial_state':str([thisdict[k] for k in samplelist1])}],columns=['initial_state'])
    
    
    trend_db_to_store = pd.DataFrame(columns=["percent_count","strategy","timeperiod_number","starting_position"])
    payoff_matrices = []
    payoff_matrices.append(str(payoff_matrix))
    payoff_matrices_timeperiod = []
    payoff_matrices_timeperiod.append(0)
    agent_and_strategy_pd['timeperiod_number'] = 0
    unique_strategies = list(range(num_strategies))
    
    
    ### initial state graph
    names_to_check=unique_strategies
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    list_of_colors = get_colors(len(names_to_check))
    color_map = []
    for j in range(len(samplelist1)):
        for i in range(len(names_to_check)):
            if samplelist1[j] == names_to_check[i]:
                color_map.append(list_of_colors[i])

    for i in range(len(G)):
        G.nodes[i]["name_offered"] = [thisdict[k] for k in samplelist1][i]

    labels_for_graph = nx.get_node_attributes(G,"name_offered")
    nx.draw(G,with_labels=True,node_color=color_map,labels=labels_for_graph)
    l,r = plt.xlim()
    plt.xlim(l-0.05,r+0.05)
    plt.savefig(path_to_save_output+"networkgraph_initial_state"+'_'+iteration_name+"_"+today+".png")
    plt.clf()
    
    for timeperiod in range(1,num_of_trials+1):
        payoff_matrix_original = payoff_matrix.copy()

        agent_payoffs_to_store = []
        for j in range(num_agents):
            try:
                
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                flat_list1.remove(j)
                row_strategy = agent_and_strategy_pd.loc[agent_and_strategy_pd['agent_no'] == j]['strategy'].values[0]
                data_to_check = agent_and_strategy_pd.loc[agent_and_strategy_pd['agent_no'].isin(flat_list1)].reset_index(drop=True)
                agent_payoff = 0
                for i in range(len(data_to_check)):
                    agent_payoff += payoff_matrix[row_strategy,data_to_check.iloc[i]['strategy']] 
                agent_payoffs_to_store.append(agent_payoff)
            except:
                agent_payoffs_to_store.append(None)


        agent_and_strategy_payoff = pd.DataFrame({'agent_no':agents_list,'payoff':agent_payoffs_to_store})        
        agent_and_strategy_pd2 = pd.merge(agent_and_strategy_pd,agent_and_strategy_payoff,on='agent_no')
        agent_strategy_to_choose = perturbed_response4_v2(num_agents,agent_and_strategy_pd2,potential_edges,fixed_agents,fixed_strategy_to_use)
        agent_and_strategy_pd= pd.DataFrame({'agent_no':agents_list,'strategy':agent_strategy_to_choose})
        
        data1_to_store = agent_and_strategy_pd['strategy'].value_counts(normalize=True).to_frame()
        data1_to_store.columns = ['percent_count']
        data1_to_store['strategy'] = data1_to_store.index.tolist()
        data1_to_store = data1_to_store.replace({"strategy":thisdict})
        data1_to_store["timeperiod_number"] = timeperiod
        data1_to_store["starting_position"] = str([thisdict[k] for k in samplelist1])
        trend_db_to_store = pd.concat([trend_db_to_store,data1_to_store],ignore_index=True)
         
        
    payoff_matrices_pd = pd.DataFrame(payoff_matrices)
    payoff_matrices_pd.columns = ['payoff_matrix']
    payoff_matrices_pd['timeperiod'] = payoff_matrices_timeperiod
    
    #### strategy converted to norm ####
    norms_to_store = []
    percent_time_frequency = []
    norms_candidates = trend_db_to_store.loc[trend_db_to_store['percent_count']>= norms_agents_frequency]
    potential_norms_candidate = np.unique(norms_candidates['strategy']).tolist()
    for k in potential_norms_candidate:
        norms_candidates2 = norms_candidates.loc[norms_candidates['strategy']==k]
        distinct_timeperiod = len(np.unique(norms_candidates2['timeperiod_number']))
        norms_candidates2 = round(distinct_timeperiod/len(np.unique(trend_db_to_store['timeperiod_number'])),2)
        if norms_candidates2 >= norms_time_frequency:
            norms_to_store.append(k)
            percent_time_frequency.append(norms_candidates2)
    
    
    try:
        norms_candidates2 = pd.DataFrame()
        norms_candidates2["percent_count"] = percent_time_frequency
        norms_candidates2["name"] = norms_to_store
        if len(norms_candidates2) > 0:
            norms_candidates2.to_excel(path_to_save_output+"normcandidates_"+iteration_name+"_"+today+".xlsx",index=None)
    except:
        pass
    
    
    
    ##### distribution at the end of timeperiod.
    color_map = []
    for j in range(len(agent_strategy_to_choose)):
        for i in range(len(names_to_check)):
            if agent_strategy_to_choose[j] == names_to_check[i]:
                color_map.append(list_of_colors[i])

    for i in range(len(G)):
        G.nodes[i]["name_offered"] = [thisdict[k] for k in agent_strategy_to_choose][i]

    labels_for_graph = nx.get_node_attributes(G,"name_offered")
    nx.draw(G,with_labels=True,node_color=color_map,labels=labels_for_graph)
    l,r = plt.xlim()
    plt.xlim(l-0.05,r+0.05)
    plt.savefig(path_to_save_output+"network_after_"+str(num_of_trials)+'_timeperiods_'+iteration_name+"_"+today+".png")
    plt.clf()
    
    
    ###  trend graph
    fig,ax = plt.subplots()
    data_for_trend_plot = trend_db_to_store.loc[trend_db_to_store['starting_position']==str([thisdict[k] for k in samplelist1])]
    data_for_trend_plot = data_for_trend_plot.reset_index(drop=True)
    for label,grp in data_for_trend_plot.groupby('strategy'):
        grp.plot(x='timeperiod_number',y='percent_count',ax=ax,label=label)
    ax.set_xlabel('Timeperiod')
    ax.set_ylabel('Count %')
    # plt.show()
    plt.savefig(path_to_save_output+"strategy_trend_"+iteration_name+"_"+today+".png")
    plt.clf()
    
    
    ###### dataframe of strategy the timeperiod when it reached the norms stage #####
    db_to_fill2 = pd.DataFrame()
    db_to_fill2["timeperiod"] = -1
    db_to_fill2["name_offered"] = -1
    if len(norms_to_store) > 0:
        for j in norms_to_store:
            foocheck = data_for_trend_plot.loc[data_for_trend_plot["strategy"]==j]
            foocheck = foocheck.loc[foocheck['percent_count']>= norms_agents_frequency].reset_index(drop=True)
            foocheck = foocheck.sort_values(["timeperiod_number"])
            foocheck["count_names_offered"] = (foocheck["strategy"]==j).cumsum()
            foocheck["cum_perc"] = foocheck["count_names_offered"]/len(np.unique(data_for_trend_plot['timeperiod_number']))
            xxxx= foocheck.loc[foocheck["cum_perc"]>=norms_time_frequency][["timeperiod_number"]].head(1)
            if xxxx.shape[0] > 0:
                timev = foocheck.loc[foocheck["cum_perc"]>=norms_time_frequency][["timeperiod_number"]].head(1)["timeperiod_number"].values[0]
                foodb = pd.DataFrame({"timeperiod":[timev],"name_offered":[j]})
                db_to_fill2 = pd.concat([db_to_fill2,foodb],ignore_index=True,sort=False)
                
    
    
    try:
        if len(db_to_fill2) > 0:
            db_to_fill2.to_excel(path_to_save_output+"time_when_reached_norm_"+iteration_name+"_"+today+".xlsx",index=None)
    except:
        pass
    
    
    trend_db_to_store.to_excel(path_to_save_output+"aggregate_data_detailed_agent_"+iteration_name+"_"+today+".xlsx",index=None)

    
    parameters_pd = pd.DataFrame([{'random_seed':random_seed,'iteration_name':iteration_name,
                              'path_to_save_output':path_to_save_output,'num_strategies':num_strategies,
                              'agent_initial_state':str(agent_initial_state),'strategy_share':str(strategy_share),
                              'strategy_pair':str(strategy_pair),'payoff_values':str(payoff_values),
                              'num_agents':num_agents,'num_neighbors':num_neighbors,
                                   'norms_agents_frequency':norms_agents_frequency,
                              'norms_time_frequency':norms_time_frequency,'min_time_period':min_time_period,
                              'enable_delta_payoff':enable_delta_payoff,'num_of_trials':num_of_trials,
                              'fixed_agents':str(fixed_agents),'fixed_strategy_to_use':fixed_strategy_to_use,
                              'function_to_use':function_to_use,'perturb_ratio':perturb_ratio,
                                   'prob_1':prob_1,
                                   'density':fooxg[0],
                                   'diameter':fooxg[1],
                                   'clustering':fooxg[2],
                                   'fattailedness':fooxg[3],
                                   'graphseed':graphseed,
                              'datetime':today}]).T
    parameters_pd.columns=["parameter_values"]
    parameters_pd["parameter"]=parameters_pd.index
    parameters_pd[["parameter","parameter_values"]].to_excel(path_to_save_output+"parameters_"+iteration_name+"_"+today+".xlsx",index=None)
    
    intial_states_pd.to_excel(path_to_save_output+"initial_state_considered_"+iteration_name+"_"+today+".xlsx",index=None)
    payoff_matrices_pd.to_excel(path_to_save_output+"payoff_matrices_considered_by_timeperiod_"+iteration_name+"_"+today+".xlsx",index=None)

    return(print("done"))