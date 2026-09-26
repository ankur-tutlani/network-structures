#pip install powerlaw
import networkx as nx
import powerlaw

def get_details2(G):
    density = round(nx.density(G),2)
    try:
        diameter = nx.diameter(G)
    except:
        diameter=-1
        
    clustering_coefficient = round(nx.average_clustering(G),2)
    degree_hist = nx.degree_histogram(G)
    degrees = [degree * count for degree, count in enumerate(degree_hist)]

    # Fit the power-law model
    fit = powerlaw.Fit(degrees)

    # Get the estimated power-law exponent
    exponent = fit.power_law.alpha
    if exponent < 2:
        fattailed='heavy'
    elif exponent >= 2 and exponent < 3:
        fattailed='medium'
    elif exponent > 3:
        fattailed='low'
    else:
        fattailed='NA'
        
    return density,diameter,clustering_coefficient,fattailed
    
