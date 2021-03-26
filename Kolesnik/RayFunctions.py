from RedistrictingFunctions import *
from AnalysisFunctions import *
from MixingTime import *

def create_meta(g, n):
    """
    Takes in a graph (g) and number of districts (n) and will return metagraph from all districting plans
    """
    #meta will be the metagraph
    meta = nx.Graph()
    numPlans = len(distPlans(g, n))

    #add nodes
    for r in range(numPlans):
        meta.add_node(r)

    l = distPlans(g, n)

    #add in edges
    for i in range(numPlans):
        for j in range(numPlans):
            if probTransition(g, l[i], l[j]) != 0 and l[i] != l[j]:
                meta.add_edge(i,j)
    
    return meta

