import matplotlib.pyplot as plt
import numpy as np 
import numpy.linalg as npla 
import copy
import itertools as it
from scipy.special import comb
from memoization import cached
import matplotlib
import math
import collections
import sys
import itertools
from numpy.polynomial.polynomial import polyfit
import graphviz
import networkx as nx


def stationaryDistComplex(g, n):
  """
  takes in any graph, and gives a stationary distribution for districting plans
  where all districting plans have n districts
  """
  if metagraphConnected(g, n):
    return stationaryDist(g, n)
  else:
    stationaryDistNotErgodic(g,n)

def metagraphConnected(g, n):
  """
  Takes in a graph (g) and number of districts (n) and will return True 
  if the metagraph from the districting plans is connected
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

  # to draw metagraph:
  # plt.figure()
  # nx.draw(meta)
  # plt.show()

  #check if graph is connected 
  return nx.is_connected(meta)
  
def stationaryDist(g, n):
  """ stationaryDist calculates the stationary distribution for ergodic graphs
      inputs: g (graph), n (integer) or the number of districts in a plan
  """
  matrix = transitionMatrix(g, n)
  numPlans = len(distPlans(g, n))
  for i in range(numPlans):
    matrix[i][i] = matrix[i][i] -1

  P = np.array(matrix)
  np.set_printoptions(threshold=sys.maxsize)
  for i in range(numPlans):
    P[i][numPlans-1] = 1
  #to find stationary distribution:
  X = np.zeros((1,numPlans))
  X[0,numPlans-1] = 1 
  X1 = X.T 
  Y = np.linalg.solve(P.T, X1) 
  PI = Y.T 
  print(numPlans) 
  print('# probability groupings: ', numDifferentGroups(g, distPlans(g, n))) 
  print('solution to version 1, pi:')
  print(PI) 
  distplan = distPlans(g, n)

  # #Drawing grid graphs 
  for i in range(numPlans):
    print("plan", i)
    print("prob = ", (PI[0][i]))
    print("plan= ", distplan[i])
    # # avEdges(g, distplan[i])
    # y.append(mostEdges(g, distplan[i]))
    plt.figure() 
    pos = {0:(0,0),1:(1,-0.25),2:(2,0),3:(3,-0.25),4:(0.25,1),5:(1,0.75),6:(2.25,1),7:(3.25,0.75),8:(0,2),9:(1,1.75),10:(2,2),11:(3,2)}
    nx.draw(g, pos=pos,node_color = [distplan[i][x] for x in g.nodes()], with_labels = True, node_size = 1000, font_size = 20 )
    plt.show()

def stationaryDistNotErgodic(g, n):
  """
  Takes in a graph g and returns the stationary distribution of all the n-district plans from g
  """
  matrix = transitionMatrix(g, n)
  numPlans = len(distPlans(g, n))

  #to find stationary distribution (P^100); change range to get different P^x
  for i in range(20):
    temp = np.dot(matrix,matrix)
    matrix = temp

  P = np.array(matrix)
  print(P[0])
  print(numPlans)
  print('Matrix:')
  print(P) 

  distplan = distPlans(g, n)
  #Drawing grid graphs 
  for i in range(numPlans):
    print ("plan", i)
    print("prob = ", (P[i][i]))
    plt.figure() 
    nx.draw(g, node_color = [distplan[i][x] for x in g.nodes()], with_labels = True, node_size = 1000, font_size = 20 )
    plt.show()
    
def transitionMatrix(g, n):
  """ transitionMatrix takes in a graph and returns a transition matrix 
      where each (i,j) entry is the probability of transitioning from the ith districting plan
      to the jth districting plan and n is number of districts
  """
  l = distPlans(g, n)
  m = len(l)
  P = np.zeros((m,m))

  for i in range(m):
    for j in range(m):
      P[i][j] = probTransition(g, l[i], l[j])
  return P 
  
@cached
def probTransition(g,p1,p2):
  """probTransition takes in a graph (g) and two districting plans (dictionaries), 
     p1 and p2 from the same graph and returns the probability of transitioning 
     from p1 to p2 (number in between 0 and 1)
  """
  # if there are no districts in common probability is 0
  # if > or < 2 districts in that are different, no way to do recomb
  numInComm = len(distsInComm(p1, p2))
  if numInComm < max(p1.values())-1:
    return 0

  # elif exactly 2 districts that are different 
  elif numInComm == (max(p1.values())-1):
    newGraph = copy.deepcopy(g)
    #find the nodes district in common
    #d = # of districts 
    d = int(max(p1.values()) + 1)
    #r = # of nodes in a district
    r = int(len(g)/d)
    for i in range(numInComm):
      for j in range(r):
        tempNode = distsInComm(p1, p2)[i][j]
        newGraph.remove_node(tempNode)

    #find number of spanning trees for newGraph and how many of those spanning trees produce the end result
    m = int(multiplier(newGraph, p2))

    #Find determinant (det = number of spanning trees)
    det = spanning_tree(newGraph)
 
    #calculate probability
    #edges finds total number of nodes in 2 districts combined 
    #to see how many edges the spanning tree has for those 2 districts
    edges = int(((2/d)*(len(g))) - 1)
    #pick finds the number of ways to pick 2 districts from the initial number of districts
    pick = int(comb(d, 2))
    prob = m*(1/pick)*(1/det)*(1/(edges))
    return prob
    
  # if p1 and p2 are same districting plan 
  # probability = 1 - (sum of other probabilities)
  else:
    # remove p2 from other_plans (list of all possible plans)
    other_plans = [k for k in distPlans(g, (max(p1.values())+1)) if k != p2]

    # sumPlanProbs is sum of probabilities from p1 -> all plans in other_plans
    # initialize sumPlanProbs
    sumPlanProbs = 0

    # calculate probabilities for every plan in other_plans
    for i in other_plans:

      # sums probabilities of other_plans
      sumPlanProbs = sumPlanProbs + probTransition(g, p1, i)
    
    # p(p1 -> p1) = 1 - p(p1 -> other_plans)
    prob3 = 1 - sumPlanProbs
    return prob3

def multiplier(g, finalPlan):
  '''
  Calculates how many different sequences exist that will get you from one plan to another
  graph input will always be 2 districts and have all districts that were in common removed
  finalPlan is a dictionary 
  '''
  #iterate through edges, if an edge connects 2 districts, we add to our counter
  j = 0
  for edge in g.edges():
    if finalPlan.get(edge[0]) != finalPlan.get(edge[1]):
      j += 1

  first = []
  second = []
  #key will be what district the smallest node in graph is in
  key = finalPlan.get(min(g.nodes()))

  #create a list for nodes in each district
  for i in g.nodes():
    if finalPlan.get(i) == key:
      first.append(i)
    else:
      second.append(i)
       
  distA = g.subgraph(first)
  distB = g.subgraph(second)
  #find number of spanning trees in each district
  a = spanning_tree(distA)
  b = spanning_tree(distB)

  return int(j*a*b)

def spanning_tree(newGraph):
  """
  Takes in a graph and will return the number of spanning trees in the graph
  """
  #Get the laplacian of newGraph
  lapl = nx.laplacian_matrix(newGraph).toarray()

  #Turn into a numpy array, not a scipy array
  laplnp = np.matrix(lapl)

  #Delete a row and column, specifically the last row and column
  laplminor = laplnp[:-1,:-1]
  
  #Find determinant (det = number of spanning trees)
  det = round(npla.det(laplminor))  

  return det

def distsInComm(plan1, plan2):
  """
  Takes in two districting plans (dictionaries) and returns a list of tuples of the districts in common
  """
  newPlan1 = []
  newPlan2 = []
  for i in range(max(plan1.values())+1):
    dist1 = []
    dist2 = []
    #need max of max because we want j to go to the highest labeled dist
    for j in range(max(max(plan1.keys()),max(plan2.keys()))+1):    
      if plan1.get(j) == i:
        dist1.append(j)
     
      if plan2.get(j) == i:
        dist2.append(j)
    
    newPlan1.append(tuple(sorted(dist1)))
    newPlan2.append(tuple(sorted(dist2)))
   
  p1 = set(newPlan1)
  p2 = set(newPlan2)
  common = p1.intersection(p2)
  common1 = list(common)
  return common1

@cached
def distPlans(g, n):
  return dPlans(g, n, {}, [], 0)

def dPlans(g, n, plan = {}, plans = [], counter = 0):
  '''
  Takes in a graph g, and an int n, where n is the number of districts you want all your plans to have and returns all districting plans
  n must be >= 2
  '''
  # if len(g)%n != 0:
  #   return
  if n==1:
    if nx.is_connected(g):
      lastDist = list(g.nodes())
      for k in lastDist:
        if k in plan:
          del plan[k] 
      t = max(plan.values())
      for k in lastDist:
        plan[k] = int(t + 1)
      plan = dict(sorted(plan.items()))
      plans.append(plan)

  else:
  # Loop through all possible plans
    distSize = int(len(g)/n)
    edges = g.edges()
    nodes = list(g.nodes())

    # make district 0: all connected sets of 4 vertices including the vertex 0
    district0conn = [d for d in list(it.combinations(g.nodes(), distSize)) if (nodes[0] in d) and (nx.is_connected(g.subgraph(d)))]

    #Loop through all possible district 0's
    for d0 in district0conn:
      # find vertex that has to be in district 1
      found1 = False
      dist1vtx = list(g.nodes())[0]
      for i in range(len(g)):
        # if i is the first vertex outside of district 0 that we've found
        #print(i)
        #print(d0)
        if i not in d0 and found1 == False:
          dist1vtx = list(g.nodes())[i]  # this vertex must be in district 1
          found1 = True
                
      # make list of vertices not in district 0
      notindistrict0 = [v for v in g.nodes() if v not in d0]
      y = nx.Graph()
      for i in notindistrict0:
        for j in notindistrict0:
          y.add_node(i)
          if i != j and (i,j) in edges:
            y.add_edge(i,j)

      for v in d0:
         plan[v] = counter

      dPlans(y, n-1, plan, plans, counter+1)
    
    return plans