import networkx as nx
import numpy as np 
import numpy.linalg as npla 
import random
from scipy.spatial import Delaunay
import matplotlib
import math
import collections
import sys
from RedistrictingFunctions.py import *

'''
For many of the functions that take in a districting plan, we found it beneficial to loop through a list of all plans 
for a graph and plot the results vs probability
'''


def mostEdges(g, dist):
  '''
  Takes in graph and a districting plan, finds the two districts with the largest number of edges, 
  then returns the total number of edges in these districts and between them
  '''
  #create list of districts
  nodes = []
  r = max(dist.values()) + 1
  for i in range(r):
    dist1 = []
    for j in dist.keys():
      if dist.get(j) == i:
        dist1.append(j)
    nodes.append(dist1)
  
  #find largest district
  maximum = 0
  for n in nodes:
    subgraph = g.subgraph(n)
    num = len(subgraph.edges())
    if num > maximum:
      maximum = num
      largestDist = n

  #find 2nd largest district
  nodes.remove(largestDist)
  maximum = 0
  for n in nodes:
    subgraph = g.subgraph(n)
    if len(subgraph.edges()) > maximum:
      maximim = len(subgraph.edges())
      largestDist2 = n

  #take union of 2 largest districts and find total number of edges in union
  largest = largestDist + largestDist2
  twoCombined = g.subgraph(largest)
  return len(twoCombined.edges())

def edgesBetween(graph, sGraph):
  """
  Takes in 2 graphs, where one is a subgraph of the other, and returns the total number of edges leaving the subgraph
  The subgraph is the second argument for the function
  """
  y = []
  #go through all nodes in graph and create a subgraph with all the nodes not in sGraph
  for i in range(len(graph.nodes())):
    if i not in sGraph.nodes():
      y.append(i)
  x = graph.subgraph(y)
  total = len(graph.edges())
  #group1 = edges in subGraph
  #group2 = edges not in sGraph snd not connected to sGraph
  group1 = len(sGraph.edges())
  group2 = len(x.edges())
  #returns edges not in sGraph but connected to sGraph
  return total - group1 - group2

def avEdges(g, dist):
  """
  Takes in a graph and a districting plan and returns the average number of edges in a district
  """
  sum = 0
  #loops through all the districts 
  for i in range(max(dist.values()) + 1):
    nodes = []
    for j in dist.keys():
      if dist.get(j) == i:
        nodes.append(j)
    y = g.subgraph(nodes)
    #add edges in one of the districts to the sum
    sum += len(y.edges())
  #divide sum by number of districts to get average 
  return sum/(max(dist.values()) + 1)

def productLeavingEdges(g,dist):
  '''
  Takes in a graph and a districting plan and returns the products of edges leaving districts in a plan
  '''
  product = 1
  #loops through districts 
  for i in range(max(dist.values()) + 1):
    nodes = []
    for j in dist.keys():
      if dist.get(j) == i:
        nodes.append(j)
    y = g.subgraph(nodes)
    #multiplies number of edges leaving a district to the product
    product *= edgesBetween(g,y)
  return product 

def smdEdges(g, dist):
  """
  Takes in a graph and districting plan and returns the sum of differences between total edges in a district and the average 
  (finds standard mean deviation for number of edges in districts)
  """
  average = avEdges(g, dist)
  deviation = 0
  for i in range(max(dist.values()) + 1):
    nodes = []
    for j in dist.keys():
      if dist.get(j) == i:
        nodes.append(j)
    y = g.subgraph(nodes)
    deviation += abs(edgesBetween(g, y) - average)
  return deviation
  
def varianceEdges(g, dist):
  """
  Takes in a graph and districting plan and returns the variance of number of edges in districts 
  """
  average = avEdges(g, dist)
  var = 0
  for i in range(max(dist.values()) + 1):
    nodes = []
    for j in dist.keys():
      if dist.get(j) == i:
        nodes.append(j)
    y = g.subgraph(nodes)
    var += (edgesBetween(g, y) - average)**2
  return var

def avSpanningTrees(g, dist):
  """
  Takes in a graph and districting plan and returns the average number of spanning trees in a district for a specific plan
  """
  sum = 0
  for i in range(max(dist.values()) + 1):
    nodes = []
    for j in dist.keys():
      if dist.get(j) == i:
        nodes.append(j)
    y = g.subgraph(nodes)
    sum += spanning_tree(y)
  return sum/(max(dist.values()) + 1)

def smdTrees(g, dist):
  """
  Takes in a graph and districting plan and returns the sum of differences between total spanning trees in a district and the average 
  (finds standard mean deviation for number of spanning trees in districts)
  """
  average = spanning_trees_average(g, dist)
  deviation = 0
  for i in range(max(dist.values()) + 1):
    nodes = []
    for j in dist.keys():
      if dist.get(j) == i:
        nodes.append(j)
    y = g.subgraph(nodes)
    deviation += abs(spanning_tree(y) - average)
  return deviation

def varianceSpanningTrees(g, dist):
  """
  Takes in a graph and districting plan and returns the variance of number of spanning trees in districts
  """
  average = spanning_trees_average(g, dist)
  var = 0
  for i in range(max(dist.values()) + 1):
    nodes = []
    for j in dist.keys():
      if dist.get(j) == i:
        nodes.append(j)
    y = g.subgraph(nodes)
    var += abs(spanning_tree(y) - average)
  return var

def numDifferentGroups(g, plans):
  """
  Takes in a graph and the list of all districting plans and returns the number of different probability groupings for a graph
  (number of different total number of edges in districts)
  """
  numbers = []
  for p in plans:
    numbers.append(edgesInDists(g, p))
  #set will ensure all entires are distinct
  return len(set(numbers))

def edgesInDists(g, dist):
  '''
  Takes in a graph and a specific districting plan and 
  returns the total # of edges in a plan that are within districts 
  '''
  sum = 0
  for i in range(max(dist.values()) + 1):
    nodes = []
    for j in dist.keys():
      if dist.get(j) == i:
        nodes.append(j)
    y = g.subgraph(nodes)
    sum += len(y.edges())
  return sum

def randomPlanar(x):
  """
  Will return a random planar nx graph with x nodes that was created using delaunay triangulation
  """
  #x is number 
  #randomly add in x vertices so # vertices is fixed, where the points are in grid is variable
  p = []

  for i in range(x):
    int1 = random.random()
    int2 = random.random()
    p.append([int1,int2])

  points = np.array(p)
  tri = Delaunay(points)

  # create a set for edges that are indexes of the points
  edges = set()
  # for each Delaunay triangle
  for n in range(tri.nsimplex):
      # for each edge of the triangle
      # sort the vertices
      # (sorting avoids duplicated edges being added to the set)
      # and add to the edges set
      edge = sorted([tri.vertices[n,0], tri.vertices[n,1]])
      edges.add((edge[0], edge[1]))
      edge = sorted([tri.vertices[n,0], tri.vertices[n,2]])
      edges.add((edge[0], edge[1]))
      edge = sorted([tri.vertices[n,1], tri.vertices[n,2]])
      edges.add((edge[0], edge[1]))

  #make a graph based on the Delaunay triangulation edges
  graph = nx.Graph(list(edges))
  sortedGraph = nx.Graph()
  sortedGraph.add_nodes_from(sorted(graph.nodes()))
  sortedGraph.add_edges_from(graph.edges())
  return sortedGraph

def createGraph(s):
  """
  Creates a random graph of size s
  """
  g = nx.Graph()

  #create graph of size s
  for i in range(s):
    g.add_node(i)

  #add edge between 2 nodes with some probability,
  #if this probability is too low, doesn't make a valid graph (individual nodes will be completely disconnected)
  for i in range(s):
    for j in range(i+1,s):
      p = random.randint(0,50)
      #probability = 50/25 = .5 
      if p >= 25:
        g.add_edge(i,j)

  #to ensure you only get a connected graph
#   if nx.is_connected(g):
#       return g
#   else:
#       createGraph(s)

  return g