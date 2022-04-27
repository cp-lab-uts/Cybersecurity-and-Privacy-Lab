import networkx as nx 
import matplotlib.pyplot as plt
import community
from community import community_louvain
import random 
import numpy as np
import sys

class Network(object):
    def __init__(self,number,degree,net):
        """
        number: number of agents
        degree: averager degree 
        net: ScaleFree, Homogeneous or Random
        """
        self.number = number
        self.degree = degree
        self.net = net        
    
    def adjacencyMatrix(self,edges):
        """ Adjacency matrix """
        Matrix = np.zeros([self.number,self.number])
        for item in edges:
            Matrix[item[0]][item[1]] = 1.
            Matrix[item[1]][item[0]] = 1.
        return np.array(Matrix)


    def generateNetworks(self,DegreeDistribution=False):
        if self.net == 'ScaleFree':
            real_net= nx.random_graphs.barabasi_albert_graph(self.number,self.degree)   
        elif self.net == 'Homogeneous':
            real_net=  nx.random_graphs.random_regular_graph(self.degree,self.number) 
        elif self.net =='Random':
            real_net = nx.random_graphs.erdos_renyi_graph(self.number,self.degree/self.number) 
        else:
            print("Erro")
     

        edges = list(real_net.edges())
        matrix = self.adjacencyMatrix(edges)
        
        # make the zero degree nodes connected
        if self.net =='Random':
            matrix = self.processMatrix(matrix)
        
        if DegreeDistribution == True:
            x,y = self.getDegree(np.sum(matrix,1))
            self.Bar(x,y)
        return matrix
    
    def processMatrix(self,matrix):
        NewMatrix = matrix.copy()
        arg = np.argwhere(np.sum(matrix,1)==0)
        for item in arg:
            randomNum = random.randint(0,self.number-1)
            for i in range(3):
                while randomNum==item[0]:
                    randomNum = random.randint(0,self.number-1)
                NewMatrix[item[0]][randomNum] = 1
                NewMatrix[randomNum][item[0]] = 1
        assert len(np.argwhere(np.sum(NewMatrix,1)==0))==0
        return NewMatrix
    

    

        
    def dynamic_networks(self, changes):
        """ generate dynamic networks based on generateNetwork()
            changes: the number of links that changes
        """
        i = 0
        j = 0
        matrix = self.generateNetworks()
        
        while i <changes:
            change_neighbors = (random.sample(range(0,self.number - 1),2)) # choose the link that changes between agents
            if matrix[change_neighbors[0],change_neighbors[1]] == 1:
                matrix[change_neighbors[0],change_neighbors[1]] = 1 - matrix[change_neighbors[0],change_neighbors[1]]
                i = i +1
                #if i == changes:
                    #continue
        while j <changes:
            change_neighbors = (random.sample(range(0,self.number - 1),2)) # choose the link that changes between agents
            if matrix[change_neighbors[1],change_neighbors[0]] == 0:
                matrix[change_neighbors[1],change_neighbors[0]] = 1 + matrix[change_neighbors[1],change_neighbors[0]]
                j = j +1
                #if i == changes:
                    #continue   
 
        return matrix