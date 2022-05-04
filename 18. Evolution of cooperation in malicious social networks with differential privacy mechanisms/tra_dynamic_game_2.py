
# coding: utf-8

# In[4]:


import numpy as np
import random 
import sys
import datetime
import parser
import argparse

import networkx as nx
import matplotlib.pyplot as plt

import community
from community import community_louvain

from Network import Network
from  Game import Game

number = 1000
degree = 4
nets = ['Homogeneous',"ScaleFree","Random" ] 
rules = ["BS", "BN","Redistribution"]
gtypes = ['Normal', 'Mali']
fractionC = 0.5
fractionM = 0.1
rounds = 500
run = r = 1


# In[5]:



epoch = rounds
def create_result_set(nets, rules, gtypes, epoch):
    cooperation_rate = {}
    for key1 in nets:
        cooperation_rate[key1] = {}
        for key2 in rules:       
            cooperation_rate[key1][key2] = {}
            for gtype in gtypes:
                cooperation_rate[key1][key2][gtype] =  np.zeros(epoch,dtype=np.float32)  
    return cooperation_rate
cooperation_rate=create_result_set(nets, rules, gtypes, epoch)

for net in nets:
    EpochResults = []
    for rule in rules:
        for gtype in gtypes:
            for r in range(run):
                DGame = Game(number, rule, fractionC, fractionM,net, gtype) 
                DCRate = DGame.play(rule, rounds, r, 'Epoch') # a set: the number of 'run' 
                cooperation_rate[net][rule][gtype] += DCRate

            cooperation_rate[net][rule][gtype] = np.around(cooperation_rate[net][rule][gtype] / run,2)
            data = cooperation_rate
            EpochResults.append(data[net][rule][gtype])
            np.savetxt('traditional_schemes_'+net+'.csv', np.array(EpochResults), fmt='%.2f')

