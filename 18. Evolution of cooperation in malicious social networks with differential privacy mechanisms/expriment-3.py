
# coding: utf-8

# In[22]:


import random
import numpy as np
import sys
import pandas as pd 

from Network import Network

alpha = 0.7
gamma = 0.1
number = 200
degree = 4
fractionC = 0.5
T = 1.2
R = 1.
P = 0.1
S = 0.
mu = 0.01
class ReinforcementLearning(object):
    def __init__(self, number, fractionC, fractionM, gameType, mu,net):
            self.number = number
            self.fractionC = fractionC
            self.fractionM = fractionM
            self.gameType = gameType
            self.mu = mu
            self.strategy = np.array([1] * round(self.number * self.fractionC) + [0] * round(self.number * (1 - self.fractionC)))
            random.shuffle(self.strategy)  
            self.net = net
            self.neighbors = {}
            self.h_action = {}
            self.QValue = {} # user: Q-value for two rows, cooperate and defect
            self.Pi = {}  # user: probability for two rows, the probability to cooperate and defect 
            self.payoffs = np.zeros(self.number) # each agent payoff of its own
            self.Reward = np.zeros(self.number) # the sum of an agent's payoff and its neighbors' payoff
            self.expected_QValue = np.ones(self.number)
            self.probability_co = np.ones(self.number) # historical records for cooperation for all agents
            self.niC = np.ones(self.number)
            self.niC = self.niC.astype(int)
            self.niD = np.ones(self.number)
            self.niD = self.niD.astype(int)
   
    def create_network(self, net):
        network = Network(number, 4, net)
        matrix = network.generateNetworks()
        self.matrix = matrix
        for i in range(self.number):
            self.neighbors[i] = list(np.reshape(np.argwhere(self.matrix[i] == 1),(1,-1)) [0]) 
            
        return self.matrix


    def initial(self):
        """ define initial Q-value, policy, users' neighbors and pi"""
        
        degree = self.matrix.sum(axis = 1) + 100  # the number of neighbors for each agent
        
        if self.gameType in ["Malicious","Malicious_DP"]:
            self.MalicAgent = np.array(sorted(random.sample(range(number),int(fractionM*number)))) #define who are mulicious agents
        
        for i in range(self.number):
            a = np.random.laplace(1.5, 0.2,int(degree[i]) + 1)
            b = np.random.laplace(1.5, 0.2,int(degree[i]) + 1)
            #a = np.ones(int(degree[i]) + 1)
            #b = np.ones(int(degree[i]) + 1)
            self.QValue[i] = np.concatenate(np.expand_dims((a,b),1))
            self.Pi[i]= np.ones((2,(int(degree[i]) + 1))) * 0.5

            #self.neighbors[i] = list(np.reshape(np.argwhere(self.matrix[i] == 1),(1,-1)) [0]) 
            self.h_action[i] = np.ones(2) # the first is to cooperate，the second is to defect
        return self.QValue, self.Pi, self.neighbors, self.h_action


    def computePayoff(self):
        """all agents' payoff"""

        #self.payoffs = np.zeros(self.number) # each agent payoff of its own
        for i in range(self.number):
            state = self.strategy[self.neighbors[i]] # each agent's neighbor's state
            self.niC[i] = len(np.argwhere(state == 1))
            #self.niD[i] = len(state) - self.niC[i]  
            self.niD[i] = len(np.argwhere(state == 0)) 
            if self.strategy[i] == 1:
                payoff = self.niC[i] * R + self.niD[i] * S
            else:
                payoff = self.niC[i] * T + self.niD[i] * P
            self.payoffs[i] = payoff
        return self.payoffs, self.niC, self.niD,self.strategy
    
    def DPNoise(self,data):         
        data_noise = np.exp(self.mu * data)
        data_noise /= np.sum(data_noise)
        return data_noise
    
    
    def computeReward(self):
        """all agent's reward"""
        if self.gameType in ["Malicious","Malicious_DP"]: 
            for m in self.MalicAgent:
                if self.strategy[m] == 0:
                    self.payoffs[m] *= 3.0
                else:
                    self.payoffs[m] *= 0.2
        
        for i in range(self.number):
            rewards = np.zeros(len(self.neighbors[i]) + 1)
            rewards[:len(self.neighbors[i])] =  self.payoffs [self.neighbors[i]] #
            rewards[-1] = self.payoffs[i]       

            weight = rewards / np.sum(rewards) 
            if self.gameType == "Malicious_DP":
                weight = self.DPNoise(weight)
            
            
            self.Reward[i] = np.sum(weight * rewards) 
        return self.Reward
    

    def computeQvalue(self):
        """compute expected_Qvalue and Qvalue"""
        #probability_co = np.ones(self.number) # historical records for cooperation for all agents
        #expected_QValue = np.ones(self.number)
        for i in range(self.number):
             self.probability_co[i] = (self.h_action[i][0] / np.sum(self.h_action[i])) 

        for j in range(self.number):    
            expected_reward_c = (self.probability_co[self.neighbors[j]] * (S + R)) 
            expected_reward_d = ((1 - self.probability_co[self.neighbors[j]]) * (T + P))
            self.expected_QValue[j] = max(sum(expected_reward_c),sum(expected_reward_d))
            
            self.QValue[j][self.strategy[j]][self.niC[j]] = (1-alpha) * self.QValue[j][self.strategy[j]][self.niC[j]] + alpha * (self.Reward[j]+ gamma * self.expected_QValue[j] )
            #self.expected_QValue[j] = expected_QValue[j]
        return self.expected_QValue,self.QValue,self.probability_co  
    
    def updatePi(self):
        for i in range(self.number):
            average_reward  = self.Pi[i][self.strategy[i]][self.niC[i]] * self.QValue[i][self.strategy[i]][self.niC[i]] + self.Pi[i][1-self.strategy[i]][self.niC[i]] * self.QValue[i][1-self.strategy[i]][self.niC[i]]
            zeta = 0.1
            self.Pi[i][self.strategy[i]][self.niC[i]] = self.Pi[i][self.strategy[i]][self.niC[i]] + zeta * (self.QValue[i][self.strategy[i]][self.niC[i]] - average_reward)
            self.Pi[i][1-self.strategy[i]][self.niC[i]] = self.Pi[i][1-self.strategy[i]][self.niC[i]] + zeta * (self.QValue[i][1-self.strategy[i]][self.niC[i]] - average_reward)
            
            if  (self.Pi[i][self.strategy[i]][self.niC[i]] < 0):
                self.Pi[i][self.strategy[i]][self.niC[i]] = 0.001
            if  (self.Pi[i][1-self.strategy[i]][self.niC[i]] < 0): 
                self.Pi[i][1-self.strategy[i]][self.niC[i]] = 0.001
            self.Pi[i][self.strategy[i]][self.niC[i]] = self.Pi[i][self.strategy[i]][self.niC[i]] / (self.Pi[i][self.strategy[i]][self.niC[i]] + self.Pi[i][1-self.strategy[i]][self.niC[i]])
            self.Pi[i][1-self.strategy[i]][self.niC[i]] =  self.Pi[i][1-self.strategy[i]][self.niC[i]] / (self.Pi[i][self.strategy[i]][self.niC[i]] + self.Pi[i][1-self.strategy[i]][self.niC[i]])
            
            if self.gameType in ["Malicious","Malicious_DP"] and i in self.MalicAgent:           
                self.strategy[i] = random.sample([0,1],1)[0]
                self.h_action[i][self.strategy[i]] = self.h_action[i][self.strategy[i]] + 1
            
            
            randnum = np.random.rand(1)
            if randnum > self.Pi[i][0][self.niC[i]]:
                self.strategy[i] = 1
                self.h_action[i][0] = self.h_action[i][0] + 1
            else:
                self.strategy[i] = 0
                self.h_action[i][1] = self.h_action[i][1] + 1
        return self.Pi, self.strategy
         
    def computeCLevel(self):
        self.cooperationLevel = list(self.strategy).count(1)/self.number
        return self.cooperationLevel
        
    def play(self, Epoch, string):
        self.Epoch = Epoch 
        #self.create_network()
        CRate = np.zeros(Epoch,dtype=np.float32) 
        for epoch in range(self.Epoch):
            if epoch% 100 == 0:
                self.create_network(net)
                
            self.computePayoff()
            self.computeReward()
            self.computeQvalue()
            self.updatePi()
            self.computeCLevel()
            CRate[epoch] = self.cooperationLevel
            #print('cooperation level is %f' % self.cooperationLevel) 
            #sys.stdout.write('\r Iteration: %d/%d \t Epoch: %d/%d \t Cooperation proportion: %2f' % (r+1,100,epoch+1,rounds,CRate))  
            #sys.stdout.flush()
            print(self.cooperationLevel)
        return CRate
            #print(self.payoffs)
            #print(self.Reward)
            #print(self.expected_QValue)
            #print(self.QValue)
            #print(self.Pi)
        


# In[19]:


mu = 0.01
number = 100
degree = 4
nets = ['Homogeneous',"ScaleFree","Random" ] 
rules = ["BS", "BN","Redistribution"]
gtypes = ['Normal', 'Mali']
gtypes = ["Malicious_DP","Normal","Malicious",]
fractionC = 0.2
fractionM = 0.2
rounds = 100
run = r = 10
epoch = rounds = 100

def create_result_set(nets, rules, gtypes, epoch):
    cooperation_rate = {}
    for key1 in nets:
        cooperation_rate[key1] = {}
        for key2 in gtypes:
            cooperation_rate[key1][key2] = np.zeros(epoch,dtype=np.float32) 
          
    return cooperation_rate

cooperation_rate=create_result_set(nets, rules, gtypes, epoch)

#np.zeros(epoch,dtype=np.float32) 


# In[ ]:


gtype = ["Malicious_DP"]
EpochResults = []
for net in nets:
        for fractionM in np.linspace(0.,0.5,10):
            Row = []
            for noise in np.linspace(0.01,0.1,10):

                 for r in range(run):
            

                    RLGame = ReinforcementLearning(number, fractionC, fractionM, gtype, mu, net)
                    #dynamic_network = DGame.create_network()
                    dynamic_network = RLGame.create_network(net)
                    initial = RLGame.initial()
                    DCRate += RLGame.play(rounds,r)
                    #cooperation_rate[net][gtype] += DCRate
                    Row.append(DCRate/run)
            EpochResults.append(Row)
        np.savetxt('Experiment2_'+net+'.txt', np.array(EpochResults), fmt='%.2f')
        data.append(np.array(EpochResults))
                #cooperation_rate[net][gtype] = np.around(cooperation_rate[net][gtype] / run,2)  
                #data = cooperation_rate
                #EpochResults.append(data[net][gtype])
    #np.savetxt('RL_schemes_'+net+'.txt', np.array(EpochResults), fmt='%.2f')


# In[21]:


cooperation_rate

