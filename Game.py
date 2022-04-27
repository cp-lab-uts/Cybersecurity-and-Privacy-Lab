import networkx as nx 
import matplotlib.pyplot as plt
import community
from community import community_louvain
import random 
import numpy as np
from Network import Network
import sys

class Game(object):
    def __init__(self,number,rule,fracC,fracM, net, gameType):
        """
        number: number of agents
        rule: IBN, IBS and Redistribution
        run: number of simulation
        round: number of iteration 
        fraction: initial fraction of cooperators 50%C-50%D
        """
        self.number = number      
        self.rule = rule
        self.fracC = fracC
        self.fracM = fracM
        #self.Matrix = Matrix
        self.gameType = gameType
        self.net = net
        self.neighbors = {}
        #for i in range(self.number):
            #self.neighbors[i] = list(np.reshape(np.argwhere(self.Matrix[i]==1),(1,-1))[0]) 
            #assert len(self.neighbors[i])>0
        # r: cooperation   b: defector  
        self.strategy = np.array(['r']*round(number*fracC)+['b']*round(number-number*fracC))
        random.shuffle(self.strategy)  
        if gameType == "Malicious":
            self.MalicAgent = np.array(sorted(random.sample(range(number),int(fracM*number))))
    
    #def create_network(self):
        #network = Network(self.number, 4, self.net)
        #Matrix = network.dynamic_networks(10)
        #self.Matrix = Matrix
        #for i in range(self.number):
            #self.neighbors[i] = list(np.reshape(np.argwhere(self.Matrix[i] == 1),(1,-1)) [0]) 
            
        #return self.Matrix
     
    def create_network(self):
        network = Network(self.number, 4, self.net)
        Matrix = network.generateNetworks()
        self.Matrix = Matrix
        for i in range(self.number):
            self.neighbors[i] = list(np.reshape(np.argwhere(self.Matrix[i] == 1),(1,-1)) [0]) 
            
        return self.Matrix
        
    
    
    def CoopRate(self):
        """  proportion of cooperation """
        return list(self.strategy).count('r')/self.number
        
    
    def ruleBS(self):
        """  Imitate-best-strategy (BS)  """
        s = np.array(['r']*self.number)
        Stra = self.strategy.copy()
        Stra = np.where(Stra=='r',1,0)     
        MatrixPlus = self.Matrix.copy()  
        for i,val in enumerate(Stra):
            MatrixPlus[i,i] = 1
        CMatrix = np.tile(Stra,[self.number,1])*MatrixPlus
        fitMatrix = np.tile(self.payoff,[self.number,1])*MatrixPlus              
        CPayoff = np.sum(CMatrix*fitMatrix,1) 
        DPayoff = np.sum(fitMatrix,1) - CPayoff
        s[np.argwhere(CPayoff<DPayoff)] = 'b'                
        return s
    
    def ruleBN(self):
        """ Imitate-best-neighbor (BN):  """
        MatrixPlus = self.Matrix.copy()
        for i in range(self.number):
            MatrixPlus[i,i] = 1
        fitMatrix = np.tile(self.payoff,[self.number,1])*MatrixPlus
        arg = np.argmax(fitMatrix,1)
        s = []
        for val in arg:
            s.append(self.strategy[val])
        return np.array(s)  
    
    def ruleRedistribution(self):         
        # probability an agent imitate its neighbor 
        p = np.zeros((self.number,self.number),dtype=np.float32)
        s = []
        for i in range(self.number):
            arg = []
            for neigb in self.neighbors[i]:
                if self.payoff[neigb]>self.payoff[i]:
                    arg.append(neigb)
            if len(arg)==0:
                p[i,i]=1.0
                s.append(self.strategy[i])
            else:
                p[i][arg] = 1/(1+np.exp(self.payoff[i]-self.payoff[arg]))
                p[i][arg] /= np.sum(p[i][arg])
                randnum = np.random.rand(1)
#                 print(p[i][arg],"  ",list(np.where(p[i][arg].cumsum()>randnum)))
#                 print( np.where(p[i][arg].cumsum()>randnum)[0].shape)
                index = np.where(p[i][arg].cumsum()>=randnum)[0][0]
#                 print(p[i][arg],"  ",index)
                s.append(self.strategy[arg[index]][0])
        return np.array(s)        
    
    def Beneficiary(self,d):
        matrix = np.zeros((self.number,self.number))
        for i in range(self.number):
            neigbors = self.neighbors[i]
            if len(neigbors)<d:
                benefitNeigbors = neigbors
            else:
                benefitNeigbors = random.sample(neigbors,d) 
            for val in benefitNeigbors:
                matrix[i,val] = 1
        return matrix
    
    def fitness(self,payoff,d=1,theta=0.2,alpha=0.1):
        f = payoff*0.0
        pay = payoff.copy()
        pay[np.argwhere(payoff<theta)]=theta        
        BenefitMatrix = self.Beneficiary(d)   
        for i,val in enumerate(payoff):  
            f[i] = max((1-alpha)*(payoff[i]-theta),0)+ np.sum(BenefitMatrix[:,i]*alpha*(np.array(pay)-theta))/d + min(theta,payoff[i])
        return f
    
    def getPayoff(self,T=1.2,R=1.,P=0.1,S=0.):
        """compute payoff of each agent"""     
        payoff = []
        for i in range(self.number):
            arg = self.neighbors[i]
            niC = list(self.strategy[arg]).count('r')
            niD = list(self.strategy[arg]).count('b')
            assert niC+niD == np.sum(self.Matrix[i])
            if self.strategy[i] == 'r':
                payoff += [niC*R+niD*S]
            else:                 
                payoff += [niC*T+niD*P]
        if self.rule == 'Redistribution': 
            self.payoff = self.fitness(np.array(payoff))
        else:
            self.payoff = np.array(payoff)
    
    def updateStrategy(self):
        if self.rule == 'Redistribution':
            s = self.ruleRedistribution()
        elif self.rule == 'BS':
            s = self.ruleBS()
        elif self.rule == 'BN':
            s = self.ruleBN()
        else:
            print("Please choose a right strategy")
        
        # random action for maliciouss
        if self.gameType == "Malicious":      
            s[self.MalicAgent] =  [random.sample(['r','b'],1)[0] for i in range(len(self.MalicAgent))]   
        self.strategy = s

    def cheat(self):
        cheatPayoff = self.payoff.copy()
        malicStrategy = self.strategy[self.MalicAgent]
        
        CMali = np.argwhere(malicStrategy=='r')
        if len(CMali)>0:
            cheatPayoff[self.MalicAgent[CMali]] *= 0.2        
        DMali = np.argwhere(malicStrategy=='b')
        if len(DMali)>0:
            cheatPayoff[self.MalicAgent[DMali]] *= 3.0  
        self.payoff = cheatPayoff

    def play(self,rule,rounds,index,string):
        CRate = np.zeros(rounds,dtype=np.float32)
        for i in range(rounds):    
            
            #if i% 100 == 0:
            self.create_network()
            
            self.getPayoff()
            if self.gameType == "Malicious":
                self.cheat()
            self.updateStrategy()
            CRate[i] = rate = self.CoopRate() 
            sys.stdout.write('\r>>Iteration: %d/%d \t Epoch: %d/%d \t Cooperation proportion: %2f' % (index+1,100,i+1,rounds,rate))  
            sys.stdout.flush()
            '''
            if string == 'InitialCL':
                if rate==0 or rate==1.:                
                    return rate   
                if i>50 and len(np.unique(CRate[i-20:i]))==1:
                    return rate
                if i>100 and np.std(CRate[i-30:i])<0.01:
                    return np.average(CRate[i-30:i])                
            elif string == 'Epoch':
                if rate==0 or rate==1.or (i>50 and len(np.unique(CRate[i-20:i]))==1):               
                    CRate[i:]=rate
                    break
        if string == 'InitialCL':
            return np.average(CRate[i-30:i])
            '''
        return CRate                      