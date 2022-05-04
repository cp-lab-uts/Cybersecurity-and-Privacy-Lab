#!/usr/bin/env python
from __future__ import print_function
# encoding: utf-8

#MODIFIED#

# First Start the server: $> bin/start.py
import numpy as np
import tensorflow.compat.v1 as tf #启用版本1
import numpy as np
tf.disable_v2_behavior() #禁用tensorflow 版本2
from collections import deque
import argparse
import itertools
import random
import math
try:
  import hfo
except ImportError:
  print('Failed to import hfo. To install hfo, in the HFO directory'\
    ' run: \"pip install .\"')
  exit()
params = {'SHT_DST':0.136664020547, 'SHT_ANG':-0.747394386098,
          'PASS_ANG':0.464086704478, 'DRIB_DST':-0.999052871962}

Q_table = {}
pi_table = {}
reward_table = {}

key = 0

action = 0

alpha = 0.25
gamma = 0.4

teammate_uniform_number = 0

unum = 0

t = 1



# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 1000 # experience replay buffer size
BATCH_SIZE = 50 # size of minibatch

n_neural = 32


class small_DQN():

  # DQN Agent
    def __init__(self):
        # init experience replay
        
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 6 
        self.action_dim = 3
     
        self.create_Q_network()
        self.create_training_method()

        self.sess2 = tf.Session() # Session2
        

    def chongzhi(self):
        self.sess2.run(tf.global_variables_initializer())
        self.replay_buffer = deque()

        
 
    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([3,n_neural])
        b1 = self.bias_variable([n_neural])
        W2 = self.weight_variable([n_neural,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,3])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2
 
    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
 
    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
          self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
          self.train_Q_network()

    def small_perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
 
    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,10)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
          done = minibatch[i][4]
          y_batch.append(reward_batch[i])
          #if done:
            #y_batch.append(reward_batch[i])
          #else :
            #y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
          self.y_input:y_batch,
          self.action_input:action_batch,
          self.state_input:state_batch
          })

    def small_train_Q_network(self,n):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,n)
        #minibatch = self.replay_buffer
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        with self.sess2.as_default():
            Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,n):
            y_batch.append(reward_batch[i])
        with self.sess2.as_default():
            self.optimizer.run(feed_dict={
              self.y_input:y_batch,
              self.action_input:action_batch,
              self.state_input:state_batch
              })
 
    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
          self.state_input:[state]
          })[0]
        if random.random() <= self.epsilon:
          return random.randint(0,self.action_dim - 1)
        else:
          return np.argmax(Q_value)
 
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
 
    def action(self,state):
        with self.sess2.as_default():
            Q_value = self.Q_value.eval(feed_dict = {
              self.state_input:[state]
              })[0]
        print('Q_value')
        print(Q_value)
        return np.argmax(Q_value)
 
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)
 
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
    def advising_action(self,state,possible_action):
        with self.sess2.as_default():
            Q_value = self.Q_value.eval(feed_dict = {
              self.state_input:[state]
              })[0]
        if len(possible_action) == 1:
            action = possible_action[0]
        else:
            if abs(possible_action[0] - possible_action[0]) < 20:
                action = np.argmax(Q_value)
            else:
                action = False
        return action
    def cal_similarity(self,state1,state2):
        if state1[0] == state2[0]:
            dataA = np.mat(state1[1:])
            dataB = np.mat(state2[1:])
            similarity = 1.0/(1.0 + np.linalg.norm(dataA - dataB)) 
        else:
            similarity = 0
        return similarity

    def give_advise(self,state,file_name1,file_name2):

        similar_threshold = 0.9
        f111 = open(file_name1, "r")
        lines = f111.readlines()

        knowledge_n = 0
        last_state = -1
        last_action = -1
        last_reward = -1

        # for line in lines:
        #     content = line.split()
        #     key = int(content[0])
        #     action = int(content[1])
        #     reward = int(content[2])
        #     if calculate_sim(last_state,state) > 0.95 and last_state!= key and last_action != action:
        #         knowledge_n += 1
        #         self.small_perceive(turn_list(last_state),last_action,last_reward,turn_list(last_state),0)
        #     last_state = key
        #     last_action = action
        #     last_reward = reward


        for line in lines:
            content = line.split()
            key = int(content[0])
            action = int(content[1])
            reward = int(content[2])
            #if calculate_sim(key,state) > 0.95 :
            if sim(key,state) < 1 :
                knowledge_n += 1
                self.small_perceive(turn_list(key),action,reward,turn_list(key),0)
        f111.close()



        if knowledge_n < 30:
          return [0,0,0,0]
        else:
          self.small_train_Q_network(knowledge_n)
          with self.sess2.as_default():
            Q_value = self.Q_value.eval(feed_dict = {
              self.state_input:[turn_list(state)]
              })[0]
          print('is_using')
          print(knowledge_n)
          return Q_value



def turn_list(key):
  state = str(key)
  #state_list = [int(state[0]), int(state[1]), int(state[2]), int(state[3])]
  state_list = [int(state[0]), int(state[1]), int(state[2])]
  return np.array(state_list)




def can_shoot(goal_dist, goal_angle):
  """Returns True if if player may have a good shot at the goal"""
  return bool((goal_dist < params['SHT_DST']) and (goal_angle > params['SHT_ANG']))

def has_better_pos(dist_to_op, goal_angle, pass_angle, curr_goal_angle):
  """Returns True if teammate is in a better attacking position"""
  
  if curr_goal_angle > goal_angle:
    b1 = 2
  else:
    b1 = 1
  
    
  if dist_to_op < params['DRIB_DST']:
    b2 = 2
  else:
    b2 = 1
    
  
  if pass_angle < params['PASS_ANG']:
    b3 = 2
  else:
    b3 = 1

  if b1 == 1:
    if b2 == 1:
      if b3 == 1:
        points = 1
      else:
        points = 2
    else:
      if b3 == 1:
        points = 3
      else:
        points = 4
  else:
    if b2 == 1:
      if b3 == 1:
        points = 5
      else:
        points = 6
    else:
      if b3 == 1:
        points = 7
      else:
        points = 8
    
    
  
  return points

def change_dist(flo):
  if flo > 0.1367:
    if flo > 0.7189:
      domain = 9
    elif flo > 0.4378:
      domain = 8
    else:
      domain = 7
  else:
    if flo > -0.0527:
      domain = 6
    elif flo > -0.2421:
      domain = 5
    elif flo > -0.4315:
      domain = 4
    elif flo > -0.6209:
      domain = 3
    elif flo > -0.8103:
      domain = 2
    else:
      domain = 1
  return domain

def change(flo):
  if flo > -0.5:
    if flo > 0.5:
      domain = 9
    elif flo > 0:
      domain = 8
    else:
      domain = 7
  else:
    if flo > -0.5833:
      domain = 6
    elif flo > -0.6666:
      domain = 5
    elif flo > -0.75:
      domain = 4
    elif flo > -0.8333:
      domain = 3
    elif flo > -0.9165:
      domain = 2
    else:
      domain = 1
  return domain


def change_dribble(flo):
  if flo > 0:
    if flo > 0.75:
      domain = 9
    elif flo > 0.5:
      domain = 8
    elif flo > 0.25:
      domain = 7
    else:
      domain = 6
  else:
    if flo > -0.25:
      domain = 5
    elif flo > -0.5:
      domain = 4
    elif flo > -0.75:
      domain = 3
    elif flo > -0.999052871962:
      domain = 2
    else:
      domain = 1
  return domain

def normalise(distribution):
  c0 = 0.5
  delta = 0.001
  d = min(distribution)
  if d < delta:
      rho = (c0 - delta)/(c0-d)
      for k in range(3):
          distribution[k] = c0 - rho * (c0 - distribution[k])
  pi = sum(distribution)
  for k in range(3):
      distribution[k] = distribution[k]/pi
  return distribution

def calculate_sim(key1,key2):
  sim = 0
  an = abs(int(key1) - int(key2))
  an = str(an)
  for i in an:
      sim +=int(i)
      
  return round((1-(sim/27)),4)

def sim(key1,key2):
  dataA = turn_list(key1)
  dataB = turn_list(key2)
  d1 = abs(dataA[0]-dataB[0])
  d2 = abs(dataA[1]-dataB[1])
  d3 = abs(dataA[2]-dataB[2])
  return d1+d2+d3


def np_sim(key1,key2):
  dataA = turn_list(key1)
  dataB = turn_list(key2)
  similarity = 1.0/(1.0 + np.linalg.norm(dataA - dataB)) 
  return similarity

def algorithm3(sim,cur_Q):
  global t
  difference = int((1-sim)*14)
  c = 0
  Q_value = cur_Q
  while c < difference:
    c = c+1
##    Q_value = math.exp((1*Q_value)/(2*10*math.log(t)))
    Q_value = math.exp((10*Q_value)/(2*10*5))
  return Q_value


def algorithm4(count,sim,Q,r,cur_Q):
  if sim == 1:
    weight = math.log(count)
  else:
    weight = sim*(1-(1/count))
  update_Q_sum = cur_Q + weight*((Q-r)**2)
  return update_Q_sum

def get_key(state):
  global teammate_uniform_number
  
  
  goal_dist = float(state[6])
  goal_op_angle = float(state[8])
  to_op = float(state[9])

  goal_dist_domain = change_dist(goal_dist)
  goal_op_angle_domain = change(goal_op_angle)
  to_op_domain = change_dribble(to_op)

  pass_angle1 = float(state[14])
  pass_angle2 = float(state[15])
  if pass_angle1 > pass_angle2:
    teammates_position = has_better_pos(float(state[12]),float(state[10]),pass_angle1,goal_op_angle)
    teammate_uniform_number=state[18]
    
    
  else:
    teammates_position = has_better_pos(float(state[12]),float(state[10]),pass_angle2,goal_op_angle)
    teammate_uniform_number=state[21]
  
  an = goal_dist_domain*100 + goal_op_angle_domain*10 + to_op_domain*1
  return an
  
def our(state,file_name1,file_name2):

    similar_threshold = 0.9
    f111 = open(file_name1, "r")
    lines = f111.readlines()

    knowledge_n = 0
    action_list = [0,0,0]
    Q_value = [0,0,0]
    reward_list = [0,0,0]

    for line in lines:
        content = line.split()
        key1 = int(content[0])
        action1 = int(content[1])
        reward1 = int(content[2])
        #if sim(key1,state) < 2 :
        if key1 == state :
            action_list[action1] += 1
            reward_list[action1] += reward1
            knowledge_n += 1
    f111.close()

    if knowledge_n < 20:
      return [0,0,0,0]
    else:
      for i in range(3):
        if action_list[i] != 0:
          Q_value[i] = reward_list[i]/action_list[i]
        else:
          Q_value[i] = 0
      return Q_value



  
def get_action(state,hfo_env,num_teammates,rand_pass):
  """Decides and performs the action to be taken by the agent."""
  global Q_table
  global pi_table
  global reward_table

  global key
  global action
  global teammate_uniform_number

  global unum
  
  key = state

  random_n = random.random()
  if key in pi_table:
    

    if random_n > pi_table[key][0]:
      if random_n > (pi_table[key][0]+pi_table[key][1]):
        action = 2
        last_chiqiu = str(unum) + ',' +str(key) + ',' +str(action)
        f=open('chiqiu.txt',"w+")
        f.write(last_chiqiu)
        f.close()
        hfo_env.act(hfo.DRIBBLE)
        return
      else:
        action = 1
        last_chiqiu = str(unum) + ',' +str(key) + ',' +str(action)
        f=open('chiqiu.txt',"w+")
        f.write(last_chiqiu)
        f.close()
        hfo_env.act(hfo.SHOOT)
        return
    else:
      action = 0
      last_chiqiu = str(unum) + ',' +str(key) + ',' +str(action)
      f=open('chiqiu.txt',"w+")
      f.write(last_chiqiu)
      f.close()
      hfo_env.act(hfo.PASS, teammate_uniform_number)
      return
  else:
    if random_n > 1/3:
      if random_n > 2/3:
        action = 2
        last_chiqiu = str(unum) + ',' +str(key) + ',' +str(action)
        f=open('chiqiu.txt',"w+")
        f.write(last_chiqiu)
        f.close()
        hfo_env.act(hfo.DRIBBLE)
        return
      else:
        action = 1
        last_chiqiu = str(unum) + ',' +str(key) + ',' +str(action)
        f=open('chiqiu.txt',"w+")
        f.write(last_chiqiu)
        f.close()
        hfo_env.act(hfo.SHOOT)
        return
    else:
      action = 0
      last_chiqiu = str(unum) + ',' +str(key) + ',' +str(action)
      f=open('chiqiu.txt',"w+")
      f.write(last_chiqiu)
      f.close()
      hfo_env.act(hfo.PASS, teammate_uniform_number)
      return

def teacher_action(state,action1,hfo_env,num_teammates):

  """Decides and performs the action to be taken by the agent."""
  global Q_table
  global pi_table
  global reward_table

  global key
  global action
  global teammate_uniform_number

  global unum

  last_chiqiu = str(unum) + ',' +str(state) + ',' +str(action1)
  f=open('chiqiu.txt',"w+")
  f.write(last_chiqiu)
  f.close()
  if action1 == 0:
    hfo_env.act(hfo.DRIBBLE)
    return
  elif action1 == 1:
    hfo_env.act(hfo.SHOOT)
    return
  else:
    hfo_env.act(hfo.PASS, teammate_uniform_number)
    return



      

    

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000, help="Server port")
  parser.add_argument('--seed', type=int, default=None,
                      help="Python randomization seed; uses python default if 0 or not given")
  parser.add_argument('--rand-pass', action="store_true",
                      help="Randomize order of checking teammates for a possible pass")
  parser.add_argument('--epsilon', type=float, default=0,
                      help="Probability of a random action if has the ball, to adjust difficulty")
  parser.add_argument('--record', action='store_true',
                      help="If doing HFO --record")
  parser.add_argument('--rdir', type=str, default='log/',
                      help="Set directory to use if doing --record")
  args=parser.parse_args()

  global Q_table
  global pi_table
  global reward_table

  global key
  global action
  global unum
  global t

  count_key = {}
  threshold = 1

  
  
  if args.seed:
    random.seed(args.seed)
  hfo_env = hfo.HFOEnvironment()
  if args.record:
    hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
                            'bin/teams/base/config/formations-dt', args.port,
                            'localhost', 'base_left', False,
                            record_dir=args.rdir)
  else:
    hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
                            'bin/teams/base/config/formations-dt', args.port,
                            'localhost', 'base_left', False)
  num_teammates = hfo_env.getNumTeammates()
  #num_opponents = hfo_env.getNumOpponents()
  if args.seed:
    if (args.rand_pass and (num_teammates > 1)) or (args.epsilon > 0):
      print("Python randomization seed: {0:d}".format(args.seed))
    else:
      print("Python randomization seed useless without --rand-pass w/2+ teammates or --epsilon >0")
  if args.rand_pass and (num_teammates > 1):
    print("Randomizing order of checking for a pass")
  if args.epsilon > 0:
    print("Using epsilon {0:n}".format(args.epsilon))

  unum = int(hfo_env.getUnum())
  if unum == 7:
    t1_num = 8
    t2_num = 11
  elif unum == 8:
    t1_num = 7
    t2_num = 11
  else:
    t1_num = 7
    t2_num = 8
  
    
  for episode in itertools.count():
    num_eps = 0
    num_had_ball = 0
    num_move = 0
    status = hfo.IN_GAME
    zeta = 0.1
    all_action = 0
    all_comm = 0
    all_useful_comm = 0
    
    f=open('chiqiu.txt',"w+")
    f.write('5')
    f.close()
    
    while status == hfo.IN_GAME:
      t = t + 1
      
      state = hfo_env.getState()
      #print(state)
      if int(state[5]) == 1: # state[5] is 1 when player has the ball
        with open('chiqiu.txt', "r") as f1:
          xinxi = f1.read()
        f1.close()
        if len(xinxi)>3:
##          t = t + 1
          all_last = xinxi.split(',')
          last_unum = int(all_last[0])
          last_key = int(all_last[1])
          if len(all_last[2]) == 1:
            last_action = int(all_last[2])
          else:
            last_action = int(all_last[2][-1])

          
          if last_key not in pi_table:
            pi_table[last_key] = [1/3,1/3,1/3] 
            reward_table[last_key] = [-2,-2,-2]
            Q_table[last_key] = [0,0,0]
            count_key[last_key] = 0
          count_key[last_key] +=1

          if last_action == 0:
            reward = 3
          else:
            reward = 1
          if unum == last_unum:
            if key not in Q_table:
              max_utility = 0
            else:
              max_utility = max(Q_table[key])
            curr_utility = Q_table[last_key][last_action]

            Q_table[last_key][last_action] = (1-alpha)*curr_utility + alpha*(reward + gamma*max_utility)
            total_reward = 0
            for i in range(3):            
              total_reward += pi_table[last_key][i]*Q_table[last_key][i]
            
            for j in range(3):
              pi_table[last_key][j] = pi_table[last_key][j] + zeta*(Q_table[last_key][j] - total_reward)
            distr = normalise([pi_table[last_key][0],pi_table[last_key][1],pi_table[last_key][2]])
            pi_table[last_key][0] = distr[0]
            pi_table[last_key][1] = distr[1]
            pi_table[last_key][2] = distr[2]
            reward_table[last_key][last_action] = reward
          #store nomarlised information in unum7811.txt file
          file_name = 'unum'+str(last_unum)+'.txt'
          content = str(last_key) + '\t' + str(last_action) + '\t' + str(reward) + '\n'
          f78112=open(file_name,"a")
          f78112.write(content)
          f78112.close()

        #do alogorithm3 and 4
        key = get_key(state)

        if key not in Q_table:
          pi_table[key] = [1/3,1/3,1/3] 
          reward_table[key] = [-2,-2,-2]
          Q_table[key] = [0,0,0]
          count_key[key] = 0
        #how many times does thi observation happen
        count_key[key] +=1

        file_name1 = 'unum'+str(t1_num)+'.txt'
        file_name2 = 'unum'+str(t2_num)+'.txt'
        #if max(pi_table[key]) > 0.5 :
        if np.var(pi_table[key]) > 0.035:
          decision = random.random()
          if decision < threshold:
            get_action(key,hfo_env,num_teammates,args.rand_pass)
            #threshold = threshold -0.00125
          else:
            action1 = np.argmax(pi_table[key])
            teacher_action(key,action1,hfo_env,num_teammates)

        else:
          dqn.chongzhi()
          new_filename = 'unum'+str(unum)+'.txt'
          three_Q = dqn.give_advise(key,new_filename,file_name2)
          #three_Q = dqn.give_advise(key,file_name1,file_name2)
          #three_Q = our(key,file_name1,file_name2)
          all_comm += 1
          if len(three_Q) == 4:
            dqn.chongzhi()
            three_Q = dqn.give_advise(key,file_name1,file_name2)
            if len(three_Q) == 4:
              get_action(key,hfo_env,num_teammates,args.rand_pass)
            else:
              action1 = np.argmax(three_Q)
              teacher_action(key,action1,hfo_env,num_teammates)
              all_useful_comm += 1 
          else:
            action1 = np.argmax(three_Q)
            teacher_action(key,action1,hfo_env,num_teammates)
            all_useful_comm += 1     
        num_had_ball += 1
        all_action += 1

        status=hfo_env.step()

      else:           
        hfo_env.act(hfo.MOVE)
        num_move += 1
        status=hfo_env.step()
        
      if status != hfo.IN_GAME:
        print('useful',all_useful_comm)
        print('comm',all_comm)
        print('all',all_action)
##        t = t + 1
        if status == hfo.GOAL:
          reward = 40
          max_utility = 40
        else:
          reward = -40
          max_utility = -40

        with open('chiqiu.txt', "r") as f2:
          xinxi = f2.read()
        f2.close()
        if len(xinxi)>3:
          all_last = xinxi.split(',')
          last_unum = int(all_last[0])
          last_key = int(all_last[1])
          if len(all_last[2]) == 1:
            last_action = int(all_last[2])
          else:
            last_action = int(all_last[2][-1])
          

          
          if last_key not in pi_table:
            pi_table[last_key] = [1/3,1/3,1/3]
            reward_table[last_key] = [-2,-2,-2]
            Q_table[last_key] = [0,0,0]
            count_key[last_key] = 0
          count_key[last_key] +=1
          
          if unum == last_unum:
            curr_utility = Q_table[last_key][last_action]

            Q_table[last_key][last_action] = (1-alpha)*curr_utility + alpha*(reward + gamma*max_utility)
            total_reward = 0
            for i in range(3):            
              total_reward += pi_table[last_key][i]*Q_table[last_key][i]
            
            for j in range(3):
              pi_table[last_key][j] = pi_table[last_key][j] + zeta*(Q_table[last_key][j] - total_reward)
            distr = normalise([pi_table[last_key][0],pi_table[last_key][1],pi_table[last_key][2]])
            pi_table[last_key][0] = distr[0]
            pi_table[last_key][1] = distr[1]
            pi_table[last_key][2] = distr[2]
            reward_table[last_key][last_action] = reward
          #store nomarlised information in unum7811.txt file
          file_name = 'unum'+str(last_unum)+'.txt'
          content = str(last_key) + '\t' + str(last_action) + '\t' + str(reward) + '\n'
          f78112=open(file_name,"a")
          f78112.write(content)
          f78112.close()

      
        

      

      
        
        
      #print(status)

    # Quit if the server goes down
    if status == hfo.SERVER_DOWN:
      hfo_env.act(hfo.QUIT)
      exit()

    # Check the outcome of the episode
    print("Episode {0:d} ended with {1:s}".format(episode,
                                                  hfo_env.statusToString(status)))
    if args.epsilon > 0:
      print("\tNum move: {0:d}; Random action: {1:d}; Nonrandom: {2:d}".format(num_move,
                                                                               num_eps,
                                                                               (num_had_ball-
                                                                                num_eps)))

if __name__ == '__main__':
  dqn = small_DQN()
  main()
