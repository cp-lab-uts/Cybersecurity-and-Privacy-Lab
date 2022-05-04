  
#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import tensorflow.compat.v1 as tf #启用版本1
import numpy as np
from collections import Counter
tf.disable_v2_behavior() #禁用tensorflow 版本2
from collections import deque


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error




# ==============================================================================
# -- DQN functions ----------------------------------------------------------
# ==============================================================================


# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 1000 # experience replay buffer size
BATCH_SIZE = 50 # size of minibatch


class DQN():
  # DQN Agent
  def __init__(self):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = 19  
    self.action_dim = 3
    self.ep = 0
 
    self.create_Q_network()
    self.create_training_method()
 
    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())
 
  def create_Q_network(self):
    # network weights
    W1 = self.weight_variable([self.state_dim,20])
    b1 = self.bias_variable([20])
    W2 = self.weight_variable([20,self.action_dim])
    b2 = self.bias_variable([self.action_dim])
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])
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
    self.ep += 1
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()
 
    if self.ep % BATCH_SIZE == 0:
      self.train_Q_network()
 
  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]
 
    # Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
 
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
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])
 
  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    
    return tf.Variable(initial)
 
  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

# ==============================================================================
# -- created functions ----------------------------------------------------------
# ==============================================================================


line1_2 = 0
line1_4 = 0
line2_3 = 0
line2_5 = 0
line3_6 = 0
line4_7 = 0
line4_9 = 0
line5_8 = 0
line5_10 = 0
line6_11 = 0
line7_8 = 0
line7_10 = 0
line8_11 = 0
line9_12 = 0
line10_13 = 0
line11_f = 0
line12_13 = 0
line13_f = 0


line1_2_count = 0
line1_4_count = 0
line2_3_count = 0
line2_5_count = 0
line3_6_count = 0
line4_7_count = 0
line4_9_count = 0
line5_8_count = 0
line5_10_count = 0
line6_11_count = 0
line7_8_count = 0
line7_10_count = 0
line8_11_count = 0
line9_12_count = 0
line10_13_count = 0
line11_f_count = 0
line12_13_count = 0
line13_f_count = 0



def get_reward(last_line,current_line,last_time,current_time):

    global line1_2
    global line1_4
    global line2_3
    global line2_5
    global line3_6
    global line4_7
    global line4_9
    global line5_8
    global line5_10
    global line6_11
    global line7_8
    global line7_10
    global line8_11
    global line9_12
    global line10_13
    global line11_f
    global line12_13
    global line13_f


    global line1_2_count
    global line1_4_count
    global line2_3_count
    global line2_5_count
    global line3_6_count
    global line4_7_count
    global line4_9_count
    global line5_8_count
    global line5_10_count
    global line6_11_count
    global line7_8_count
    global line7_10_count
    global line8_11_count
    global line9_12_count
    global line10_13_count
    global line11_f_count
    global line12_13_count
    global line13_f_count


    time = int(current_time) - int(last_time)
    if last_line == 1:
        if current_line == 2:
            all_time = line1_2
            count = line1_2_count
            line1_2_count += 1
            line1_2 += time
        else:
            all_time = line1_4
            count = line1_4_count
            line1_4_count += 1
            line1_4 += time      
    elif last_line == 2:
        if current_line == 3:
            all_time = line2_3
            count = line2_3_count
            line2_3_count += 1
            line2_3 += time
        else:
            all_time = line2_5
            count = line2_5_count
            line2_5_count += 1
            line2_5 += time
    elif last_line == 3:
        all_time = line3_6
        count = line3_6_count
        line3_6_count += 1
        line3_6 += time
    elif last_line == 4:
        if current_line == 7:
            all_time = line4_7
            count = line4_7_count
            line4_7_count += 1
            line4_7 += time
        else:
            all_time = line4_9
            count = line4_9_count
            line4_9_count += 1
            line4_9 += time

    elif last_line == 5:
        if current_line == 8:
            all_time = line5_8
            count = line5_8_count
            line5_8_count += 1
            line5_8 += time
        else:
            all_time = line5_10
            count = line5_10_count
            line5_10_count += 1
            line5_10 += time
    elif last_line == 6:
        all_time = line6_11
        count = line6_11_count
        line6_11_count += 1
        line6_11 += time

    elif last_line == 7:
        if current_line == 8:
            all_time = line7_8
            count = line7_8_count
            line7_8_count += 1
            line7_8 += time
        else:
            all_time = line7_10
            count = line7_10_count
            line7_10_count += 1
            line7_10 += time
    elif last_line == 8:
        all_time = line8_11
        count = line8_11_count
        line8_11_count += 1
        line8_11 += time
    elif last_line == 9:
        all_time = line9_12
        count = line9_12_count
        line9_12_count += 1
        line9_12 += time
    elif last_line == 10:
        all_time = line10_13
        count = line10_13_count
        line10_13_count += 1
        line10_13 += time
    elif last_line == 11:
        all_time = line11_f
        count = line11_f_count
        line11_f_count += 1
        line11_f += time
    elif last_line == 12:
        all_time = line12_13
        count = line12_13_count
        line12_13_count += 1
        line12_13 += time
    elif last_line == 13:
        all_time = line13_f
        count = line13_f_count
        line13_f_count += 1
        line13_f += time

    if count == 0:
        return 0
    else:
        if time > (all_time/count):
            return (time/(all_time/count)-1)*(-20)

        else:
            return (1-time/(all_time/count))*30




def get_state(all_v):
    state = []
    d1_center = 0
    d1_1 = 0
    d1_2 = 0
    d1_3 = 0
    d1_4 = 0

    d2_center = 0
    d2_1 = 0
    d2_2 = 0
    d2_3 = 0
    d2_4 = 0

    d3_center = 0
    d3_1 = 0
    d3_2 = 0
    d3_3 = 0
    d3_4 = 0

    d4_center = 0
    d4_1 = 0
    d4_2 = 0
    d4_3 = 0
    d4_4 = 0

    d5_center = 0
    d5_1 = 0
    d5_2 = 0
    d5_3 = 0
    d5_4 = 0

    d6_center = 0
    d6_1 = 0
    d6_2 = 0
    d6_3 = 0
    d6_4 = 0

    d7_center = 0
    d7_1 = 0
    d7_2 = 0
    d7_3 = 0
    d7_4 = 0

    d8_center = 0
    d8_1 = 0
    d8_2 = 0
    d8_3 = 0
    d8_4 = 0

    d9_center = 0
    d9_1 = 0
    d9_2 = 0
    d9_3 = 0
    d9_4 = 0

    for v in all_v:
        if -179<v.x<80 and -178<v.y<140 :
            #node 1
            if -179<v.x<-140 and -90<v.y<-78:
                d1_1 += 1
            elif -126.7<v.x<-113.5 and -78<v.y<-10:
                d1_2 += 1
            elif -113.5<v.x<-61.5 and -101<v.y<-89.7:
                d1_3 += 1
            elif -140<v.x<-126.7 and -128.5<v.y<-101:
                d1_4 += 1


            #node 2
            elif -61.5<v.x<-39 and -101<v.y<-78.5:
                d2_center += 1
            elif -110<v.x<-61.5 and -89.7<v.y<-78.5:
                d2_1 += 1
            elif -49.5<v.x<-39 and -78.5<v.y<-10:
                d2_2 += 1
            elif -39<v.x<20.5 and -101<v.y<-89.7:
                d2_3 += 1
            elif -61.5<v.x<-49.5 and -140<v.y<-101:
                d2_4 += 1


            #node 3
            elif 20.5<v.x<42 and -101<v.y<-78.5:
                d3_center += 1
            elif -39<v.x<20.5 and -89.7<v.y<-78.5:
                d3_1 += 1
            elif 30.5<v.x<41 and -78.5<v.y<-10:
                d3_2 += 1
            elif 42<v.x<72 and -101<v.y<-89.7:
                d3_3 += 1
            elif 20.5<v.x<30.5 and -160<v.y<-101:
                d3_4 += 1

            #node 4
            if -179<v.x<-140 and 1<v.y<12.5:
                d4_1 += 1
            elif -126<v.x<-113 and 12<v.y<78:
                d4_2 += 1
            elif -113<v.x<-60 and -10<v.y<0.5:
                d4_3 += 1
            elif -140<v.x<-126.5 and -78<v.y<-10:
                d4_4 += 1

            #node 5    
            elif -60<v.x<-39 and -10<v.y<12:
                d5_center += 1
            elif -110<v.x<-60 and 0.5<v.y<12:
                d5_1 += 1
            elif -49.5<v.x<-39 and 12<v.y<78:
                d5_2 += 1
            elif -39<v.x<19 and -10<v.y<0.5:
                d5_3 += 1
            elif -61.5<v.x<-49.5 and -78.5<v.y<-10:
                d5_4 += 1

            #node 6
            elif 19<v.x<40 and -10<v.y<12:
                d6_center += 1
            elif -39<v.x<19 and 0.5<v.y<12:
                d6_1 += 1
            elif 29.8<v.x<40 and 12<v.y<78:
                d6_2 += 1
            elif 40<v.x<80 and -10<v.y<0.5:
                d6_3 += 1
            elif 20<v.x<30.5 and -78.5<v.y<-10:
                d6_4 += 1

            #node 7
            if -179<v.x<-140 and 90<v.y<101:
                d7_1 += 1
            elif -125.4<v.x<-111 and 101<v.y<138:
                d7_2 += 1
            elif -111<v.x<-62 and 78<v.y<90:
                d7_3 += 1
            elif -140<v.x<-126 and 12<v.y<78:
                d7_4 += 1

            #node 8
            elif -62<v.x<-39 and 78<v.y<101:
                d8_center += 1
            elif -110<v.x<-62 and 90<v.y<101:
                d8_1 += 1
            elif -49.5<v.x<-39 and 101<v.y<141:
                d8_2 += 1
            elif -39<v.x<20 and 78<v.y<90:
                d8_3 += 1
            elif -62<v.x<-49.5 and 12<v.y<78:
                d8_4 += 1
            #node 9
            elif 19<v.x<40 and 78<v.y<101:
                d9_center += 1
            elif -39<v.x<20 and 90<v.y<101:
                d9_1 += 1
            elif 29.8<v.x<40 and 101<v.y<141:
                d9_2 += 1
            elif 40<v.x<70 and 78<v.y<90:
                d9_3 += 1
            elif 19<v.x<29.8 and 11<v.y<78:
                d9_4 += 1


    d1_x = d1_1 + d1_3
    d1_y = d1_2 + d1_4

    d2_x = d2_1 + d2_3
    d2_y = d2_2 + d2_4

    d3_x = d3_1 + d3_3
    d3_y = d3_2 + d3_4

    d4_x = d4_1 + d4_3
    d4_y = d4_2 + d4_4

    d5_x = d5_1 + d5_3
    d5_y = d5_2 + d5_4

    d6_x = d6_1 + d6_3
    d6_y = d6_2 + d6_4

    d7_x = d7_1 + d7_3
    d7_y = d7_2 + d7_4

    d8_x = d8_1 + d8_3
    d8_y = d8_2 + d8_4

    d9_x = d9_1 + d9_3
    d9_y = d9_2 + d9_4

    state = [d1_x, d1_y, d2_x, d2_y, d3_x, d3_y, d4_x, d4_y, d5_x, d5_y, d6_x, d6_y, d7_x, d7_y, d8_x, d8_y, d9_x, d9_y]
    return state
            


def which_line(location):
    #which node the car is
    line = 0
    x = location.x
    y= location.y
    if x<-140:
        line = 1
    elif -113.5<x<-61.5 and -101<y<-78.5:
        line = 2
    elif -39<x<20.5 and -101<y<-78.5:
        line = 3
    elif -140<x<-113.5 and -78.5<y<-10:
        line = 4
    elif -60<x<-39 and -78.5<y<-10:
        line = 5
    elif 19<x<41 and -78.5<y<-10:
        line = 6
    elif -113.5<x<-60 and -10<y<12:
        line = 7

    elif -39<x<20.5 and -10<y<12:
        line = 8
    elif -140<x<-113.5 and 12<y<78:
        line = 9
    elif -60<x<-39 and 12<y<78:
        line = 10
    elif 19<x<41 and 12<y<78:
        line = 11
    elif -113.5<x<-62 and 78<y<101:
        line = 12
    elif -39<x<20.5 and 78<y<101:
        line = 13
    elif x>42:
        line = 14
    return line



def can_do_action(location,act):
    #if the car can do this action return true
    # 1 is Turn left 
    # 2 is straight
    # 3 is Turn right
    action = act + 1
    current_line = location

    if current_line == 1:
        if action == 1:
            return False
        else:
            return True

    elif current_line == 2:
        if action == 1:
            return False
        else:
            return True
    elif current_line == 3:
        if action == 3:
            return True
        else:
            return False

    elif current_line == 4:
        if action == 3:
            return False
        else:
            return True

    elif current_line == 5:
        if action == 3:
            return False
        else:
            return True
    elif current_line == 6:
        if action == 2:
            return True
        else:
            return False
    elif current_line == 7:
        if action == 1:
            return False
        else:
            return True

    elif current_line == 8:
        if action == 3:
            return True
        else:
            return False
    elif current_line == 9:
        if action == 1:
            return True
        else:
            return False
    elif current_line == 10:
        if action == 1:
            return True
        else:
            return False
    elif current_line == 11:
        if action == 1:
            return True
        else:
            return False
    elif current_line == 12:
        if action == 2:
            return True
        else:
            return False
    elif current_line == 13:
        if action == 2:
            return True
        else:
            return False



def action_to_destination(location,act):
    #turn the action to the new destination
    action = act + 1
    point_2 = (22,-85,0)
    point_3 = (-55,-5,0)
    point_4 = (25,5,0)
    point_5 = (-53,83,0)
    point_6 = (25,95,0)
    final = (58,95.6,0)
    current_line = location
    if current_line ==1:
        if action == 2:         
            return (-56,-85,0)
        else:
            return (-132,-4,0)

    elif current_line == 2:
        if action == 2:
            return point_2
        else:
            return  point_3

    elif current_line == 3:
        return point_4

    elif current_line == 4:
        if action == 1:
            return (-55,5.5,0)
        else:
            return (-131,85,0)

    elif current_line == 5:
        if action == 1:
            return point_4
        else:
            return point_5

    elif current_line == 6:
        return (25,82,0)

    elif current_line == 7:
        if action == 2:
            return (24,6,0)
        else:
            return point_5

    elif current_line == 8:
        return (25,82,0)

    elif current_line == 9:
        return (-57,95,0)

    elif current_line == 10:
        return point_6

    elif current_line == 11:
        return final

    elif current_line == 12:
        return point_6
        
    elif current_line == 13:
        return final

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name



# other angents 











# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = 'model3'
        self._gamma = args.gamma
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0


    def other(self):
    
        host='127.0.0.1'
        port = 2000
        number_of_vehicles = 240
        filterv = 'vehicle.*'
        number_of_walkers = 40
        filterw = 'walker.pedestrian.*'
        tm_port = 8000
        sync = False
        hybrid = False
        safe = True


        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        vehicles_list = []
        walkers_list = []
        all_id = []
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        synchronous_master = False

        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        if hybrid:
            traffic_manager.set_hybrid_physics_mode(True)

        if sync:
            settings = self.world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = self.world.get_blueprint_library().filter(filterv)
        blueprintsWalkers = self.world.get_blueprint_library().filter(filterw)

        if safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not sync or not synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)
        self.vehicles_list = vehicles_list
        self.walkers_list = walkers_list
        self.all_actors = all_actors
        self.all_id = all_id



    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        print("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            
            # spawn_point = carla.Transform(carla.Location(x=1, y=2, z=3), carla.Rotation())
            # self.player = world.try_spawn_actor(blueprint, spawn_point)
            #starting points
            spawn_point = carla.Transform(carla.Location(x=-167.5, y=-84.7, z=1), carla.Rotation())
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        self.other()
        

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        client = carla.Client('127.0.0.1',2000)

        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """ Main loop for agent"""

    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21
    destination_point = (-135,-84.7,0)
    if_final = False

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        if args.agent == "Roaming":
            agent = RoamingAgent(world.player)
        elif args.agent == "Basic":
            agent = BasicAgent(world.player)
            # spawn_point = world.map.get_spawn_points()[0]
            # spawn_point = (233.4,-20,0)
            agent.set_destination(destination_point)
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)

            spawn_points = world.map.get_spawn_points()
            random.shuffle(spawn_points)

            if spawn_points[0].location != agent.vehicle.get_location():
                destination = spawn_points[0].location
            else:
                destination = spawn_points[1].location

            agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        clock = pygame.time.Clock()
        start_time = hud.simulation_time
        
        last_time = hud.simulation_time
        
        
        if_run = True
        while if_run:
            clock.tick_busy_loop(60)
            if controller.parse_events():
                return

            # As soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue

            if args.agent == "Roaming" or args.agent == "Basic":

                current_time = hud.simulation_time
                core_time = int(current_time) - int(start_time)

                if core_time>500:
                    if_run = False

                if (agent._vehicle.get_location()).x >50:

                    if_run = False
                    done = 1
                    
                    current_line = which_line(agent._vehicle.get_location())

                    vehicles = world.world.get_actors().filter('vehicle.*')
                    vehicles = [x.get_location() for x in vehicles if x.id != world.player.id]
                    part_state = np.array(get_state(vehicles))
                    state = np.append([current_line],part_state)
                    current_time = hud.simulation_time
                    reward = get_reward(last_line,current_line,last_time,current_time)
                    f = open("exp3_lap4_test2.txt", "a")
                    score_time = int(current_time) - int(start_time)
                    if c<200:
                        f.write("\n")
                        f.write("test")
                        f.write(str(score_time))
                    else:
                        f.write("\n")
                        f.write("train")
                        f.write(str(score_time))

                    f.close()

                    
                    content = str(last_state) + '\t' + str(action) + '\t' + str(reward) + '\t' + str(state) + '\t' + str(last_line) + '\t' + str(current_line) + '\t' + str(int(current_time)-int(last_time)) + '\t' + str(done) + '\n'
                    f2 = open("exp3_lap4_state2.txt", "a")
                    f2.write(content)
                    f2.close()

                if controller.parse_events():
                    return

                # as soon as the server is ready continue!
                world.world.wait_for_tick(10.0)


                distance_to_destination = math.sqrt((destination_point[0] - (agent._vehicle.get_location()).x)**2 + (destination_point[1] - (agent._vehicle.get_location()).y)** 2 + (destination_point[2] - (agent._vehicle.get_location()).z)**2)

                if distance_to_destination<40 and not if_final:

                    current_line = which_line(agent._vehicle.get_location())
                    print(current_line)
                    

                    vehicles = world.world.get_actors().filter('vehicle.*')
                    vehicles = [x.get_location() for x in vehicles if x.id != world.player.id]
                    part_state = [current_line] + get_state(vehicles)
                    state = np.array(exp_noise(part_state))
                    
                    if current_line != 1:
                        current_time = hud.simulation_time
                        reward = get_reward(last_line,current_line,last_time,current_time)
                        print(int(current_time)-int(last_time))
                        done = 0
                        dqn.perceive(last_state,action,reward,state,done)
                        
                        
                        content = str(last_state) + '\t' + str(action) + '\t' + str(reward) + '\t' + str(state) + '\t' + str(last_line) + '\t' + str(current_line) + '\t' + str(int(current_time)-int(last_time)) + '\t' + str(done) + '\n'
                        f1 = open("exp3_lap4_state2.txt", "a")
                        f1.write(content)
                        f1.close()

                    
                    action = dqn.action(state)
                    

                    while not can_do_action(current_line,action):
                        print('action')
                        print(action)
                        reward = -40
                        done = 1
                        dqn.perceive(state,action,reward,state,done)
                        #action = dqn.action(state)
                        action = random.choice([0,1,2])

                
                    destination_point = action_to_destination(current_line,action)
                    agent.set_destination(destination_point)
                    print(destination_point)
                    if destination_point == (58,95.6,0):
                        if_final = True
                    
                    last_state = state
                    last_line = current_line
                    last_time = hud.simulation_time





                
                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)


            else:
                agent.update_information(world)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()

                # Set new destination when target has been reached
                if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints and args.loop:
                    agent.reroute(spawn_points)
                    tot_target_reached += 1
                    world.hud.notification("The target has been reached " +
                                           str(tot_target_reached) + " times.", seconds=4.0)

                elif len(agent.get_local_planner().waypoints_queue) == 0 and not args.loop:
                    print("Target reached, mission accomplished...")
                    break

                speed_limit = world.player.get_speed_limit()
                agent.get_local_planner().set_speed(speed_limit)

                control = agent.run_step()
                world.player.apply_control(control)


    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')



#noise
t_l = [1,2,3,4,5,6,7,8,9]
location = {1:(0,2),2:(1,2),3:(2,2),4:(0,1),5:(1,1),6:(2,1),7:(0,0),8:(1,0),9:(2,0)}

distribution = {}
exp_epsilon = 3
exp_alpha = 1.0

fake_location = []
privacy_location = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

def distance(l1,l2):
    d = np.sqrt(np.square(abs(l1[0] - l2[0])) + np.square(abs(l1[1] - l2[1])))
    return d

def exponential(li,lj):
    p = math.exp((exp_epsilon * (-1) * distance(li,lj))/(2 * exp_alpha))
    return p

def normalise(distribution_):
    c0 = 0.5
    delta = 0.001
    d = min(distribution_.values())
    if d < delta:
        rho = (c0 - delta)/(c0-d)
        for k in range(len(location.keys())):
            distribution_[list(location.keys())[k]] = c0 - rho * (c0 - distribution_[list(location.keys())[k]])
    pi = sum(distribution_.values())
    for k in range(len(location.keys())):
        distribution_[list(location.keys())[k]] = distribution_[list(location.keys())[k]]/pi
    return distribution_

#for k in range(len(location.keys())):
    #distribution[list(location.keys())[k]] = exponential(true_Location, location[list(location.keys())[k]])



def random_result(dis,tr):
    global privacy_location
    
    c = 0
    while c<10:
        p = np.array(list(dis.values()))
        fake = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], p = p.ravel())
        privacy_location[fake].append(tr)
        c += 1

distribution_1 = {}
distribution_2 = {}
distribution_3 = {}
distribution_4 = {}
distribution_5 = {}
distribution_6 = {}
distribution_7 = {}
distribution_8 = {}
distribution_9 = {}

for k in location.keys():
    distribution_1[k] = exponential(location[1], location[k])
distribution_1 = normalise(distribution_1)

#print(distribution_1)

for k in location.keys():
    distribution_2[k] = exponential(location[2], location[k])
distribution_2 = normalise(distribution_2)

#print(distribution_2)


for k in location.keys():
    distribution_3[k] = exponential(location[3], location[k])
distribution_3 = normalise(distribution_3)


for k in location.keys():
    distribution_4[k] = exponential(location[4], location[k])
distribution_4 = normalise(distribution_4)


for k in location.keys():
    distribution_5[k] = exponential(location[5], location[k])
distribution_5 = normalise(distribution_5)


for k in location.keys():
    distribution_6[k] = exponential(location[6], location[k])
distribution_6 = normalise(distribution_6)


for k in location.keys():
    distribution_7[k] = exponential(location[7], location[k])
distribution_7 = normalise(distribution_7)


for k in location.keys():
    distribution_8[k] = exponential(location[8], location[k])
distribution_8 = normalise(distribution_8)


for k in location.keys():
    distribution_9[k] = exponential(location[9], location[k])
distribution_9 = normalise(distribution_9)

def exp_noise(state):
    noise_state = []
    noise_state.append(state[0])
    global fake_location
    global privacy_location
    fake_location = []
    privacy_location = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

    random_result(distribution_1,1)

    random_result(distribution_2,2)

    random_result(distribution_3,3)

    random_result(distribution_4,4)

    random_result(distribution_5,5)

    random_result(distribution_6,6)

    random_result(distribution_7,7)

    random_result(distribution_8,8)

    random_result(distribution_9,9)
    #print(privacy_location)
    for i in privacy_location.keys():
        maxNum_sample = Counter(privacy_location[i]).most_common(1)
        
        fake_location.append(maxNum_sample[0][0])
    for j in fake_location:
        beta = 1/4
        noise1 = 0 
        noise2 = 0
        for i in range(10):
            noise1 += float(np.random.laplace(0,beta,1))
            noise2 += float(np.random.laplace(0,beta,1))
        lap_noise1 = round(noise1/10)

        lap_noise2 = round(noise2/10)
        noise_state1 = state[j*2-1] + lap_noise1
        if noise_state1<0:
            noise_state1 = 0
        noise_state2 = state[j*2] + lap_noise2
        if noise_state2<0:
            noise_state2 = 0


        noise_state.append(noise_state1)
        
        noise_state.append(noise_state2)
    return noise_state




if __name__ == '__main__':
    dqn = DQN()
    file_name1 = 'random_big_state.txt'
    f111 = open(file_name1, "r")
    lines = f111.readlines()
    last_cost=0





    for line in lines:
        line = line[1:]
        num = line.find(']')
        num2 = line.find('[')
        num3 = line.find(']',num2)
        
        part1 = line[0:num]
        part2 = line[num+1:num2]
        part3 = line[num2+1:num3]
        part4 = line[num3+1:]

        p1 = part1.split()
        state1 = []
        for i in p1:
            state1.append(int(i))
            
        next_state1 = []
        p3= part3.split()
        for j in p3:
            next_state1.append(int(j))
        p2 = part2.split()
        action1 = int(p2[0])
        reward1 = float(p2[1])
        p4 = part4.split()


        
        last_line1 = int(p4[0])
        current_line1 = int(p4[1])
        current_time1 = int(p4[2])
        exp_state1 = exp_noise(state1)
        last_state1 = np.array(exp_state1)
        exp_next_state1 = exp_noise(next_state1)
        current_state1 = np.array(exp_next_state1)

        reward1 = get_reward(last_line1,current_line1,0,current_time1)

        done1 = int(p4[3])
        if current_time1<70:
            dqn.perceive(last_state1,action1,reward1,current_state1,done1)
        
        

    f111.close()
    c=1
    while c < 5000:
        main()
        c = c+1
        if c >500:
            c = 1
