import numpy as np
import pyglet
import random
import math
import collections
import datetime

# write a txt file
# file = open('Ex1Random.txt','w')
currentDT = datetime.datetime.now()
filename = "our(" + currentDT.strftime("%H-%M-%S %Y-%m-%d") + ").txt"
file = open(filename,'w')

# window size
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# grid size （i.e 50 * 50）
HORIZONTAL_GRID_NUM = int(WINDOW_WIDTH/50)
VERTICAL_GRID_NUM = int(WINDOW_HEIGHT/50)
GRID_WIDTH = WINDOW_WIDTH / HORIZONTAL_GRID_NUM
GRID_HEIGHT = WINDOW_HEIGHT / VERTICAL_GRID_NUM

# block color & amount & positions
BLOCK_COLOR = (0, 0, 0)
BLOCK_NUM = 45
#BLOCK_POSITION = [(14.0, 8.0), (4.0, 13.0), (13.0, 12.0), (23.0, 1.0), (17.0, 5.0), (20.0, 14.0), (20.0, 12.0), (8.0, 12.0), (20.0, 10.0), (5.0, 9.0), (24.0, 14.0), (12.0, 7.0), (11.0, 1.0), (6.0, 6.0), (8.0, 9.0), (19.0, 1.0), (2.0, 1.0), (4.0, 7.0), (23.0, 13.0), (13.0, 10.0), (24.0, 8.0), (7.0, 3.0), (14.0, 1.0), (1.0, 8.0), (21.0, 13.0), (1.0, 15.0), (2.0, 2.0), (19.0, 15.0), (5.0, 6.0), (15.0, 11.0), (20.0, 3.0), (14.0, 16.0), (21.0, 9.0), (19.0, 10.0), (2.0, 7.0), (21.0, 11.0), (21.0, 14.0), (6.0, 3.0), (14.0, 15.0), (18.0, 10.0), (23.0, 12.0), (21.0, 2.0), (18.0, 8.0), (6.0, 14.0), (12.0, 14.0)]
BLOCK_POSITION = []
for i in range(BLOCK_NUM):
    LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
    LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    # avoid duplication
    if (LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in BLOCK_POSITION:
        while ((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in BLOCK_POSITION):
            LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
            LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    else:
        LEFT_BOT_X = LEFT_BOT_X
        LEFT_BOT_Y = LEFT_BOT_Y
    BLOCK_POSITION.append((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1))

# rubbish color & amount & dynamic positions
RUBBISH_COLOR = (222, 227, 255)
RUBBISH_NUM = 80

CLEAN_COLOR = (255,255,255)
#RUBBISH_POSITION = [(10.0, 2.0), (7.0, 2.0), (3.0, 9.0), (3.0, 16.0), (18.0, 11.0), (17.0, 1.0), (11.0, 7.0), (17.0, 13.0), (21.0, 1.0), (9.0, 8.0), (15.0, 1.0), (15.0, 7.0), (5.0, 8.0), (10.0, 13.0), (4.0, 12.0), (15.0, 12.0), (9.0, 1.0), (6.0, 2.0), (2.0, 6.0), (22.0, 5.0), (19.0, 4.0), (13.0, 14.0), (5.0, 1.0), (12.0, 9.0), (6.0, 1.0), (8.0, 6.0), (10.0, 16.0), (6.0, 11.0), (1.0, 11.0), (8.0, 15.0), (20.0, 11.0), (24.0, 4.0), (24.0, 5.0), (5.0, 4.0), (10.0, 14.0), (19.0, 8.0), (20.0, 15.0), (14.0, 10.0), (3.0, 5.0), (9.0, 12.0), (12.0, 13.0), (3.0, 7.0), (14.0, 9.0), (22.0, 10.0), (4.0, 5.0), (8.0, 2.0), (22.0, 7.0), (13.0, 13.0), (17.0, 2.0), (20.0, 6.0), (21.0, 6.0), (22.0, 9.0), (20.0, 16.0), (8.0, 13.0), (18.0, 3.0), (23.0, 6.0), (20.0, 4.0), (18.0, 15.0), (10.0, 10.0), (12.0, 3.0), (13.0, 7.0), (17.0, 9.0), (10.0, 12.0), (16.0, 4.0), (15.0, 10.0), (6.0, 10.0), (5.0, 5.0), (22.0, 3.0), (9.0, 3.0), (14.0, 4.0), (22.0, 15.0), (11.0, 9.0), (18.0, 6.0), (5.0, 15.0), (14.0, 12.0), (13.0, 11.0), (11.0, 3.0), (5.0, 10.0), (5.0, 3.0), (3.0, 8.0)]
CLEAN_POSITION = []
RUBBISH_POSITION = []
for i in range(RUBBISH_NUM):
    LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
    LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    # rubbish has a unique position and cannot be duplicated with the block
    if ((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in BLOCK_POSITION) or ((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in RUBBISH_POSITION):
        while (((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in BLOCK_POSITION) or ((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in RUBBISH_POSITION)):
            LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
            LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    else:
        LEFT_BOT_X = LEFT_BOT_X
        LEFT_BOT_Y = LEFT_BOT_Y
    RUBBISH_POSITION.append((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1))

# bot number & color & initial positions
BOT_NUM = 6
BOT_COLOR = (255, 0, 0)
BOT_POSITION = []
BOT_LEFT_BOT_X = []
BOT_LEFT_BOT_Y = []
for i in range(BOT_NUM):
    X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
    Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    while ((X/50+1,Y/50+1) in BLOCK_POSITION) or ((X/50+1,Y/50+1) in BOT_POSITION):
        X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
        Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    BOT_POSITION.append((X/50+1, Y/50+1))
    BOT_LEFT_BOT_X.append(X)
    BOT_LEFT_BOT_Y.append(Y)

# initialise bots' observation，distribution of actions and utility 
observation = []    # [{key: observation (bot:-1,block:-1,boundary:-1,vacant:0,rubbish:1), value: happened frequency}]
distribution = []   # [{key: observation, value: {key: action, value: probability}}]
utility = []        # [{key: observation, value: {key: action, value: utility; key: result, value: (action taken, reward, observation)}}]
exp_memory = []
for i in range(BOT_NUM):
    observation.append({})
    utility.append({})
    distribution.append({})
    exp_memory.append({})
tmp_observation = [None] * BOT_NUM
reward_matrix = [0] * BOT_NUM

# parameters
# 600*400: 0.2, 0.95, 0.1
# 1200*800: 0.25, 0.96, 0.08 
alpha = 0.2
gamma = 0.85
zeta = 0.1

epsilon = 1
sensitivity = 1
ln_t = 1 # ln_t = 1 to 10

# lines color and start position
LINE_COLOR = (120, 120, 120, 120)
START_X = 0
START_Y = 0

hit_num = 0
communication = 0
# communication = {}
# for i in range(BOT_NUM):
#     communication[i+1] = 0

class BotEnv(object):
    viewer = None
    actions = ['up', 'down', 'left', 'right']

    def __init__(self):
        self.bot_info = np.zeros(BOT_NUM, dtype=[('x', np.float32), ('y', np.float32)])
        for i in range(BOT_NUM):
            self.bot_info[i]['x'] = BOT_POSITION[i][0]
            self.bot_info[i]['y'] = BOT_POSITION[i][1]

    # take an action from bot
    def step(self, action):
        done = False
        reward = 0
        global BOT_POSITION
        global RUBBISH_POSITION
        global CLEAN_POSITION
        global distribution
        global utility
        global exp_memory

        TMP_BOT_POSITION = []
        for i in range(BOT_NUM):
            self.bot_info[i]['x'] = BOT_POSITION[i][0]
            self.bot_info[i]['y'] = BOT_POSITION[i][1]
            if action[i] == 'up':
                self.bot_info[i]['x'] += 0
                self.bot_info[i]['y'] += 1
            elif action[i] == 'down':
                self.bot_info[i]['x'] += 0
                self.bot_info[i]['y'] += -1
            elif action[i] == 'left':
                self.bot_info[i]['x'] += -1
                self.bot_info[i]['y'] += 0
            elif action[i] == 'right':
                self.bot_info[i]['x'] += 1
                self.bot_info[i]['y'] += 0
            TMP_BOT_POSITION.append((self.bot_info[i]['x'], self.bot_info[i]['y']))
        
        # done and reward
        for i in range(BOT_NUM):
            global hit_num 
            REST_BOT_POSITION = []
            tmp_list = TMP_BOT_POSITION[:]
            tmp_list.pop(i)
            REST_BOT_POSITION.append(tmp_list)
            # reward if move to a rubbish position and move
            if (self.bot_info[i]['x'], self.bot_info[i]['y']) in RUBBISH_POSITION:
                # print('1')
                reward = 15
                BOT_POSITION[i] = TMP_BOT_POSITION[i]
                RUBBISH_POSITION.remove(TMP_BOT_POSITION[i])
                CLEAN_POSITION.append(TMP_BOT_POSITION[i])
                done = True
            # punish if hit a block and do not move
            elif (self.bot_info[i]['x'], self.bot_info[i]['y']) in BLOCK_POSITION:
                reward = -5
                hit_num += 1
                done = True
                # print('2')
            # punish if hit the boundary and do not move
            elif self.bot_info[i]['x'] < 1:
                reward = -5
                hit_num += 1
                done = True
                # print('3')
            elif self.bot_info[i]['x'] > HORIZONTAL_GRID_NUM:
                reward = -5
                hit_num += 1
                done = True
                # print('4')
            elif self.bot_info[i]['y'] < 1:
                reward = -5
                hit_num += 1
                done = True
                # print('5')
            elif self.bot_info[i]['y'] > VERTICAL_GRID_NUM:
                reward = -5
                hit_num += 1
                done = True
                # print('6')
            # punish if hit other bots and do not move
            elif (self.bot_info[i]['x'], self.bot_info[i]['y']) in REST_BOT_POSITION:
                reward = -10
                hit_num += 1
                done = True
                # print('7')
            # neither reward nor punish if move to a vacant grid
            else:
                # print('8')
                reward = 0
                BOT_POSITION[i] = TMP_BOT_POSITION[i]
                done = True
            # calculate reward matrix
            reward_matrix[i] += reward

            # print('utility = ',utility)
            if utility != [{}] * BOT_NUM:
                if tmp_observation[i] not in exp_memory[i]:
                #print(tmp_observation[i])
                    exp_memory[i][tmp_observation[i]] = []
                

                exp_memory[i][tmp_observation[i]].append((action[i], reward))

                # utility of t+1
                new_observation = BotEnv().get_observation(i)
                if new_observation in observation:
                    max_utility = max(utility[i][new_observation].values())
                else:
                    max_utility = 1
                #print(action)
                utility[i][tmp_observation[i]][action[i]] = (1 - alpha) * utility[i][tmp_observation[i]][action[i]] + alpha * (reward + gamma * max_utility)
                #{key: result, value: (action taken, reward, observation, happened frequency}
                utility[i][tmp_observation[i]]['result'] = (action[i], reward, tmp_observation[i], observation[i][tmp_observation[i]])
                #print(tmp_observation[i])
                # total reward
                total_reward = 0
                for j in range(len(self.actions)):
                    total_reward += distribution[i][tmp_observation[i]][self.actions[j]] * utility[i][tmp_observation[i]][self.actions[j]]
                for j in range(len(self.actions)):
                    distribution[i][tmp_observation[i]][self.actions[j]] = distribution[i][tmp_observation[i]][self.actions[j]] + zeta * (utility[i][tmp_observation[i]][self.actions[j]] - total_reward)
                BotEnv().normalise(distribution[i][tmp_observation[i]])
        return reward, done
    #GUI
    def render(self):
        if self.viewer is None:
            self.viewer = Viewer()
        self.viewer.render()

    # random move
    def sample_action(self):
        action = []
        for i in range(BOT_NUM):
            action.append(random.choice(self.actions))    # generate an action for each bot
        return action

    # get the observation of a bot
    def get_observation(self, i):
        x = BOT_POSITION[i][0]
        y = BOT_POSITION[i][1]
        around = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1)]
        observation_ = ""
        for j in range(len(around)):
            if around[j] in RUBBISH_POSITION:
                observation_ += "1,"
            elif around[j] in BLOCK_POSITION:
                observation_ += "-1,"
            elif around[j] in BOT_POSITION:
                observation_ += "-2,"
            elif around[j][0] < 1:
                observation_ += "-1,"
            elif around[j][0] > HORIZONTAL_GRID_NUM:
                observation_ += "-1,"
            elif around[j][1] < 1:
                observation_ += "-1,"
            elif around[j][1] > VERTICAL_GRID_NUM:
                observation_ += "-1,"
            else:
                observation_ += "0,"
        observation_ = observation_[:-1]
        return observation_

    # normalise the distribution of observations
    def normalise(self, distribution_):
        c0 = 0.5
        delta = 0.001
        d = min(distribution_.values())
        if d < delta:
            rho = (c0 - delta)/(c0-d)
            for k in range(len(self.actions)):
                distribution_[self.actions[k]] = c0 - rho * (c0 - distribution_[self.actions[k]])
        pi = sum(distribution_.values())
        for k in range(len(self.actions)):
            distribution_[self.actions[k]] = distribution_[self.actions[k]]/pi
        return distribution_

    # find the most similar observation
    def similar_observation(self, key_, dict1):
        keys = list(dict1.keys())
        key_ = key_.split(',')
        similar_observation_ = {}
        for i in range(len(keys)):
            keys[i] = keys[i].split(',')
            # make a comparison for each surroundings
            difference = 0
            similar = True
            for j in range(len(keys[i])):
                if keys[i][j] != key_[j]:
                    difference += 1
                # set the tolerated diffenence
                if difference > 1:
                    similar = False
                    break
            if similar:
                similar_observation_[list(dict1.keys())[i]] = (list(dict1.values())[i], difference)
        # return all the similar observations
        if similar_observation_ != {}:
            return similar_observation_
        else:
            return {}

    # add laplace noise
    def laplace(self, m, n, mu, b):
        u = random.uniform(0, 1)
        u = u - 0.5
        sigma = 1
        b = sigma/math.sqrt(2)
        if u >= 0:
            y = mu - b * u * math.log(1 - 2 * abs(u))
        else:
            y = mu + b * u * math.log(1 - 2 * abs(u))
        return y

    # weighted transfer learning algorithm with adding noise
    def algorithm(self):
        action = []
        global BOT_POSITION
        global observation 
        global distribution
        global utility
        global tmp_observation
        global communication
        global exp_memory

        for i in range(BOT_NUM):
            key = BotEnv().get_observation(i)
            tmp_observation[i] = key
            #if this obs is old
            #print(key)
            #print(observation[i])
            if key in observation[i]:
                max_action = max(distribution[i][key], key=distribution[i][key].get)
                max_pro = distribution[i][key][max_action]
                observation[i][key] += 1
                if max_pro > 0.4 and observation[i][key] > 10 :
                    #np.random.seed(0)
                    #p = np.array([distribution[i][key]['up'], distribution[i][key]['down'], distribution[i][key]['left'], distribution[i][key]['right']])
                    #index = np.random.choice(['up', 'down', 'left', 'right'], p = p.ravel())
                    action.append(max_action)
                    #print('self')
                else:
                    max_num_obs = 0
                    teacher_id = -1
                    communication += 1
                    for j in range(BOT_NUM):
                        if j != i:
                            #tmp_dict = observation[j].copy()
                            if len(exp_memory[j]) != 0:
                                tmp_dict = exp_memory[j].copy()
                                knowledges = BotEnv().similar_observation(key, tmp_dict)
                                #how many times doest this obs happen
                                num_obs = 0
                                for k in knowledges.keys():
                                    num_obs += observation[j][k]
                                if num_obs > 30 and num_obs > max_num_obs:
                                    teacher_id = j
                                    max_num_obs = num_obs
                                    teacher_knowledges = knowledges.copy()
                    if teacher_id != -1:
                        action_reward = {}
                        action_count = {}
                        for k in teacher_knowledges.keys():
                            for p in exp_memory[teacher_id][k]:
                                if p[0] not in action_reward:
                                    action_reward[p[0]] = p[1]
                                    action_count[p[0]] = 1
                                else:
                                    action_reward[p[0]] += p[1]
                                    action_count[p[0]] += 1
                        for u in action_reward.keys():
                            action_reward[u] = action_reward[u]/action_count[u]
                        #print(action_reward)
                        teacher_action = max(action_reward, key=action_reward.get)
                        if action_reward[teacher_action] > 0.1:
                            
                            action.append(teacher_action)
                        else:
                            action.append(max_action)
                        #print('teacher')
                    else:
                        #max_action = random.choice(['up','down','left','right'])
                        action.append(max_action)
            #if this obs is new
            else:
                max_num_obs = 0
                teacher_id = -1
                communication += 1
                for j in range(BOT_NUM):
                    if j != i:
                        #tmp_dict = observation[j].copy()
                        if len(exp_memory[j]) != 0:
                            tmp_dict = exp_memory[j].copy()
                            #print(tmp_dict)
                            knowledges = BotEnv().similar_observation(key, tmp_dict)
                            #how many times doest this obs happen
                            num_obs = 0
                            for k in knowledges.keys():
                                num_obs += observation[j][k]
                            if num_obs > 30 and num_obs > max_num_obs:
                                teacher_id = j
                                max_num_obs = num_obs
                                teacher_knowledges = knowledges.copy()
                if teacher_id != -1:
                    action_reward = {}
                    action_count = {}
                    for k in teacher_knowledges.keys():
                        for p in exp_memory[teacher_id][k]:
                            if p[0] not in action_reward:
                                action_reward[p[0]] = p[1]
                                action_count[p[0]] = 1
                            else:
                                action_reward[p[0]] += p[1]
                                action_count[p[0]] += 1
                    for u in action_reward.keys():
                        action_reward[u] = action_reward[u]/action_count[u]

                    teacher_action = max(action_reward, key=action_reward.get)
                    if action_reward[teacher_action] > 0.1:
                        
                        action.append(teacher_action)
                    else:
                        max_action = random.choice(['up','down','left','right'])
                        action.append(max_action)
                else:
                    max_action = random.choice(['up','down','left','right'])
                    action.append(max_action)

                #print('run')
                observation[i][key] = 1
                utility[i][key] = {}
                utility[i][key]['up'] = 1
                utility[i][key]['down'] = 1
                utility[i][key]['left'] = 1
                utility[i][key]['right'] = 1
                distribution[i][key] = {}
                distribution[i][key]['up'] = 0.25
                distribution[i][key]['down'] = 0.25
                distribution[i][key]['left'] = 0.25
                distribution[i][key]['right'] = 0.25
        return action



# GUI
class Viewer(pyglet.window.Window):
    def __init__(self):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(WINDOW_WIDTH, WINDOW_HEIGHT, resizable=False, caption='Room', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.batch = pyglet.graphics.Batch()                       # display whole batch at once
        # draw blocks
        for i in range(BLOCK_NUM):
            BLOCK_LEFT_BOT_X = BLOCK_POSITION[i][0] * 50 - 50
            BLOCK_LEFT_BOT_Y = BLOCK_POSITION[i][1] * 50 - 50
            self.point = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,                       # 4 corners
                ('v2f', [BLOCK_LEFT_BOT_X, BLOCK_LEFT_BOT_Y,       # location
                         BLOCK_LEFT_BOT_X, BLOCK_LEFT_BOT_Y + 50,
                         BLOCK_LEFT_BOT_X + 50, BLOCK_LEFT_BOT_Y + 50,
                         BLOCK_LEFT_BOT_X + 50, BLOCK_LEFT_BOT_Y]),
                ('c3B', (BLOCK_COLOR) * 4))                        # color

        # draw rubbish
        for i in range(RUBBISH_NUM):
            RUBBISH_LEFT_BOT_X = RUBBISH_POSITION[i][0] * 50 - 50
            RUBBISH_LEFT_BOT_Y = RUBBISH_POSITION[i][1] * 50 - 50
            self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,                       # 4 corners
                ('v2f', [RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y,   # location
                         RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y]),
                ('c3B', (RUBBISH_COLOR) * 4))                      # color

        # draw bots at the initial positions 
        self.bots = []
        for i in range(BOT_NUM): 
            self.bot = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [BOT_LEFT_BOT_X[i], BOT_LEFT_BOT_Y[i],             # location
                     BOT_LEFT_BOT_X[i], BOT_LEFT_BOT_Y[i] + 50,
                     BOT_LEFT_BOT_X[i] + 50, BOT_LEFT_BOT_Y[i] + 50,
                     BOT_LEFT_BOT_X[i] + 50, BOT_LEFT_BOT_Y[i]]),
            ('c3B', (BOT_COLOR) * 4))                                  # color
            self.bots.append(self.bot)

    # draw grid
    def draw_grid(self,start_x,start_y):
        rows = VERTICAL_GRID_NUM+ 1
        columns = HORIZONTAL_GRID_NUM + 1
        # draw rows
        for row in range(rows):
            pyglet.graphics.draw(
                2, pyglet.gl.GL_LINES,
                ('v2f',
                    (
                        start_x, row * GRID_HEIGHT + start_y,
                        GRID_HEIGHT * HORIZONTAL_GRID_NUM + start_x, row * GRID_HEIGHT + start_y
                    )
                ),
                ('c4B', LINE_COLOR * 2)
            )

        # draw columns
        for column in range(columns):
            pyglet.graphics.draw(
                2, pyglet.gl.GL_LINES,
                ('v2f',
                    (
                        column * GRID_WIDTH + start_x, start_y,
                        column * GRID_WIDTH + start_x, GRID_WIDTH * VERTICAL_GRID_NUM + start_y
                    )
                ),
                ('c4B', LINE_COLOR * 2)
            )

    def render(self):
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.draw_grid(START_X,START_Y)

    def _update(self):
        for i in range(BOT_NUM):
            self.bots[i].vertices = np.concatenate(([BOT_POSITION[i][0] * 50 -50, BOT_POSITION[i][1] * 50 - 50], [BOT_POSITION[i][0] * 50, BOT_POSITION[i][1] * 50 - 50], [BOT_POSITION[i][0] * 50, BOT_POSITION[i][1] * 50], [BOT_POSITION[i][0] * 50 -50, BOT_POSITION[i][1] * 50]))
        
        # re-draw the rubbish
        for i in range(len(CLEAN_POSITION)):
            RUBBISH_LEFT_BOT_X = CLEAN_POSITION[i][0] * 50 - 50
            RUBBISH_LEFT_BOT_Y = CLEAN_POSITION[i][1] * 50 - 50
            self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,                       # 4 corners
                ('v2f', [RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y,   # location
                         RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y]),
                ('c3B', (CLEAN_COLOR) * 4))                      # color

        for i in range(len(RUBBISH_POSITION)):
            RUBBISH_LEFT_BOT_X = RUBBISH_POSITION[i][0] * 50 - 50
            RUBBISH_LEFT_BOT_Y = RUBBISH_POSITION[i][1] * 50 - 50
            self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,                       # 4 corners
                ('v2f', [RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y,   # location
                         RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y]),
                ('c3B', (RUBBISH_COLOR) * 4))                      # color

if __name__ == '__main__':
    env = BotEnv()
    TotalStep = 1
    TurnStep = 1
    turn = 1
    file.write("Parameter Settings:" + "\n")
    file.write("alpha = " + str(alpha) + "\n")
    file.write("gamma = " + str(gamma) + "\n")
    file.write("zeta = " + str(zeta) + "\n")
    file.write("epsilon = "+ str(epsilon) + "\n")
    file.write("sensitivity = "+ str(sensitivity) + "\n")
    file.write("ln_t = "+ str(ln_t) + "\n")
    file.write("block position = " + str(BLOCK_POSITION) + "\n")
    file.write("rubbish position = " + str(RUBBISH_POSITION) + "\n")
    file.write("Turn     " + "Block     " + "Rubbish     " + "Hit        " + "communication        " + "TurnStep     " + "TotalStep     " + "avg_comm, avg_hit, avg_step " +"\n")
    file.flush()
    while turn <= 600:
        while len(RUBBISH_POSITION) > 5: # Done?
            #env.render() # GUI
            env.step(env.sample_action()) # GUI
            env.step(env.algorithm())
            #alpha = (TotalStep/(TotalStep + 1)) * alpha
            # print("turn = ", turn, "TotalStep = ", TotalStep, "TurnStep = ", TurnStep)
            # print('Block Position: ', BLOCK_POSITION)
            # print('Clean Position: ', CLEAN_POSITION)
            # print('Rubbish Position: ', RUBBISH_POSITION)
            # print('Bot Position: ', BOT_POSITION)
            TurnStep += 1
            TotalStep += 1
        file.write(str(turn) +"        "+ str(BLOCK_NUM) +"        "+ str(RUBBISH_NUM) +"          "+ str(hit_num) + "        "+ str(communication) + "             " + str(TurnStep) + "           " + str(TotalStep) + "				" + str(round((communication/turn), 1)) + "                " + str(round((hit_num/turn), 1)) + "               " + str(round((TotalStep/turn), 1))+'\n')
        file.flush()
        # for i in range(BOT_NUM):
        #     communication[i+1] = 0
        turn += 1
        TurnStep = 1
        TotalStep += 1
        RUBBISH_POSITION += CLEAN_POSITION
        CLEAN_POSITION = []
    file.close()
