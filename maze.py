import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import time
import csv


LEARNING_RATE = 0.4
DISCOUNT = 0.96
EPISODES = 500

EPSILON = 0.1

def show_maze():
    # print("SHOW!",self.state[0], self.state[1])
    plt.grid('on')
    nrows, ncols = maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(maze)
    canvas[nrows-1, ncols-1] = 0.8 # terminal block
    img = plt.imshow(canvas, interpolation='none', cmap='CMRmap')
    img = plt.show()
    return img

maze = np.array([
    [  1.0, 1.0,1.0, 0.0,0.0, 0.0,1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 0.0,  0.0,  1.0,  1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 0.0,0.0, 0.0,1.0, 1.0,0.0, 1.0,  0.0,  1.0,  1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 0.0,1.0, 1.0,0.0, 0.0,0.0, 1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0,  0.0,  1.0,  0.0],
    [  1.0, 0.0,0.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  0.0,  1.0,  1.0,  0.0,  1.0,  1.0,  1.0,  0.0,  1.0,  0.0],
    [  1.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  0.0,  1.0,  1.0,  0.0,  1.0,  1.0,  1.0,  0.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  0.0,  1.0,  0.0,  1.0,  1.0,  0.0,  1.0,  0.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  0.0,  1.0,  0.0,  1.0,  1.0,  0.0,  1.0,  0.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  0.0,  1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  1.0,  1.0,  0.0,  1.0,  0.0,  1.0,  1.0,  0.0,  1.0,  1.0],
    [  1.0, 0.0,1.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  0.0,  1.0,  1.0,  0.0,  1.0,  1.0],
    [  1.0, 1.0,1.0, 0.0,1.0, 1.0,1.0, 1.0,0.0, 0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  1.0,  0.0,  1.0,  1.0],
    [  1.0, 1.0,1.0, 0.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  1.0,  0.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0],
    [  0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,  0.0,  0.0,  1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    [  1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    [  1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 0.0,0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [  1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 0.0,1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.0,  1.0],
    [  1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 0.0,1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.0,  1.0,  0.0,  1.0],
    [  1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.0,  1.0,  1.0,  1.0]
])

# maze = np.array([
#     [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#     [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
#     [ 1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
#     [ 1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#     [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
# ])

show_maze()

visited_mark = 0.8  # Cells visited by the agent block be painted by gray 0.8
agent_mark = 0.5      # The current agent block will be painted by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

nLEFT = (-1,0)
nUP = (0,-1)
nRIGHT = (1,0)
nDOWN = (0,1)
def Average(lst):
    return sum(lst) / len(lst)

class Stats():
    def __init__(self):
        self.all_best_visited = []
        self.all_best_ep_reward = []
        self.all_best_episode = []

    def show_best(self, alg,maze):
        print("Algorithm: " + alg)
        print(f"Average Episode: {Average(self.all_best_episode)} \n Reward: {Average(self.all_best_ep_reward)}")
        print(f"Steps taken: {Average([len(x) for x in self.all_best_visited])}")

        # canvas = np.copy(self.mazeCopy)
        position = 0
        newT = time.time()

        newT = time.time()
        canvas = np.copy(maze.mazeCopy)
        while position < len(maze.best_visited):
            if time.time() - newT >= 0.2:
                plt.close()
                plt.grid('on')
                nrows, ncols = maze.mazeCopy.shape
                ax = plt.gca()
                ax.set_xticks(np.arange(0.5, nrows, 1))
                ax.set_yticks(np.arange(0.5, ncols, 1))
                canvas[nrows-1, ncols-1] = 0.8
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                for visited in range(0,position):
                    vrow = maze.best_visited[visited][0]
                    vcol = maze.best_visited[visited][1]
                    canvas[vrow,vcol] = 0.6

                agent_row, agent_col = maze.best_visited[position]
                canvas[agent_row, agent_col] = 0.3   # agent block
                 # terminal block
                position += 1
                newT = time.time()
                img = plt.imshow(canvas, interpolation='none', cmap='CMRmap')
                img = plt.show()

class Maze(object):

    def __init__(self, alg, maze, agent = (0,0)):
        self.epsilon = EPSILON
        self.maze = np.array(maze) #MAZE 10x10
        nrows, ncols = self.maze.shape #nrows = 10, ncols = 10 (10, 10)
        self.target = (nrows-1, ncols-1)   # terminal block where the target is (9, 9)
        self.freeblocks = [(r,c) for r in range(nrows) for c in range(ncols) if self.maze[r,c] == 1.0]
        self.freeblocks.remove(self.target)
        self.qtable = {}
        if self.maze[self.target] == 0.0:
            raise Exception("Invalid Maze: Target block cannot be blocked!")
        if agent not in self.freeblocks:
            raise Exception("Invalid Agent Location: can only sit on free blocks!")
        self.best_visited = set()
        self.best_ep_reward = -1000000
        self.best_episode = 0


        self.reset(agent)
        if alg == "Sarsa":
            self.sarsa()
        else:
            self.qlearning()

    def init_qtable(self):
        self.q_table = {}
        for row in range(0,self.maze.shape[0]):
            for col in range(0,self.maze.shape[1]):
                self.q_table[(row,col)] = {}
                for ax in self.possible_actions(row,col):
                    self.q_table[(row,col)][ax] = random.randrange(-2,0)
                        # qtable = {(0,1): { (0,1): -1.3,
                        #                     (1,0): -0.4,
                        #                     },
                        #         (0,1): { (0,-1): -1,
                        #                 (1,0): -1.7,
                        #                 (0,1): -0.6,
                        #                 }
                        # }

    def reset(self, agent = (0,0), keepQ=None):
        self.agent = agent
        self.mazeCopy = np.copy(self.maze)
        nrow, ncols = self.mazeCopy.shape
        row, col = agent
        self.maze[row, col] = agent_mark
        self.state = (row, col)
        self.min_reward = -0.5*self.maze.size
        self.total_reward = 0
        self.visited = []

        if not keepQ: self.init_qtable()
        return self.state

    def update_state(self, action):
        nrows, ncols = self.mazeCopy.shape
        nrow, ncol = agent_row, agent_col = self.state

        if self.mazeCopy[agent_row, agent_col] > 0.0:
            self.visited.append((agent_row, agent_col))
        # self.visited = set(self.visited)
        valid_actions = self.valid_actions()

        if action in valid_actions:
            if action == nLEFT:
                ncol -= 1
            elif action == nUP:
                nrow -= 1
            if action == nRIGHT:
                ncol += 1
            elif action == nDOWN:
                nrow += 1
        # new state
        self.state = (nrow, ncol)

    def get_reward(self):
        agent_row, agent_col = self.state
        nrows, ncols = self.mazeCopy.shape
        if agent_row == nrows-1 and agent_col == ncols-1:
            return 1.0
        if (agent_row, agent_col) in self.visited:
            return -1.0
        return -0.2

    def actions(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return self.state, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.mazeCopy)
        nrows, ncols = self.mazeCopy.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the agent
        row, col = self.state
        canvas[row, col] = agent_mark
        return canvas

    def game_status(self):
        # if len(self.visited) > 100:
        #     return 'lost'
        if self.total_reward < self.min_reward:
            return 'lost'
        agent_row, agent_col = self.state
        nrows, ncols = self.mazeCopy.shape
        if agent_row == nrows-1 and agent_col == ncols-1:
            return 'won'
        return 'not_over'

    def possible_actions(self, row, col):
        actions = [(-1,0), (0,-1), (1,0), (0,1)]
        nrows, ncols = self.mazeCopy.shape
        if row == 0:
            actions.remove(nUP)
        elif row == nrows-1:
            actions.remove(nDOWN)

        if col == 0:
            actions.remove(nLEFT)
        elif col == ncols-1:
            actions.remove(nRIGHT)

        if row>0 and self.mazeCopy[row-1,col] == 0.0:
            actions.remove(nUP)
        if row<nrows-1 and self.mazeCopy[row+1,col] == 0.0:
            actions.remove(nDOWN)

        if col>0 and self.mazeCopy[row,col-1] == 0.0:
            actions.remove(nLEFT)
        if col<ncols-1 and self.mazeCopy[row,col+1] == 0.0:
            actions.remove(nRIGHT)

        return actions


    def valid_actions(self, cell = None):
        if cell is None:
            row, col = self.state
        else:
            row, col = cell
        actions = [(-1,0), (0,-1), (1,0), (0,1)]
        nrows, ncols = self.mazeCopy.shape
        if row == 0:
            actions.remove(nUP)
        elif row == nrows-1:
            actions.remove(nDOWN)

        if col == 0:
            actions.remove(nLEFT)
        elif col == ncols-1:
            actions.remove(nRIGHT)

        if row>0 and self.mazeCopy[row-1,col] == 0.0:
            actions.remove(nUP)
        if row<nrows-1 and self.mazeCopy[row+1,col] == 0.0:
            actions.remove(nDOWN)

        if col>0 and self.mazeCopy[row,col-1] == 0.0:
            actions.remove(nLEFT)
        if col<ncols-1 and self.mazeCopy[row,col+1] == 0.0:
            actions.remove(nRIGHT)

        return actions

    def qlearning(self):
        breakplz = False
        times_won = 0
        for episode in range(EPISODES):
            self.reset(keepQ=True)

            episode_reward = 0
            SHOW_EVERY = 1
            done = False

            while not done:
                if np.random.random() > self.epsilon:
                    action = max(self.q_table[self.state], key=self.q_table[self.state].get)
                else:
                    action = random.choice(self.valid_actions())

                current_state = self.state                      #(0,0)
                current_q = self.q_table[self.state][action]         #-1.3

                new_state, reward, status = self.actions(action) #new_state=(0,1), reward= -0.04, not_done
                episode_reward += reward

                if status == "not_over":
                    max_future_q = max(self.q_table[new_state].values()) #-0.6
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                    self.q_table[current_state][action] = new_q
                else:
                    if status == "won":
                        if episode_reward > self.best_ep_reward:
                            self.best_visited = self.visited
                            self.best_ep_reward = episode_reward
                            self.best_episode = episode
                        times_won += 1
                    done = True
            # if status == "won":
            #     break

    def sarsa(self):
        breakplz = False
        times_won = 0
        for episode in range(EPISODES):
            self.reset(keepQ=True)

            episode_reward = 0
            SHOW_EVERY = 1
            done = False

            if np.random.random() > self.epsilon:
                action = max(self.q_table[self.state], key=self.q_table[self.state].get)
            else:
                action = random.choice(self.valid_actions())

            while not done:
                current_state = self.state
                current_q = self.q_table[self.state][action]         #-1.3

                new_state, reward, status = self.actions(action) #new_state=(0,1), reward= -0.04, not_done
                episode_reward += reward

                if np.random.random() > self.epsilon:
                    new_action = max(self.q_table[new_state], key=self.q_table[new_state].get)
                else:
                    new_action = random.choice(self.valid_actions())

                if status == "not_over":
                    future_q = self.q_table[new_state][new_action]
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_q)
                    self.q_table[current_state][action] = new_q
                    action = new_action
                else:
                    if status == "won":
                        if episode_reward > self.best_ep_reward:
                            self.best_visited = self.visited
                            self.best_ep_reward = episode_reward
                            self.best_episode = episode
                        times_won += 1
                    done = True
            # if status == "won":
            #     break
    def show(self):
        # print("SHOW!",self.state[0], self.state[1])
        plt.grid('on')
        nrows, ncols = self.mazeCopy.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(self.mazeCopy)
        for row,col in self.visited:
            canvas[row,col] = 0.6
        agent_row, agent_col = self.state
        canvas[agent_row, agent_col] = 0.3   # agent block
        canvas[nrows-1, ncols-1] = 0.8 # terminal block
        img = plt.imshow(canvas, interpolation='none',cmap='CMRmap')
        img = plt.show()
        return img

algs = ["Sarsa", "qlearning"]
for alg in algs:
    print("Algorithm:", alg)
    stats = Stats()
    for i in range(0,1):
        qmaze = Maze(alg,maze)
        print(i, "Best episode",qmaze.best_episode)
        stats.all_best_visited.append(qmaze.best_visited)
        stats.all_best_ep_reward.append(qmaze.best_ep_reward)
        stats.all_best_episode.append(qmaze.best_episode)
    qmaze.show()
    stats.show_best(alg,qmaze)
