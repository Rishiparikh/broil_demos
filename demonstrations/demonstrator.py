import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
import copy
import os
import pickle
from gym.spaces import Discrete, Box
from spinup.envs.pointbot import *
import datetime
import os
import pickle
import math
import sys
import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from PIL import Image

from spinup.envs.pointbot_const import *

from spinup.rewards.cvar_utils import cvar_enumerate_pg
from spinup.rewards.pointbot_reward_utils import PointBotReward

class LineBuilder:
    def __init__(self, line, env, fig, typ):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.xv = [0]
        self.yv = [0]
        self.xt = []
        self.yt = []
        self.typ = typ
        self.env = env
        self.steps = 0
        self.state = env.state
        self.states = [self.state]
        self.actions = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.info = fig.canvas.mpl_connect('key_press_event', self.press)
        self.transitions = []
        self.obs = None
        self.counter = 0

    def press(self, event):
        sys.stdout.flush()
        if event.key == 'p':
            print("X: ", str(self.env.state[0]), " Y: ", str(self.env.state[1]))
        if event.key == 'w':
            print("Most recent state: ", str(self.states[-1]), "\n states: ", str(self.states))
        if event.key == 't':
            if TRASH:
                print("Closest Trash X: ", str(self.env.closest_trash(self.env.state)[0]), " Y: ", str(self.env.closest_trash(self.env.state)[1]))
            else:
                print("No TRASH on the field!")
        if event.key == 'a':
            print("Current Feature: ", str(self.env.feature), " Remaining steps: " + str(HORIZON - self.steps))
        if event.key == 'e':
            if TRASH:
                plt.savefig('demonstrations/visualization_' + self.typ + "_" + str(args.dem_num) + '.png')
                plt.close()
            else:
                return
                # if np.linalg.norm(np.subtract(GOAL_STATE, self.states[-1][:4])) <= GOAL_THRESH:
                #     plt.savefig('demonstrations/visualization_' + self.typ + "_" + str(args.dem_num) + '.png')
                #     plt.close()
                # else:
                #     print("\nNot proper ending! X distance from END: " + str(self.xs[-1] - END_POS[0]) + " Y distance from END: " + str(self.ys[-1] - END_POS[1]))
        if event.key == 'r':
            if (os.path.exists("demonstrations/states_" + str(args.dem_num) + ".txt")):
                os.remove("demonstrations/states_" + str(args.dem_num) + ".txt")
        if event.key == 'g':
            if event.inaxes!=self.line.axes: return

            if self.steps == HORIZON:
                plt.savefig('demonstrations/visualization_' + self.typ + "_" + str(args.dem_num) + '.png')
                plt.close()
                return

            final_x = END_POS[0]
            final_y = END_POS[1]
            init_x = self.xs[-1]
            init_y = self.ys[-1]

            diff_x = final_x - init_x
            diff_y = final_y - init_y

            x_f = diff_x
            y_f = diff_y

            print(diff_x)
            if diff_x >= 0 and diff_y >= 0:
                if abs(diff_x) >= abs(diff_y):
                    x_f = abs(diff_x/diff_x) * MAX_FORCE/10
                    y_f = abs(diff_y/diff_x) * MAX_FORCE/10
                else:
                    x_f = abs(diff_x/diff_y) * MAX_FORCE/10
                    y_f = abs(diff_y/diff_y) * MAX_FORCE/10

            elif diff_x < 0 and diff_y >= 0:
                if abs(diff_x) >= abs(diff_y):
                    x_f = -abs(diff_x/diff_x) * MAX_FORCE/10
                    y_f = abs(diff_y/diff_x) * MAX_FORCE/10
                else:
                    x_f = -abs(diff_x/diff_y) * MAX_FORCE/10
                    y_f = abs(diff_y/diff_y) * MAX_FORCE/10

            elif diff_x >= 0 and diff_y < 0:
                if abs(diff_x) >= abs(diff_y):
                    x_f = abs(diff_x/diff_x) * MAX_FORCE/10
                    y_f = -abs(diff_y/diff_x) * MAX_FORCE/10
                else:
                    x_f = abs(diff_x/diff_y) * MAX_FORCE/10
                    y_f = -abs(diff_y/diff_y) * MAX_FORCE/10

            elif diff_x < 0 and diff_y < 0:
                if abs(diff_x) >= abs(diff_y):
                    x_f = -abs(diff_x/diff_x) * MAX_FORCE/5
                    y_f = -abs(diff_y/diff_x) * MAX_FORCE/10
                else:
                    x_f = -abs(diff_x/diff_y) * MAX_FORCE/10
                    y_f = -abs(diff_y/diff_y) * MAX_FORCE/10

            act = tuple((x_f, y_f))
            _, _, _, state = self.env.step(act)
            new_state = state["next_state"]
            self.actions.append(self.env.curr_action)

            if TRASH:
                plt.scatter([self.env.next_trash[0]],[self.env.next_trash[1]], [20], '#000000')

            self.xs.append(new_state[0])
            self.ys.append(new_state[1])
            self.steps += 1
            self.xv.append(new_state[1])
            self.yv.append(new_state[3])
            self.states.append(new_state)
            self.line.set_data(self.xs, self.ys)
            print(self.line)
            self.line.figure.canvas.draw()

    def __call__(self, event):
        if event.inaxes!=self.line.axes: return
        if self.steps == HORIZON:
            plt.savefig('demonstrations/visualization_' + self.typ + "_" + str(args.dem_num) + '.png')
            plt.close()
            return

        final_x = event.xdata
        final_y = event.ydata
        init_x = self.xs[-1]
        init_y = self.ys[-1]

        diff_x = final_x - init_x
        diff_y = final_y - init_y

        x_f = diff_x
        y_f = diff_y

        if diff_x >= 0 and diff_y >= 0:
            if abs(diff_x) >= abs(diff_y):
                x_f = abs(diff_x/diff_x) * MAX_FORCE
                y_f = abs(diff_y/diff_x) * MAX_FORCE
            else:
                x_f = abs(diff_x/diff_y) * MAX_FORCE
                y_f = abs(diff_y/diff_y) * MAX_FORCE

        elif diff_x < 0 and diff_y >= 0:
            if abs(diff_x) >= abs(diff_y):
                x_f = -abs(diff_x/diff_x) * MAX_FORCE
                y_f = abs(diff_y/diff_x) * MAX_FORCE
            else:
                x_f = -abs(diff_x/diff_y) * MAX_FORCE
                y_f = abs(diff_y/diff_y) * MAX_FORCE

        elif diff_x >= 0 and diff_y < 0:
            if abs(diff_x) >= abs(diff_y):
                x_f = abs(diff_x/diff_x) * MAX_FORCE
                y_f = -abs(diff_y/diff_x) * MAX_FORCE
            else:
                x_f = abs(diff_x/diff_y) * MAX_FORCE
                y_f = -abs(diff_y/diff_y) * MAX_FORCE

        elif diff_x < 0 and diff_y < 0:
            if abs(diff_x) >= abs(diff_y):
                x_f = -abs(diff_x/diff_x) * MAX_FORCE
                y_f = -abs(diff_y/diff_x) * MAX_FORCE
            else:
                x_f = -abs(diff_x/diff_y) * MAX_FORCE
                y_f = -abs(diff_y/diff_y) * MAX_FORCE

        act = tuple((x_f, y_f))
        next_obs, reward, done, info = self.env.step(act)
        transition = {'obs': info['next_state'], 'action': act, 'reward': float(reward),
                      }
        # print({k: v.dtype for k, v in transition.items() if 'obs' in k})
        state = info['next_state']
        self.obs = next_obs
        new_state = state
        self.xs.append(new_state[0])
        self.ys.append(new_state[1])
        self.steps += 1
        self.states.append(new_state)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        im = self.env.render()
        im = Image.fromarray(im, 'RGB')
        # plt.imshow(im)
        # plt.show()
        im.save("./demonstrations/demos/image_{}.png".format(self.counter))
        self.counter += 1
        self.transitions.append(transition)

def init(typ="Good"):
    env = gym.make('PointBot-v1')
    env.reset()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('PointBot Env '+ typ +' Demonstrator')
    line, = ax.plot([env.state[0]], env.state[1])  # empty line
    linebuilder = LineBuilder(line, env, fig, typ)
    # num_obst = len(env.obstacle.obs)
    # for i in range(num_obst):
    #     xbound = env.obstacle.obs[i].boundsx
    #     ybound = env.obstacle.obs[i].boundsy
    #     rect = patches.Rectangle((xbound[0],ybound[0]),abs(xbound[1] - xbound[0]),abs(ybound[1] - ybound[0]),linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3', fill = True)
    #     ax.add_patch(rect)

    if TRASH:
        plt.scatter([env.next_trash[0]],[env.next_trash[1]], [25], '#000000')

    ax.scatter([env.state[0]],[env.state[1]],  [5], '#00FF00')
    if not TRASH:
        ax.scatter([END_POS[0]],[END_POS[1]],  [5], '#FF0000')

    ax.set_xlim([env.grid[0], env.grid[1]])
    ax.set_ylim([env.grid[2], env.grid[3]])
    plt.show()
    return linebuilder

def end(linebuilder, typ="Good"):
    feature_length = 0 #sum(linebuilder.env.feature)
    linebuilder.env.draw(trajectories=linebuilder.transitions, show=True)
    if TRASH:
        for _ in range(HORIZON-feature_length):
            next_state = [linebuilder.env.state[0], 0, linebuilder.env.state[1], 0] + NOISE_SCALE * np.random.randn(4)
            next_state = np.concatenate((next_state, linebuilder.env.closest_trash(linebuilder.env.state)))
            linebuilder.states.append(next_state)
            linebuilder.actions.append([0, 0])
    else:
        for _ in range(HORIZON-feature_length):
            linebuilder.states.append([END_POS[0], 0, END_POS[1], 0] + NOISE_SCALE * np.random.randn(4))

    if not os.path.exists('demonstrations'):
        os.makedirs('demonstrations')

    try:
        f = open("demonstrations/states_" + str(args.dem_num) + ".txt", "a")
        # f.write("\n" + typ)
        # f.write("\nFeature: " + str(linebuilder.env.feature))
        f.write("\n\nStates: " + str(linebuilder.transitions))
        f.close()
        # return linebuilder.env.feature, linebuilder.states, linebuilder.actions
        return None
    except AssertionError as msg:
        print(msg)
        return None


if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--dem_num', type=int, default=1)
    parser.add_argument('--single', type=bool, default=False)
    args = parser.parse_args()

    if args.single:
        linebuilder = init("Optimal")
        feature, states, actions = end(linebuilder, "Optimal")

        dic = {"feature": feature,  "states": states, "actions": actions}
        p = open("demonstrations/features_states_actions_" + str(args.dem_num) + ".pkl", "wb")
        pickle.dump(dic, p)
        p.close()
    else:
        linebuilder = init()
        good_feature, good_states, good_actions = end(linebuilder)

        linebuilder = init("Bad")
        bad_feature, bad_states, bad_actions = end(linebuilder, "Bad")

        dic = {"Good_feature": good_feature, "Bad_feature": bad_feature, "Good_states": good_states, "Good_actions": good_actions, "Bad_states": bad_states, "Bad_actions": bad_actions}
        p = open("demonstrations/features_states_actions_" + str(args.dem_num) + ".pkl", "wb")
        pickle.dump(dic, p)
        p.close()
