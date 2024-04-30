import copy
from collections import OrderedDict
import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.spaces import *
from abc import ABCMeta, abstractmethod
import random
import numpy as np

class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def take_action(self, state):
        pass


class Agent(BaseAgent):
    def __init__(self, action_space, num_actions=1):
        self.action_space = action_space
        self.num_actions = num_actions

    def take_action(self, s, state=None):
        action = {}
        selected_actions = random.sample(list(s), self.num_actions)  # problem!!!!!!!
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
        return action


class FlattenAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)

        print(self.action_space)

    def action(self, act):
        act = spaces.unflatten(self.env.action_space, act)
        agent = Agent(action_space=self.env.action_space, num_actions=self.env.numConcurrentActions)
        return agent.take_action(act)


class BoxContinuousAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.l = []
        for k, v in self.action_space.items():
            self.l.append(k)
        self.new_action_space= copy.deepcopy(self.env.action_space)

        for i in range(self.env.numConcurrentActions):
            self.new_action_space[str(i)] = Box(0, len(self.env.action_space) - 0.00001, shape=(1, 1))
        self.action_space = spaces.flatten_space(self.new_action_space)

        print(self.action_space)

    def action(self, act):

        act = spaces.unflatten(self.new_action_space, act)
        act_dic = {}
        for i in range(self.env.numConcurrentActions):
            value = int(act[str(i)])
            action = self.l[value]
            action_value = float(act[action])
            act_dic[action] = action_value
        agent = Agent(action_space=self.env.action_space, num_actions=self.env.numConcurrentActions)
        return act_dic

class CartpoleActionWapper(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = Discrete(2)
        print(self.action_space)

    def action(self, act):
        dic = {}
        dic['force-side'] = act
        # print(dic)
        return dic

class WaterworldActionWapper(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = Box(low=-35, high=35, shape=(2,), dtype=int)
        print(self.action_space)
    def action(self, act):
        dic = {}
        dic['ag-move___x'] = act[0]
        dic['ag-move___y'] = act[1]
        # print(dic)
        return dic


class BoxDiscreteAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.l = []
        for k, v in self.action_space.items():
            self.l.append(k)
        self.action_space = Box(low=0, high=len(self.env.action_space) -1, shape=(self.env.numConcurrentActions,), dtype=np.int)
        print(self.action_space)

    def action(self, act):
        dic = {}
        for i in range(self.env.numConcurrentActions):
            dic[self.l[act[i]]] = 1
        #print(dic)
        return dic

class MultiDiscreteAction(gym.ActionWrapper):
    # support concurrent actions
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.l = []
        for k, v in self.action_space.items():
            self.l.append(k)

        matrix = np.full(self.env.numConcurrentActions, len(self.env.action_space))

        self.action_space = MultiDiscrete(matrix)
        print(self.action_space)

    def action(self, act):
        dic = {}
        for i in range(self.env.numConcurrentActions):
                dic[self.l[act[i]]] = 1
        # print(dic)
        return dic

class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = flatten_space(env.observation_space)

    def observation(self, obs):
        return flatten(self.env.observation_space, obs)