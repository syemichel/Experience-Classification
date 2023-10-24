import gym
import gym.spaces as spaces
from gym.spaces import *

from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent
from abc import ABCMeta, abstractmethod
import random
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch
from pyRDDLGym import RDDLEnv
from PRG_SB3 import *
from get_plan import *
from Experience_classify import *

if __name__ == "__main__":
    rddl_name = 'waterworld3'
    inst_name = 'inst21'
    env = RDDLEnv.RDDLEnv(domain=('benchmarks/' + rddl_name + '/' + "waterworld2" + '.rddl'),
                          instance=('benchmarks/' + rddl_name + '/' + inst_name + '.rddl'))

    env = FlattenAction(env)
    i = 0
    T = 0

    for i in range(1):

        obs = env.reset()
        #print(obs)
        total_reward = 0

        done = False
        while not done:
            action = env.action_space.sample()
            obs, rewards, done, done1, info = env.step(action)

            print(rewards)
            total_reward += rewards
            T += 1

        print(total_reward)