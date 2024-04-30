import re
import time

from PRG_SB3 import *
import numpy as np
from pyRDDLGym import RDDLEnv
class MyEnv(RDDLEnv.RDDLEnv):

    def __init__(self, domain, instance, state_list):
        super(MyEnv, self).__init__(domain, instance)
        self.state_list = state_list
    def reset(self):
        self.total_reward = 0
        self.currentH = 0
        if len(self.state_list) > 0:
            self.obs_to_init_values(self.state_list[0])
            self.state_list.pop()
        obs, self.done = self.sampler.reset()
        self.state = self.sampler.states
        return obs, {}

    def obs_to_init_values(self, obs):
        last_key = ""
        for key, value in obs.items():
            init_key = re.sub("__.*", "", key)
            if init_key != last_key:
                i = 0
            init_value = self.sampler.init_values[init_key]
            if isinstance(init_value, np.ndarray):
                shape = init_value.shape
                index = np.unravel_index(i, shape)
                init_value[index] = value.item()
            else:
                value1 = value.item()
                if isinstance(self.sampler.init_values[init_key], bool):
                    value1 = bool(value1)
                self.sampler.init_values[init_key] = value1
            i += 1
            last_key = init_key

