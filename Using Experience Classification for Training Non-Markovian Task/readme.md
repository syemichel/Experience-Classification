# Experience classification (EC)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install pyRDDLGym
pip install git+https://github.com/DLR-RM/stable-baselines3
pip install gymnasium
```
## Make pyRDDLGym compatible with SB3
### Change PRG
in /pyRDDLGym/Core/Env/RDDLEnv.py：
```bash
return self.state, 0.0, self.done,self.done, {}# line199
return obs,reward, self.done, self.done,{}# line251
return obs, {} # line286
```
in /pyRDDLGym/Core/Simulator/RDDLSimulator.py：
```bash
def step(self, actions: Args) -> Args: # line 308
        '''Samples and returns the next state from the CPF expressions.

        :param actions: a dict mapping current action fluent to their values
        '''
        rddl = self.rddl
        actions = self._process_actions(actions)
        subs = self.subs
        subs.update(actions)

        # evaluate CPFs in topological order
        for (cpf, expr, dtype) in self.cpfs:
            sample = self._sample(expr, subs)
            RDDLSimulator._check_type(sample, dtype, cpf, expr)
            subs[cpf] = sample

        # evaluate reward
        #reward = self.sample_reward()

        # update state
        self.state = {}
        for (state, next_state) in rddl.next_state.items():
            subs[state] = subs[next_state]
            self.state.update(rddl.ground_values(state, subs[state]))

        # update observation
        if self._pomdp:
            obs = {}
            for var in rddl.observ:
                obs.update(rddl.ground_values(var, subs[var]))
        else:
            obs = self.state
            # evaluate CPFs in topological order
        for (cpf, expr, dtype) in self.cpfs:
            sample = self._sample(expr, subs)
            RDDLSimulator._check_type(sample, dtype, cpf, expr)
            subs[cpf] = sample
        reward = self.sample_reward()
        done = self.check_terminal_states()
        return obs, reward, done
```
in /stable-baselines3/common/vec_env/dummy_vec_env.py
```bash
self.buf_obs[key][env_idx] = obs[0][key] #line 116
```
in /shimmy/openai_gym_compatibility.py
```bash
obs, reward, done, done1, info = self.gym_env.step(action) #line 256
```
in gymnasium/core.py
```bash
return self.env.reset()
```
change "import gym" to "import gymnasium as gym" in the source code of pyRDDLGy
## Usage
run train.py for training models using EC, the running example is as follows：
```python
from PRG_SB3 import *
from stable_baselines3.common.evaluation import evaluate_policy
from Experience_classify import *
import argparse
from pyRDDLGym import RDDLEnv
from sb3_contrib import *
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from Experience_classify2 import *
from stable_baselines3.common.callbacks import EvalCallback
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=200000, help='train num')
    parser.add_argument('-r', type=str, default='waterworld3', help='rddl name')
    parser.add_argument('-i', type=str, default='inst11', help='inst name')
    parser.add_argument('-l', type=str, default='waterworld2_2', help='inst name')
    args = parser.parse_args()

    rddl_name = args.r
    inst_name = args.i
    total_train = args.n
    l = args.l
    log_inter = 100
    env = RDDLEnv.RDDLEnv(domain=('benchmarks/' + rddl_name + '/' + l + '.rddl'),
                          instance=('benchmarks/' + rddl_name + '/' + inst_name + '.rddl'))

    env = FlattenAction(env)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = SAC("MultiInputPolicy", env, verbose=1,  learning_starts=1000,
                learning_rate=3e-4, batch_size=64, tensorboard_log='log', train_freq=1,)
    model.learn(total_timesteps=total_train, log_interval=log_inter, tb_log_name='TD3_' + l, )

    model.save("model/racecar")
    log = evaluate_policy(model, env, n_eval_episodes=30, deterministic=True)
    print(log)
```
