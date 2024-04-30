# Experience-Classification
Experience Classification combining prioritized experience replay with automatic reward shaping achieves successful results in solving non-Markovian tasks.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install git+https://github.com/DLR-RM/stable-baselines3
```
Install all toolkits in your environment.

## Usage
run train.py for training models using EC, the running example is as follows：
```python
from PRG_SB3 import *
from pyRDDLGym import RDDLEnv
from EC import SAC_EC
from stable_baselines3.common.logger import configure
import sys
sys.setrecursionlimit(1000000)

if __name__ == '__main__':
    domain = "benchmarks/waterworld3/waterworld3.rddl" # path for domain
    instance = "benchmarks/waterworld3/inst21.rddl" # path for instance
    log_path = "progress.csv" # path for log

    env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
    env = FlattenAction(env)
    model = SAC_EC("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                        learning_rate=3e-4, batch_size=256, train_freq=1, device='cpu')  
    # SAC_EC can be changed by other algorithms in EC/RS/PER/MOD 
    new_logger = configure(log_path, ["csv"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=500000, log_interval=100) # set training steps and log interval
```
