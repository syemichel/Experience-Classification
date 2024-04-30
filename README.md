# Experience-Classification
Experience Classification combining prioritized experience replay with automatic reward shaping achieves successful results in solving non-Markovian tasks. Our framework is shown as follows:
![image](https://github.com/syemichel/Experience-Classification/blob/main/framework.png)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install stablebaseline.

```bash
pip install stable-baselines3==2.2.1
```
Install all toolkits in your environment.

## Usage
run train.py in the source code for training models using EC, the running example is as follows:
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
run generate_trans.py in the source code for automatically generating dfa informations in rddl, the running example is as follows:
```python
from classify import *

# Example usage
file_path = 'dot.txt'  # write in transition information for dfa in the dot file
trans_express = 2
# We achieve two conversion formats, defaulting to the second one, your can use the first one by changing it to 1
with open(file_path, 'r') as file:
    lines = [line.strip() for line in file]
classified_result = classify_lines(lines)
# To print or use the classified result
if trans_express == 2:
    converted_transitions = [convert_transition_with_regex(transition) for transition in lines]
    # 输出结果
    for converted in converted_transitions:
        print(converted)
graph = {}
for destination, lines in classified_result.items():
    trans = []
    for line in lines:
        new_tuple = generate_transitions(line)
        if str(new_tuple[0]) not in graph:
            graph[str(new_tuple[0])] = []
        graph[str(new_tuple[0])].append(str(new_tuple[1]))
        trans.append(new_tuple)
    if trans_express == 1:
        print(generate_expression(trans))

classify_dfa_states(graph, accepted_state='145', error_state='', num=14)
# set the accepted_state and the classification number
generate_reward_function(num=14, interval=7)
# generate reward functions
```
