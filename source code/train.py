import csv

from PRG_SB3 import *
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
from pyRDDLGym import RDDLEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from Experience_classify import SAC_EC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
import time
import sys
sys.setrecursionlimit(1000000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1000000, help='train num')
    parser.add_argument('-r', type=str, default='task9', help='rddl name')
    parser.add_argument('-i', type=str, default='inst1', help='inst name')
    parser.add_argument('-l', type=str, default='waterworld1', help='inst name')
    parser.add_argument('-log', type=str, default='task9', help='inst name')
    parser.add_argument('-a', type=float, default=0.75, help='sample alpha')
    parser.add_argument('-k', type=int, default=1, help='sample alpha')
    args = parser.parse_args()

    rddl_name = args.r
    inst_name = args.i
    train_num = args.n
    log = args.log
    l = args.l
    a = args.a
    log_inter = 20
    k = args.k
    domain = 'benchmarks/' + rddl_name + '/' + l + '.rddl'
    instance = 'benchmarks/' + rddl_name + '/' + inst_name + '.rddl'
    log_path = log + '/' + inst_name + '_' + str(k) + '/conv.csv'

    env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
    env = FlattenAction(env)
    model = SAC_EC("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                        learning_rate=3e-4, batch_size=256, train_freq=1, device='cpu')
    new_logger = configure(log + '/' + inst_name + '_' + str(k), ["csv"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=train_num, log_interval=log_inter)