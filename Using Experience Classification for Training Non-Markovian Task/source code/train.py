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
    start_time = time.time()
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
    end_time = time.time()

    run_time = end_time - start_time

    print("代码运行时间为：", run_time, "秒")
