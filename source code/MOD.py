import argparse
import copy
import csv
import os

from stable_baselines3 import *
from stable_baselines3.common.base_class import maybe_make_env, BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import SelfOffPolicyAlgorithm
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from copy import deepcopy
from ReplayBuffer import RBC_Replay_Buffer
import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from PRG_SB3 import *
from pyRDDLGym import RDDLEnv
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from MySAC4 import MySAC
from MyEnv1 import MyEnv
from MyTD3 import MyTD3
time_start = time.time()

class SACOX:
    def __init__(self, domain, eval_domain, instance, classify_num, term_num, log_path, model_class=MySAC, action_noise=None):
        self.models = [None for _ in range(classify_num)]
        env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain, instance=instance))
        self.models[0] = model_class("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                                 learning_rate=3e-4, batch_size=256, train_freq=1, action_noise=action_noise,
                                 buffer_size=300000, classify_num=classify_num, term_num=term_num, device='cpu')
        self.state_list = []
        for i in range(1, classify_num):
            env = FlattenAction(MyEnv(domain=domain, instance=instance, state_list=self.state_list))
            self.models[i] = model_class("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                                        learning_rate=3e-4, batch_size=256, train_freq=1, action_noise=action_noise,
                                        buffer_size=300000, classify_num=classify_num, term_num=term_num, device='cpu')
        eval_env = FlattenAction(RDDLEnv.RDDLEnv(domain=eval_domain, instance=instance))
        self.eval_env = BaseAlgorithm._wrap_env(maybe_make_env(eval_env, 1), 1, True)
        self.eval_interval = 1000
        self.eval_time = 5
        self.current_dfa = 0
        self.current_state = None
        self.train_steps = 0
        self.classify_num = classify_num
        self.num_per_class = self.models[0].num_per_class
        self.log_path = log_path
        self.rollout_reward = []
        self.rollout_length = []
    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        # callback = [None for _ in range(self.classify_num)]
        for i in range(self.classify_num):
            total_timesteps, callback = self.models[i]._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )
        while self.train_steps < total_timesteps:
            dfa = self.current_dfa
            rollout, next_dfa_state, new_states, infos, dones = self.models[dfa].collect_rollouts(self.models[dfa].env, callback=callback,
                                                  train_freq=self.models[dfa].train_freq,
                                                  replay_buffer=self.models[dfa].replay_buffer,
                                                  action_noise=self.models[dfa].action_noise,
                                                  learning_starts=self.models[dfa].learning_starts,
                                                  log_interval=log_interval, state=self.current_state, dfa=dfa)
            if next_dfa_state != self.current_dfa and next_dfa_state < self.classify_num:
                # print(next_dfa_state)
                self.current_dfa = next_dfa_state
                self.current_state = new_states
                if self.current_dfa != 0:
                    self.state_list.append(new_states)
            else:
                self.current_state = None
            if self.models[dfa].num_timesteps > 0 and self.models[dfa].num_timesteps > self.models[dfa].learning_starts:
                gradient_steps = self.models[dfa].gradient_steps if self.models[dfa].gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.models[dfa].batch_size, gradient_steps=gradient_steps, dfa=dfa)

            if (self.train_steps + 1) % self.eval_interval == 0:
                self.evaluate_agent(self.train_steps + 1)

            if dones:
                self.rollout_reward.append(infos[0]['episode']['r'])
                self.rollout_length.append(infos[0]['episode']['l'])
            if len(self.rollout_reward) == log_interval:
                data = [
                    [self.train_steps, np.mean(self.rollout_reward), np.mean(self.rollout_length)],
                ]
                with open(self.log_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows(data)
                self.rollout_reward = []
                self.rollout_length = []
            self.train_steps += 1

    def predict1(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            dfa: int = 0,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.models[dfa].policy.set_training_mode(False)

        observation, vectorized_env = self.models[dfa].policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.models[dfa].actor_target(observation, deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.models[dfa].policy.action_space.shape))

        if isinstance(self.models[dfa].policy.action_space, spaces.Box):
            if self.models[dfa].policy.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.models[dfa].policy.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.models[dfa].policy.action_space.low, self.models[dfa].policy.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def evaluate_agent(self, timestep):
        mean_reward, mean_length = 0, 0
        for times in range(self.eval_time):
            state = self.eval_env.reset()
            done = False
            while not done:
                dfa = int(state['as'].item() / self.num_per_class)
                action, _ = self.predict1(state, deterministic=True, dfa=dfa)
                next_state, reward, done, info = self.eval_env.step(action)
                state = next_state
                mean_reward += reward
                mean_length += 1
        time_now = time.time()
        seconds = time_now - time_start
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        hour = str(hours) + "h" + str(minutes) + "min" + str(second) + 's'
        print('time:', hours, "h", minutes, "min", second, 's')
        print('n_step:', timestep+1, 'mean_reward:', mean_reward/self.eval_time, 'mean_length:', mean_length/self.eval_time)

    def train(self, gradient_steps: int, batch_size: int = 64, dfa: int = 0) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.models[dfa].policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.models[dfa].actor.optimizer, self.models[dfa].critic.optimizer]
        if self.models[dfa].ent_coef_optimizer is not None:
            optimizers += [self.models[dfa].ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self.models[dfa]._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data, nums = self.models[dfa].replay_buffer.sample(batch_size,
                                                                env=self.models[dfa]._vec_normalize_env)  # type: ignore[union-attr]
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.models[dfa].use_sde:
                self.models[dfa].actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.models[dfa].actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.models[dfa].ent_coef_optimizer is not None and self.models[dfa].log_ent_coef is not None:
                ent_coef = th.exp(self.models[dfa].log_ent_coef.detach())
                ent_coef_loss = -(self.models[dfa].log_ent_coef * (log_prob + self.models[dfa].target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.models[dfa].ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.models[dfa].ent_coef_optimizer is not None:
                self.models[dfa].ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.models[dfa].ent_coef_optimizer.step()

            target_q_values = th.tensor([], device=self.models[dfa].device)
            index = 0
            for i in range(self.models[dfa].classify_num):
                if nums[i] == 0:
                    continue
                next_observations = {}
                for key, value in replay_data.next_observations.items():
                    next_observations[key] = value[index:index + nums[i]]
                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.models[i].actor_target.action_log_prob(next_observations)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.models[i].critic_target(next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    target_q_values1 = replay_data.rewards[index:index + nums[i]] + (
                                1 - replay_data.dones[index:index + nums[i]]) * self.models[dfa].gamma * next_q_values
                    target_q_values = th.cat((target_q_values, target_q_values1), dim=0)
                index += nums[i]

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.models[dfa].critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.models[dfa].critic.optimizer.zero_grad()
            critic_loss.backward()
            self.models[dfa].critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.models[dfa].critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.models[dfa].actor.optimizer.zero_grad()
            actor_loss.backward()
            self.models[dfa].actor.optimizer.step()

            # Update target networks
            if gradient_step % self.models[dfa].target_update_interval == 0:
                polyak_update(self.models[dfa].critic.parameters(), self.models[dfa].critic_target.parameters(),
                              self.models[dfa].tau)
                polyak_update(self.models[dfa].actor.parameters(),
                              self.models[dfa].actor_target.parameters(), self.models[dfa].tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.models[dfa].batch_norm_stats, self.models[dfa].batch_norm_stats_target, 1.0)

        self.models[dfa]._n_updates += gradient_steps
class TD3OX:
    def __init__(self, domain, eval_domain, instance, classify_num, term_num, log_path, model_class=MyTD3, action_noise=None):
        self.models = [None for _ in range(classify_num)]
        env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain, instance=instance))
        self.models[0] = model_class("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                                 learning_rate=3e-4, batch_size=256, train_freq=1, action_noise=action_noise,
                                 buffer_size=300000, classify_num=classify_num, term_num=term_num, device='cpu')
        self.state_list = []
        for i in range(1, classify_num):
            env = FlattenAction(MyEnv(domain=domain, instance=instance, state_list=self.state_list))
            self.models[i] = model_class("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                                        learning_rate=3e-4, batch_size=256, train_freq=1, action_noise=action_noise,
                                        buffer_size=300000, classify_num=classify_num, term_num=term_num, device='cpu')
        eval_env = FlattenAction(RDDLEnv.RDDLEnv(domain=eval_domain, instance=instance))
        self.eval_env = BaseAlgorithm._wrap_env(maybe_make_env(eval_env, 1), 1, True)
        self.eval_interval = 1000
        self.eval_time = 5
        self.current_dfa = 0
        self.current_state = None
        self.train_steps = 0
        self.classify_num = classify_num
        self.num_per_class = self.models[0].num_per_class
        self.log_path = log_path
        self.rollout_reward = []
        self.rollout_length = []
        self.current_length = 0

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        # callback = [None for _ in range(self.classify_num)]
        for i in range(self.classify_num):
            total_timesteps, callback = self.models[i]._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )
        while self.train_steps < total_timesteps:
            dfa = self.current_dfa
            rollout, next_dfa_state, new_states, infos, dones = self.models[dfa].collect_rollouts(self.models[dfa].env, callback=callback,
                                                  train_freq=self.models[dfa].train_freq,
                                                  replay_buffer=self.models[dfa].replay_buffer,
                                                  action_noise=self.models[dfa].action_noise,
                                                  learning_starts=self.models[dfa].learning_starts,
                                                  log_interval=log_interval, state=self.current_state, dfa=dfa)
            self.current_length += 1
            if next_dfa_state != self.current_dfa and next_dfa_state < self.classify_num:
                # print(next_dfa_state)
                self.current_dfa = next_dfa_state
                self.current_state = new_states
                if self.current_dfa != 0:
                    self.state_list.append(new_states)
            else:
                self.current_state = None
            if self.models[dfa].num_timesteps > 0 and self.models[dfa].num_timesteps > self.models[dfa].learning_starts:
                gradient_steps = self.models[dfa].gradient_steps if self.models[dfa].gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.models[dfa].batch_size, gradient_steps=gradient_steps, dfa=dfa)

            '''if (self.train_steps + 1) % self.eval_interval == 0:
                t = threading.Thread(target=self.evaluate_agent, args=(self.train_steps,))
                t.start()'''
            if dones:
                self.rollout_reward.append(infos[0]['episode']['r'])
                self.rollout_length.append(self.current_length)
                self.current_length = 0
            if len(self.rollout_reward) == log_interval:
                data = [
                    [self.train_steps, np.mean(self.rollout_reward), np.mean(self.rollout_length)],
                ]
                with open(self.log_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows(data)
                self.rollout_reward = []
                self.rollout_length = []
            self.train_steps += 1

    def predict1(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            dfa: int = 0,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.models[dfa].policy.set_training_mode(False)

        observation, vectorized_env = self.models[dfa].policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.models[dfa].actor_target(observation, deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.models[dfa].policy.action_space.shape))

        if isinstance(self.models[dfa].policy.action_space, spaces.Box):
            if self.models[dfa].policy.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.models[dfa].policy.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.models[dfa].policy.action_space.low, self.models[dfa].policy.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def evaluate_agent(self, timestep):
        mean_reward, mean_length = 0, 0
        for times in range(self.eval_time):
            state = self.eval_env.reset()
            done = False
            while not done:
                dfa = int(state['as'].item() / self.num_per_class)
                action, _ = self.predict1(state, deterministic=True, dfa=dfa)
                next_state, reward, done, info = self.eval_env.step(action)
                state = next_state
                mean_reward += reward
                mean_length += 1
        time_now = time.time()
        seconds = time_now - time_start
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        hour = str(hours) + "h" + str(minutes) + "min" + str(second) + 's'
        data = [
            [timestep + 1, hour, mean_reward / self.eval_time, mean_length / self.eval_time],
        ]
        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)
        print('time:', hours, "h", minutes, "min", second, 's')
        print('n_step:', timestep+1, 'mean_reward:', mean_reward/self.eval_time, 'mean_length:', mean_length/self.eval_time)

    def train1(self, gradient_steps: int, batch_size: int = 64, dfa: int = 0) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.models[dfa].policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.models[dfa].actor.optimizer, self.models[dfa].critic.optimizer]
        if self.models[dfa].ent_coef_optimizer is not None:
            optimizers += [self.models[dfa].ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self.models[dfa]._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data, nums = self.models[dfa].replay_buffer.sample(batch_size,
                                                                env=self.models[dfa]._vec_normalize_env)  # type: ignore[union-attr]
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.models[dfa].use_sde:
                self.models[dfa].actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.models[dfa].actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.models[dfa].ent_coef_optimizer is not None and self.models[dfa].log_ent_coef is not None:
                ent_coef = th.exp(self.models[dfa].log_ent_coef.detach())
                ent_coef_loss = -(self.models[dfa].log_ent_coef * (log_prob + self.models[dfa].target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.models[dfa].ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.models[dfa].ent_coef_optimizer is not None:
                self.models[dfa].ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.models[dfa].ent_coef_optimizer.step()

            target_q_values = th.tensor([], device=self.models[dfa].device)
            index = 0
            for i in range(self.models[dfa].classify_num):
                if nums[i] == 0:
                    continue
                next_observations = {}
                for key, value in replay_data.next_observations.items():
                    next_observations[key] = value[index:index + nums[i]]
                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.models[i].actor_target.action_log_prob(next_observations)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.models[i].critic_target(next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    target_q_values1 = replay_data.rewards[index:index + nums[i]] + (
                                1 - replay_data.dones[index:index + nums[i]]) * self.models[dfa].gamma * next_q_values
                    target_q_values = th.cat((target_q_values, target_q_values1), dim=0)
                index += nums[i]

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.models[dfa].critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.models[dfa].critic.optimizer.zero_grad()
            critic_loss.backward()
            self.models[dfa].critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.models[dfa].critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.models[dfa].actor.optimizer.zero_grad()
            actor_loss.backward()
            self.models[dfa].actor.optimizer.step()

            # Update target networks
            if gradient_step % self.models[dfa].target_update_interval == 0:
                polyak_update(self.models[dfa].critic.parameters(), self.models[dfa].critic_target.parameters(),
                              self.models[dfa].tau)
                polyak_update(self.models[dfa].actor.parameters(),
                              self.models[dfa].actor_target.parameters(), self.models[dfa].tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.models[dfa].batch_norm_stats, self.models[dfa].batch_norm_stats_target, 1.0)

        self.models[dfa]._n_updates += gradient_steps

    def train(self, gradient_steps: int, batch_size: int = 100, dfa: int = 0) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.models[dfa].policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self.models[dfa]._update_learning_rate([self.models[dfa].actor.optimizer, self.models[dfa].critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self.models[dfa]._n_updates += 1
            # Sample replay buffer
            replay_data, nums = self.models[dfa].replay_buffer.sample(batch_size,
                                                                      env=self.models[dfa]._vec_normalize_env)

            target_q_values = th.tensor([], device=self.models[dfa].device)
            index = 0
            for i in range(self.models[dfa].classify_num):
                if nums[i] == 0:
                    continue
                next_observations = {}
                for key, value in replay_data.next_observations.items():
                    next_observations[key] = value[index:index + nums[i]]
                with th.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = replay_data.actions[index:index + nums[i]].clone().data.normal_(0, self.models[dfa].target_policy_noise)
                    noise = noise.clamp(-self.models[dfa].target_noise_clip, self.models[dfa].target_noise_clip)
                    next_actions = (self.models[dfa].actor_target(next_observations) + noise).clamp(-1, 1)

                    # Compute the next Q-values: min over all critics targets
                    next_q_values = th.cat(self.models[dfa].critic_target(next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_q_values1 = replay_data.rewards[index:index + nums[i]] + (1 - replay_data.dones[index:index + nums[i]]) * self.models[dfa].gamma * next_q_values
                    target_q_values = th.cat((target_q_values, target_q_values1), dim=0)
                index += nums[i]

            # Get current Q-values estimates for each critic network
            current_q_values = self.models[dfa].critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.models[dfa].critic.optimizer.zero_grad()
            critic_loss.backward()
            self.models[dfa].critic.optimizer.step()

            # Delayed policy updates
            if self.models[dfa]._n_updates % self.models[dfa].policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.models[dfa].critic.q1_forward(replay_data.observations, self.models[dfa].actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.models[dfa].actor.optimizer.zero_grad()
                actor_loss.backward()
                self.models[dfa].actor.optimizer.step()

                polyak_update(self.models[dfa].critic.parameters(), self.models[dfa].critic_target.parameters(), self.models[dfa].tau)
                polyak_update(self.models[dfa].actor.parameters(), self.models[dfa].actor_target.parameters(), self.models[dfa].tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.models[dfa].critic_batch_norm_stats, self.models[dfa].critic_batch_norm_stats_target, 1.0)
                polyak_update(self.models[dfa].actor_batch_norm_stats, self.models[dfa].actor_batch_norm_stats_target, 1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='log2/task5/', help='log path')
    parser.add_argument('-i', type=str, default='inst1', help='inst name')
    parser.add_argument('-k', type=int, default=4, help='inst name')
    args = parser.parse_args()
    domain = 'benchmarks/cartpole2/cartpole4.rddl'
    eval_domain = domain
    instance = 'benchmarks/cartpole2/' + args.i + '.rddl'
    log_path = args.log + args.i + '_' + str(args.k) + '.csv'
    data = [
        ['n_steps', 'mean_reward', 'mean_length'],
    ]
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(log_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain,
                                        instance=instance))
    eval_env = FlattenAction(RDDLEnv.RDDLEnv(domain=eval_domain,
                                             instance=instance))


    agent = SACOX(domain, eval_domain, instance, classify_num=5, term_num=1, log_path=log_path)

    agent.learn(total_timesteps=200000)