import copy
import sys
import time
from copy import deepcopy
from numpy import ndarray
from stable_baselines3 import *
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.buffers import *
from stable_baselines3.td3.policies import TD3Policy
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from ReplayBuffer import RBC_Replay_Buffer
import numpy as np
import torch as th
from collections import deque
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

class MyTD3(TD3):

    def __init__(
            self,
            policy: Union[str, Type[TD3Policy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 100,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = RBC_Replay_Buffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            sample_alpha: float = 1,
            classify_num: int = 0,
            term_num: int = 2,
    ):

        self.automata_states = env.env.observation_space['as'].n - 1
        self.horizon = env.env.horizon
        self.episode_reward = 0
        self.episode_length = 0
        if classify_num == 0:
            self.classify_num = self.automata_states - term_num
        else:
            self.classify_num = classify_num
        self.num_per_class = (self.automata_states - term_num) / self.classify_num

        # 调用父类的__init__方法，从而继承其属性和方法
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq,
                         gradient_steps, action_noise, replay_buffer_class, {"automata_states": self.automata_states},
                         optimize_memory_usage, policy_delay, target_policy_noise, target_noise_clip, stats_window_size,
                         tensorboard_log, policy_kwargs,
                         verbose, seed, device, _init_setup_model)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _store_transition(
            self,
            replay_buffer: RBC_Replay_Buffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        # print("start", self._last_obs)
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        add_state = int(new_obs['as'].item() / self.num_per_class)
        if add_state >= self.classify_num:
            add_state = self.classify_num - 1
        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
            add_state=add_state,
        )
        self._last_obs = new_obs
        # print("end", self._last_obs)
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        state = None,
        dfa: int = 0,
    ):
        # reset env if it is needed
        if state is not None:
            state = env.reset()
            assert int(state['as'].item() / self.num_per_class) == dfa
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action

            new_obs, rewards, dones, infos = env.step(actions)

            real_rewards = rewards
            rewards = rewards + (new_obs['as'] - self._last_obs['as']) * 10
            next_dfa_state = int(new_obs['as'].item() / self.num_per_class)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            # self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        pass
                        #print("dfa state is", dfa_state)
                        #self._dump_logs()

        callback.on_rollout_end()
        if dones:
            next_dfa_state = 0
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training), next_dfa_state, new_obs, infos, dones