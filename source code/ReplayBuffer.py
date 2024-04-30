from stable_baselines3.common.buffers import *
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples


class RBC_Replay_Buffer(DictReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            automata_states: int = 3,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination)
        self.automata_states = automata_states
        self.state_full = np.array([False for _ in range(self.automata_states)])
        self.state_pos = np.array([i * (buffer_size // self.automata_states) for i in range(self.automata_states)])
        self.lower_bound = np.array([self.buffer_size // self.automata_states * add_state for add_state in range(self.automata_states)])

    def add(
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            state_exp_num: np.ndarray = np.array([]),
            add_state: int = 0
    ) -> None:
        # calculate pos
        pos = self.state_pos[add_state]
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][pos] = np.array(next_obs[key]).copy()

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[pos] = np.array(action).copy()
        self.rewards[pos] = np.array(reward).copy()
        self.dones[pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.state_pos[add_state] += 1
        if self.state_pos[add_state] == self.buffer_size // self.automata_states * (add_state + 1):
            self.state_full[add_state] = True
            self.state_pos[add_state] = self.buffer_size // self.automata_states * add_state

    def sample(
            self,
            batch_size: int,
            env: Optional[VecNormalize] = None,
    ):
        upper_bound = np.array([(self.buffer_size // self.automata_states * (state + 1) if self.state_full[
            state] else self.state_pos[state]) for state in range(self.automata_states)])
        dfa_state_num = upper_bound - self.lower_bound
        prob = dfa_state_num / np.sum(dfa_state_num)
        chosen_state_num = np.zeros(self.automata_states, dtype=int)
        for _ in range(batch_size):
            chosen_state_num[np.random.choice(a=range(self.automata_states), p=prob)] += 1
        batch_inds = np.array([], dtype=int)
        i = 0
        for num in chosen_state_num:
            a = np.random.randint(self.lower_bound[i],
                                  upper_bound[i], size=num)
            batch_inds = np.concatenate((batch_inds, a), axis=0)
            i += 1
        return self._get_samples(batch_inds, env=env), chosen_state_num

class Clem_Replay_Buffer(DictReplayBuffer):
    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        parallel_num: int = 1,
    ) -> None:  # pytype: disable=signature-mismatch
        # Copy to avoid modification by reference
        for i in range(parallel_num):
            for key in self.observations.keys():
                # Reshape needed when using multiple envs with discrete observations
                # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
                self.observations[key][i] = np.array(obs[key][i])

            for key in self.next_observations.keys():
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
                self.next_observations[key][i] = np.array(next_obs[key][i]).copy()

            # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
            action = action.reshape((self.n_envs, self.action_dim))

            self.actions[i] = np.array(action[i]).copy()
            self.rewards[i] = np.array(reward[i]).copy()
            self.dones[i] = np.array(done[i]).copy()


    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
        batch_inds = np.array(range(batch_size))
        buffer = self._get_samples(batch_inds, env=env)
        return buffer
