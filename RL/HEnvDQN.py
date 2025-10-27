# HEnv_fixed.py
import gym
from gym import spaces
import numpy as np
import random
import math
from typing import List
from copy import deepcopy
from collections import defaultdict

from tool.log import Log
from backlog import *
from cadence.hardware import Hardware, hardware_debug

MAX_STEP = 1000
MAX_BACKLOG_SIZE = 50
log = Log(f"{__file__}.log", terminal=True)


# backlog = random_backlog(20)
# log.println(f"backlog: {backlog}")





class SystolicArrayEnv(gym.Env):
    """
    Discrete-action DVFS/backlog env compatible with DQN.
    Action: single Discrete(Nv * Nf) integer -> mapped to (voltage_index, freq_index)
    Observation: normalized finite 7-vector (safe for DQN).
    """

    def __init__(self):
        super().__init__()

        self.hardware = Hardware()

        self.frequency_levels = list(range(100, 1000+1, 100))

        self.Nf = len(self.frequency_levels)
        self.action_space = spaces.Discrete(self.Nf)

        # observation: ...
        low     = np.array([0.]*8, dtype=np.float32)
        high    = np.array([1.]*8, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # parameters
        self.max_backlog = 200
        self.max_step = MAX_STEP

        self.max_voltage = self.hardware.get_max_mapping_voltage()
        self.max_frequency = float(self.frequency_levels[-1])
        self.max_latency = self.hardware.get_max_mapping_delay()
        self.max_power = self.hardware.get_max_mapping_power() * self.max_frequency

        # internal state
        self.vdd = 0.0
        self.current_frequency = 0.0

        self.latency = 0.0
        self.power = 0.0

        self.backlog: Backlog = None
        self.backlog_size = 0
        self.backlog_not_active = 0
        self.backlog_ok         = 0
        self.backlog_linear     = 0
        self.backlog_crashed    = 0 

        self.prev_frequency = 0.0
        self.current_step = 0

        self.seed()

    def seed(self, s=None):
        random.seed(s)
        np.random.seed(s)


    def reset(self, insert_rate=None):
        self.current_step = 0
        
        self.hardware = Hardware()
        self.current_frequency = self.frequency_levels[0]
        self.prev_frequency = self.current_frequency
        self.vdd = self.hardware.get_vdd(self.current_frequency)

        self.backlog = random_backlog(random.randrange(0, 200))
        self.backlog_size       = self.backlog.get_active_size(self.current_step)
        self.backlog_not_active = self.backlog.get_status_size(self.current_step, "not_active")
        self.backlog_ok         = self.backlog.get_status_size(self.current_step, "ok")
        self.backlog_linear     = self.backlog.get_status_size(self.current_step, "linear")
        self.backlog_crashed    = self.backlog.get_status_size(self.current_step, "crash")

        self._update_derived()
        return self._get_obs()


    def step(self, action, inference=False):
        """
        action: integer in [0, Nv*Nf)
        mapped to (v_idx, f_idx)
        Order:
          1) interpret action -> set voltage/frequency
          2) process backlog using current frequency (before arrivals)
          3) add arrivals (stochastic around insert_rate)
          4) compute reward = (processed - arrived)/max_processed - power_penalty - switch_penalty
        """
        assert self.action_space.contains(action), "Invalid action"

        # map action int -> (v_idx, f_idx)
        # v_idx = action // self.Nf
        f_idx = action #% self.Nf
        # clamp indices
        # v_idx = int(np.clip(v_idx, 0, self.Nv - 1))
        f_idx = int(np.clip(f_idx, 0, self.Nf - 1))

        # apply action
        self.current_frequency = float(self.frequency_levels[f_idx])
        self._update_derived()

        self.backlog.render(self.current_step, self.current_frequency, 2000)
        self.backlog_size       = self.backlog.get_active_size(self.current_step)
        self.backlog_not_active = self.backlog.get_status_size(self.current_step, "not_active")
        self.backlog_ok         = self.backlog.get_status_size(self.current_step, "ok")
        self.backlog_linear     = self.backlog.get_status_size(self.current_step, "linear")
        self.backlog_crashed    = self.backlog.get_status_size(self.current_step, "crash")


        reward = 0

        reward -= self.backlog_linear * 0.02
        if self.backlog_crashed > 0:
            reward -= 7
        
        reward -= 0.5 * (self.power / (self.max_power + 1e-9))

        if inference:
            # log.println(
            #     hardware_debug(
            #         self.hardware, 
            #         self.current_step, 
            #         self.current_frequency, 
            #         self.vdd, 
            #         self.hardware.get_delay_power(self.vdd)
            #     )
            # )
            log.println(
                f"[{self.current_step}] ({reward:6.3f}), " +\
                f"[f:{self.current_frequency:6}] [f_max:{self.hardware.get_max_freq():.0f}], " +\
                f"[{self.backlog_size}] [{self.backlog_not_active}, {self.backlog_ok}, {self.backlog_linear}, {self.backlog_crashed}], "
            )

            
        # info handy for debugging/inference
        info = {
            "processed": 0,
            "arrived": 0,
            "f_applied": float(self.current_frequency),
            "power": float(self.power),
            # "backlog": int(self.backlog_running),
            "insert_rate": self.backlog_crashed,
            "max_freq": self.hardware.get_max_freq()
        }
        
        self.current_step += 1
        done = (self.current_step >= self.max_step) #or (self.backlog_size >= self.max_backlog)

        obs = self._get_obs()
        return obs, reward, done, info


    def _update_derived(self):
        # update latency & power for normalization / observation
        self.vdd = self.hardware.get_vdd(self.current_frequency)
        delay_power = self.hardware.get_delay_power(self.vdd)
        self.latency = delay_power[0]
        self.power = delay_power[1] * self.current_frequency

        step_time_factor = 100
        t0 = self.current_step * step_time_factor #seconds
        t1 = (self.current_step + 1) * step_time_factor
        t0, t1 = 10*t0, 10*t1   #each steo as 10 seconds
        # self.hardware.apply_aging(self.vdd, t0, t1)

    def _get_obs(self):
        # normalized observation vector, clipped into [0,1]
        obs = np.array([
            float(self.vdd) / (self.max_voltage + 1e-9),
            float(self.current_frequency) / (self.max_frequency + 1e-9),
            
            float(self.latency) / (self.max_latency + 1e-9),
            float(self.power) / (self.max_power + 1e-9),
            
            float(self.backlog_size)        / (self.max_backlog + 1e-9),
            # float(self.backlog_not_active)  / (self.max_backlog + 1e-9),
            float(self.backlog_ok)          / (self.max_backlog + 1e-9),
            float(self.backlog_linear)      / (self.max_backlog + 1e-9),
            float(self.backlog_crashed)     / (self.max_backlog + 1e-9),
            
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def render(self, mode='human'):
        print(f"step {self.current_step:4d} freq {self.current_frequency:4.0f}MHz backlog {self.backlog_size:4d} insert {self.insert_rate} proc {self.processed} power {self.power:.2f}")
