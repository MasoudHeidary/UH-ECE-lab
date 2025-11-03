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


class TransformerModel:
    def __init__(self):
        self.model = [
            {'flops': 18054400000 * 1E3, 'loss': 4.973566660360128},
            {'flops': 28054400000 * 1E3, 'loss': 2.973566660360128},
        ]

    def get_inference_flops(self, index):
        return self.model[index]["flops"]
    
    def get_max_flops(self):
        max_flops = 0
        for model in self.model:
            max_flops = max(model['flops'], max_flops)
        return max_flops
    

class SystolicArrayEnv(gym.Env):
    """
    Discrete-action DVFS/backlog env compatible with DQN.
    Action: single Discrete(Nv * Nf) integer -> mapped to (voltage_index, freq_index)
    Observation: normalized finite 7-vector (safe for DQN).
    """

    def __init__(self):
        super().__init__()

        self.hardware = Hardware()
        self.transformer = TransformerModel()

        self.freq_levels = list(range(0, 1000+1, 100))
        self.tran_levels = list(range(0, len(self.transformer.model)))

        self.Nf = len(self.freq_levels)
        self.Nt = len(self.tran_levels)
        self.action_space = spaces.Discrete(self.Nf * self.Nt)

        # bound parameters
        low     = np.array([0.]*8, dtype=np.float32)
        high    = np.array([1.]*8, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # maximum parameters
        self.max_step       = MAX_STEP
        self.max_backlog    = MAX_BACKLOG_SIZE

        self.max_voltage    = self.hardware.get_max_mapping_voltage()
        self.max_freq       = float(self.freq_levels[-1])
        self.max_inference_flops      = self.transformer.get_max_flops()
        self.max_latency    = self.hardware.get_max_mapping_delay()
        self.max_power      = self.hardware.get_max_mapping_power() * self.max_freq

        # action states
        self.freq   = 0.0   # current frequency
        self.inference_flops  = 0.0 # current transformer model flops

        # derived variables
        self.vdd    = 0.0
        self.latency    = 0.0
        self.power      = 0.0

        self.backlog: Backlog   = None
        # self.backlog_size       = 0
        self.backlog_not_active = 0
        self.backlog_ok         = 0
        self.backlog_linear     = 0
        self.backlog_crashed    = 0 

        self.prev_freq     = 0.0
        self.curr_step       = 0
        self.seed()

    def seed(self, s=None):
        random.seed(s)
        np.random.seed(s)

    def reset(self, insert_rate=None):
        self.curr_step = 0
        
        self.hardware = Hardware()
        self.freq = self.freq_levels[0]
        self.prev_freq = self.freq
        self.vdd = self.hardware.get_vdd(self.freq)

        self.backlog            = random_backlog(random.randrange(0, 2000), max_rate=3, max_step = MAX_STEP - 100)
        # self.backlog_size       = self.backlog.get_active_size(self.curr_step)
        self.backlog_not_active = self.backlog.get_status_size(self.curr_step, "not_active")
        self.backlog_ok         = self.backlog.get_status_size(self.curr_step, "ok")
        self.backlog_linear     = self.backlog.get_status_size(self.curr_step, "linear")
        self.backlog_crashed    = self.backlog.get_status_size(self.curr_step, "crash")

        self._update_derived()
        return self._get_obs()


    def step(self, action, inference=False):
        """
        action: integer in [0, space)
        mapped to (t_idx, f_idx)
        """
        assert self.action_space.contains(action), "Invalid action"

        # map action int -> (t_idx, f_idx)
        t_idx = action // self.Nf
        f_idx = action % self.Nf
        t_idx = int(np.clip(t_idx, 0, self.Nt - 1))
        f_idx = int(np.clip(f_idx, 0, self.Nf - 1))

        # apply action
        self.freq = float(self.freq_levels[f_idx])
        self.inference_flops = float(self.transformer.get_inference_flops(t_idx))
        self._update_derived()

        hardware_flops          = self.hardware.get_flops(self.freq)
        self.backlog.render(self.curr_step, hardware_flops, self.inference_flops)
        # self.backlog_size       = self.backlog.get_active_size(self.curr_step)
        self.backlog_not_active = self.backlog.get_status_size(self.curr_step, "not_active")
        self.backlog_ok         = self.backlog.get_status_size(self.curr_step, "ok")
        self.backlog_linear     = self.backlog.get_status_size(self.curr_step, "linear")
        self.backlog_crashed    = self.backlog.get_status_size(self.curr_step, "crash")


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
                f"[{self.curr_step}] ({reward:6.3f}), " +\
                f"[f:{self.freq:6}, f_max:{self.max_freq:.0f}], " +\
                f"[inference_flops: {self.inference_flops:.1e}] [{self.backlog_not_active}, {self.backlog_ok}, {self.backlog_linear}, {self.backlog_crashed}], "
            )

            
        # info handy for debugging/inference
        # info = {
        #     "processed": 0,
        #     "arrived": 0,
        #     "f_applied": float(self.freq),
        #     "power": float(self.power),
        #     # "backlog": int(self.backlog_running),
        #     "insert_rate": self.backlog_crashed,
        #     "max_freq": self.hardware.get_max_freq()
        # }
        info = {}
        
        self.curr_step += 1
        done = (self.curr_step >= self.max_step)

        obs = self._get_obs()
        return obs, reward, done, info


    def _update_derived(self):
        # update latency & power for normalization / observation
        self.vdd = self.hardware.get_vdd(self.freq)

        delay_power = self.hardware.get_delay_power(self.vdd)
        self.latency = delay_power[0]
        self.power = delay_power[1] * self.freq

        step_time_factor = 100
        t0 = self.curr_step * step_time_factor #seconds
        t1 = (self.curr_step + 1) * step_time_factor
        t0, t1 = 10*t0, 10*t1   #each steo as 10 seconds
        # self.hardware.apply_aging(self.vdd, t0, t1)

    def _get_obs(self):
        # normalized observation vector, clipped into [0,1]
        obs = np.array([
            float(self.freq)                / (self.max_freq + 1e-9),
            float(self.inference_flops)                / float(self.max_inference_flops),

            float(self.vdd)                 / (self.max_voltage + 1e-9),
            float(self.latency)             / (self.max_latency + 1e-9),
            float(self.power)               / (self.max_power + 1e-9),
            
            # float(self.backlog_size)        / (self.max_backlog + 1e-9),
            # float(self.backlog_not_active)  / (self.max_backlog + 1e-9),
            float(self.backlog_ok)          / (self.max_backlog + 1e-9),
            float(self.backlog_linear)      / (self.max_backlog + 1e-9),
            float(self.backlog_crashed)     / (self.max_backlog + 1e-9),
            
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)
