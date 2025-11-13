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
from cadence.hardware import Hardware

MAX_STEP = 1000
MAX_BACKLOG_SIZE = 50
log = Log(f"{__file__}.log", terminal=True)


# backlog = random_backlog(20)
# log.println(f"backlog: {backlog}")


class TransformerModel:
    def __init__(self):
        self.valid_d_model = [128, 256, 512, 1024]
        self.valid_num_layer = [1, 2, 4, 6, 8]

        self.model = [
            # ftp32
            {'d_model': 128,    'num_layer': 1, 'precision': 'ftp32',   'flops': 79712000   * 1e4,    'loss': 4.13485},
            {'d_model': 128,    'num_layer': 2, 'precision': 'ftp32',   'flops': 143891200  * 1e4,    'loss': 4.10043},
            {'d_model': 128,    'num_layer': 4, 'precision': 'ftp32',   'flops': 272249600  * 1e4,    'loss': 4.05687},
            {'d_model': 128,    'num_layer': 6, 'precision': 'ftp32',   'flops': 400608000  * 1e4,    'loss': 3.99571},
            {'d_model': 128,    'num_layer': 8, 'precision': 'ftp32',   'flops': 528966400  * 1e4,    'loss': 3.96060},
            
            {'d_model': 256,    'num_layer': 1, 'precision': 'ftp32',   'flops': 179084800  * 1e4,    'loss': 3.84123},
            {'d_model': 256,    'num_layer': 2, 'precision': 'ftp32',   'flops': 327104000  * 1e4,    'loss': 3.78952},
            {'d_model': 256,    'num_layer': 4, 'precision': 'ftp32',   'flops': 623142400  * 1e4,    'loss': 3.76332},
            {'d_model': 256,    'num_layer': 6, 'precision': 'ftp32',   'flops': 919180800  * 1e4,    'loss': 3.74532},
            {'d_model': 256,    'num_layer': 8, 'precision': 'ftp32',   'flops': 1215219200 * 1e4,    'loss': 3.71966},

            {'d_model': 512,    'num_layer': 1, 'precision': 'ftp32',   'flops': 436812800  * 1e4,    'loss': 3.41552},
            {'d_model': 512,    'num_layer': 2, 'precision': 'ftp32',   'flops': 811494400  * 1e4,    'loss': 3.37815},
            {'d_model': 512,    'num_layer': 4, 'precision': 'ftp32',   'flops': 1560857600 * 1e4,    'loss': 3.35598},
            {'d_model': 512,    'num_layer': 6, 'precision': 'ftp32',   'flops': 2310220800 * 1e4,    'loss': 3.34076},
            {'d_model': 512,    'num_layer': 8, 'precision': 'ftp32',   'flops': 3059584000 * 1e4,    'loss': 3.33272},

            {'d_model': 1024,   'num_layer': 1, 'precision': 'ftp32',   'flops': 1188198400 * 1e4,    'loss': 3.40941},
            {'d_model': 1024,   'num_layer': 2, 'precision': 'ftp32',   'flops': 2252134400 * 1e4,    'loss': 3.36589},
            {'d_model': 1024,   'num_layer': 4, 'precision': 'ftp32',   'flops': 4380006400 * 1e4,    'loss': 3.34026},
            {'d_model': 1024,   'num_layer': 6, 'precision': 'ftp32',   'flops': 6507878400 * 1e4,    'loss': 3.32685},
            {'d_model': 1024,   'num_layer': 8, 'precision': 'ftp32',   'flops': 8635750400 * 1e4,    'loss': 3.29182},

            # ftp16
            {'d_model': 128,    'num_layer': 1, 'precision': 'ftp16',   'flops': 79712000   * 1e4,    'loss': 4.17394},
            {'d_model': 128,    'num_layer': 2, 'precision': 'ftp16',   'flops': 143891200  * 1e4,    'loss': 4.13692},
            {'d_model': 128,    'num_layer': 4, 'precision': 'ftp16',   'flops': 272249600  * 1e4,    'loss': 4.08615},
            {'d_model': 128,    'num_layer': 6, 'precision': 'ftp16',   'flops': 400608000  * 1e4,    'loss': 4.02786},
            {'d_model': 128,    'num_layer': 8, 'precision': 'ftp16',   'flops': 528966400  * 1e4,    'loss': 3.99056},
            
            {'d_model': 256,    'num_layer': 1, 'precision': 'ftp16',   'flops': 179084800  * 1e4,    'loss': 3.89982},
            {'d_model': 256,    'num_layer': 2, 'precision': 'ftp16',   'flops': 327104000  * 1e4,    'loss': 3.83931},
            {'d_model': 256,    'num_layer': 4, 'precision': 'ftp16',   'flops': 623142400  * 1e4,    'loss': 3.80797},
            {'d_model': 256,    'num_layer': 6, 'precision': 'ftp16',   'flops': 919180800  * 1e4,    'loss': 3.79364},
            {'d_model': 256,    'num_layer': 8, 'precision': 'ftp16',   'flops': 1215219200 * 1e4,    'loss': 3.75623},

            {'d_model': 512,    'num_layer': 1, 'precision': 'ftp16',   'flops': 436812800  * 1e4,    'loss': 3.45409},
            {'d_model': 512,    'num_layer': 2, 'precision': 'ftp16',   'flops': 811494400  * 1e4,    'loss': 3.41083},
            {'d_model': 512,    'num_layer': 4, 'precision': 'ftp16',   'flops': 1560857600 * 1e4,    'loss': 3.38601},
            {'d_model': 512,    'num_layer': 6, 'precision': 'ftp16',   'flops': 2310220800 * 1e4,    'loss': 3.36381},
            {'d_model': 512,    'num_layer': 8, 'precision': 'ftp16',   'flops': 3059584000 * 1e4,    'loss': 3.35524},

            {'d_model': 1024,   'num_layer': 1, 'precision': 'ftp16',   'flops': 1188198400 * 1e4,    'loss': 3.45220},
            {'d_model': 1024,   'num_layer': 2, 'precision': 'ftp16',   'flops': 2252134400 * 1e4,    'loss': 3.40569},
            {'d_model': 1024,   'num_layer': 4, 'precision': 'ftp16',   'flops': 4380006400 * 1e4,    'loss': 3.36988},
            {'d_model': 1024,   'num_layer': 6, 'precision': 'ftp16',   'flops': 6507878400 * 1e4,    'loss': 3.35982},
            {'d_model': 1024,   'num_layer': 8, 'precision': 'ftp16',   'flops': 8635750400 * 1e4,    'loss': 3.32522},

            # ftp8
            {'d_model': 128,    'num_layer': 1, 'precision':  'ftp8',   'flops': 79712000   * 1e4,    'loss': 4.23952},
            {'d_model': 128,    'num_layer': 2, 'precision':  'ftp8',   'flops': 143891200  * 1e4,    'loss': 4.18807},
            {'d_model': 128,    'num_layer': 4, 'precision':  'ftp8',   'flops': 272249600  * 1e4,    'loss': 4.13429},
            {'d_model': 128,    'num_layer': 6, 'precision':  'ftp8',   'flops': 400608000  * 1e4,    'loss': 4.07153},
            {'d_model': 128,    'num_layer': 8, 'precision':  'ftp8',   'flops': 528966400  * 1e4,    'loss': 4.03815},
            
            {'d_model': 256,    'num_layer': 1, 'precision':  'ftp8',   'flops': 179084800  * 1e4,    'loss': 3.98115},
            {'d_model': 256,    'num_layer': 2, 'precision':  'ftp8',   'flops': 327104000  * 1e4,    'loss': 3.90957},
            {'d_model': 256,    'num_layer': 4, 'precision':  'ftp8',   'flops': 623142400  * 1e4,    'loss': 3.86845},
            {'d_model': 256,    'num_layer': 6, 'precision':  'ftp8',   'flops': 919180800  * 1e4,    'loss': 3.85537},
            {'d_model': 256,    'num_layer': 8, 'precision':  'ftp8',   'flops': 1215219200 * 1e4,    'loss': 3.81516},

            {'d_model': 512,    'num_layer': 1, 'precision':  'ftp8',   'flops': 436812800  * 1e4,    'loss': 3.51250},
            {'d_model': 512,    'num_layer': 2, 'precision':  'ftp8',   'flops': 811494400  * 1e4,    'loss': 3.45935},
            {'d_model': 512,    'num_layer': 4, 'precision':  'ftp8',   'flops': 1560857600 * 1e4,    'loss': 3.41406},
            {'d_model': 512,    'num_layer': 6, 'precision':  'ftp8',   'flops': 2310220800 * 1e4,    'loss': 3.41089},
            {'d_model': 512,    'num_layer': 8, 'precision':  'ftp8',   'flops': 3059584000 * 1e4,    'loss': 3.41374},

            {'d_model': 1024,   'num_layer': 1, 'precision':  'ftp8',   'flops': 1188198400 * 1e4,    'loss': 3.45990},
            {'d_model': 1024,   'num_layer': 2, 'precision':  'ftp8',   'flops': 2252134400 * 1e4,    'loss': 3.38753},
            {'d_model': 1024,   'num_layer': 4, 'precision':  'ftp8',   'flops': 4380006400 * 1e4,    'loss': 3.37995},
            {'d_model': 1024,   'num_layer': 6, 'precision':  'ftp8',   'flops': 6507878400 * 1e4,    'loss': 3.37062},
            {'d_model': 1024,   'num_layer': 8, 'precision':  'ftp8',   'flops': 8635750400 * 1e4,    'loss': 3.33013},
        ]

    def get_model(self, d_model, num_layer):
        for model in self.model:
            if (model['d_model'] == d_model) and (model['num_layer'] == num_layer):
                return model
        raise ValueError("model not found!")
        
    def get_inference_flops(self, d_model, num_layer):
        model = self.get_model(d_model, num_layer)
        return model['flops']
    
    def get_max_flops(self):
        d_model = self.valid_d_model[-1]
        num_layer = self.valid_num_layer[-1]
        max_model = self.get_model(d_model, num_layer)
        return max_model['flops']
    

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

        # action levels
        self.freq_levels    = list(range(0, 1000+1, 100))
        self.dmodel_levels  = self.transformer.valid_d_model.copy()
        self.numlay_levels  = self.transformer.valid_num_layer.copy()

        # action space
        self.Nf             = len(self.freq_levels)
        self.Nd             = len(self.dmodel_levels)
        self.Nl             = len(self.numlay_levels)
        self.action_space   = spaces.Discrete(self.Nf * self.Nd * self.Nl)

        # action states
        self.freq: int
        self.dmodel: int
        self.numlay: int

        # observation space
        low                     = np.array([0.]*9, dtype=np.float32)
        high                    = np.array([1.]*9, dtype=np.float32)
        self.observation_space  = spaces.Box(low=low, high=high, dtype=np.float32)

        # maximum parameters
        self.max_freq       = float(self.freq_levels[-1])
        self.max_dmodel     = int(self.dmodel_levels[-1])
        self.max_numlay     = int(self.numlay_levels[-1])

        self.max_step       = MAX_STEP
        self.max_backlog    = MAX_BACKLOG_SIZE

        self.max_voltage    = self.hardware.get_max_mapping_voltage()
        self.max_inference_flops      = self.transformer.get_max_flops()
        self.max_latency    = self.hardware.get_max_mapping_delay()
        self.max_energy      = self.hardware.get_max_mapping_power() * self.max_freq

        # derived variables
        self.vdd        :float
        self.latency    :float
        self.energy      :float

        self.backlog            :Backlog
        self.backlog_not_active :int
        self.backlog_ok         :int
        self.backlog_linear     :int
        self.backlog_crashed    :int 

        self.prev_freq :int
        self.curr_step :int
        self.seed()

    def seed(self, s=None):
        random.seed(s)
        np.random.seed(s)

    def reset(self):
        self.curr_step = 0
        self.inference_flops = 0
        self.hardware_flops = 0
        
        self.hardware = Hardware()
        self.transformer = TransformerModel()

        # default action values
        self.freq       = self.freq_levels[0]
        self.dmodel     = self.dmodel_levels[0]
        self.numlay     = self.numlay_levels[0]

        self.prev_freq = self.freq
        self.prev_dmodel = self.dmodel
        self.prev_numlay = self.numlay

        # self.prev_freq = self.freq
        # derived parameters
        self.vdd = self.hardware.get_vdd(self.freq)
        delay_power = self.hardware.get_delay_power(self.vdd)
        self.latency, self.energy = delay_power[0], delay_power[1] * self.freq

        self.backlog            = random_backlog(random.randrange(0, 1500), max_rate=4, max_step = MAX_STEP - 100)
        # self.backlog_size       = self.backlog.get_active_size(self.curr_step)
        self.backlog_not_active = self.backlog.get_status_size(self.curr_step, "not_active")
        self.backlog_ok         = self.backlog.get_status_size(self.curr_step, "ok")
        self.backlog_linear     = self.backlog.get_status_size(self.curr_step, "linear")
        self.backlog_crashed    = self.backlog.get_status_size(self.curr_step, "crash")

        
        return self._get_obs()


    def step(self, action, inference=False):
        assert self.action_space.contains(action), "Invalid action"
        
        reward = 0
        
        # decode action
        f_idx = action % self.Nf
        d_idx = (action // self.Nf) % self.Nd
        l_idx = (action // (self.Nf * self.Nd)) % self.Nl
        # f_idx = int(np.clip(f_idx, 0, self.Nf - 1))

        # apply action
        self.freq = float(self.freq_levels[f_idx])
        self.dmodel = int(self.dmodel_levels[d_idx])
        self.numlay = int(self.numlay_levels[l_idx])

        if (self.prev_dmodel != self.dmodel) and (self.prev_numlay != self.numlay):
            reward -= 0.1
        self.prev_dmodel = self.dmodel
        self.prev_numlay = self.numlay


        # derived parameters
        self.vdd = self.hardware.get_vdd(self.freq)
        delay_power = self.hardware.get_delay_power(self.vdd)
        self.latency, self.energy = delay_power[0], delay_power[1] * self.freq

        self.inference_flops    = float(self.transformer.get_inference_flops(self.dmodel, self.numlay))
        self.hardware_flops     = self.hardware.get_flops(self.freq, "ftp32")

        # update backlog observation values
        self.backlog.render(self.curr_step, self.hardware_flops, self.inference_flops)
        self.backlog_not_active = self.backlog.get_status_size(self.curr_step, "not_active")
        self.backlog_ok         = self.backlog.get_status_size(self.curr_step, "ok")
        self.backlog_linear     = self.backlog.get_status_size(self.curr_step, "linear")
        self.backlog_crashed    = self.backlog.get_status_size(self.curr_step, "crash")


        # reward -= self.backlog_linear * 0.02
        if self.backlog_crashed > 0:
            reward -= 5
        
        reward -= 1 * (self.energy / (self.max_energy + 1e-9))

        if inference:
            log.println(
                f"[{self.curr_step}] ({reward:6.3f}), " +\
                f"[f:{self.freq:6}, f_max:{self.max_freq:.0f}], " +\
                f"flops[infe: {self.inference_flops:.1e}, comp: {self.hardware_flops:.1e} {(self.dmodel, self.numlay)}], " +\
                f"backlog[{self.backlog_not_active}, {self.backlog_ok}, {self.backlog_linear}, {self.backlog_crashed}], "
            )

            
        info = {}
        self.curr_step += 1
        done = (self.curr_step >= self.max_step)

        obs = self._get_obs()
        return obs, reward, done, info


    # def _update_derived(self):
    #     # update latency & power for normalization / observation
    #     step_time_factor = 100
    #     t0 = self.curr_step * step_time_factor #seconds
    #     t1 = (self.curr_step + 1) * step_time_factor
    #     t0, t1 = 10*t0, 10*t1   #each steo as 10 seconds
    #     # self.hardware.apply_aging(self.vdd, t0, t1)

    def _get_obs(self):
        # normalized observation vector, clipped into [0,1]
        obs = np.array([
            float(self.freq)                / (self.max_freq + 1e-9),
            float(self.dmodel)              / (self.max_dmodel + 1e-9),
            float(self.numlay)              / (self.max_numlay + 1e-9),
            # float(self.inference_flops)                / float(self.max_inference_flops),

            float(self.vdd)                 / (self.max_voltage + 1e-9),
            float(self.latency)             / (self.max_latency + 1e-9),
            float(self.energy)               / (self.max_energy + 1e-9),
            
            float(self.backlog_ok)          / (self.max_backlog + 1e-9),
            float(self.backlog_linear)      / (self.max_backlog + 1e-9),
            float(self.backlog_crashed)     / (self.max_backlog + 1e-9),
            
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)
