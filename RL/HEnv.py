# HEnv - Hardware Environment 

import gym
from gym import spaces
import numpy as np
import random
import math

MAX_STEP = 1000
MAX_BACKLOG_SIZE = 1000


class Instruction():
    def __init__(self, current_step):
        self.start_step = current_step
        
    def delay(self, current_step):
        return current_step - self.start_step
    
    def done(self, current_step):
        return self.delay(current_step)
    
    def __repr__(self):
        return f"Inst({self.start_step})"
    
    
class Backlog():
    """simple FIFO"""
    def __init__(self):
        self.vec = []
        
    def __add__(self, obj: Instruction):
        self.vec += [obj]
    
    def size(self):
        return len(self.vec)
    
    def pop(self) -> Instruction:
        return self.vec.pop(0)
    
    def done(self, current_step):
        return self.pop().done(current_step)
    
    def __repr__(self):
        return f"FIFO{self.vec}"


class SystolicArrayEnv(gym.Env):
    def __init__(self):
        super(SystolicArrayEnv, self).__init__()

        self.voltage_levels = [0.60, 0.70, 0.80, 0.90]
        self.frequency_levels = list(range(0, 1000+1, 100))

        self.action_space = spaces.MultiDiscrete([len(self.voltage_levels), len(self.frequency_levels)])

        # voltage, frequency, latency, power, inference delay,
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)

        # internal state
        self.current_voltage = 0
        self.current_frequency = 0
        self.latency = 0
        self.power = 0
        
        self.backlog_size = 0
        self.processed = 0
        self.insert_rate = 0
        self.prev_freq = self.frequency_levels[0]

        self.current_step = 0

        # normalization parameter
        self.max_voltage = self.voltage_levels[-1]
        self.max_frequency = self.frequency_levels[-1]
        self.max_latency = self.get_latency(self.frequency_levels[-1])
        self.max_power = self.get_power(self.voltage_levels[-1], self.frequency_levels[-1])
        self.max_backlog_size = MAX_BACKLOG_SIZE
        self.max_processed = 25

    def reset(self):
        self.current_voltage = self.voltage_levels[0]
        self.current_frequency = self.frequency_levels[0]
        self.latency = self.get_latency()
        self.power = self.get_power()

        self.backlog_size = 0 #random.randrange(0, self.max_backlog_size+1)
        self.processed = 0
        self.insert_rate = 0

        self.current_step = 0
        
        return np.array([
            self.current_voltage / self.max_voltage,
            self.current_frequency / self.max_frequency,
            
            self.latency / self.max_latency,
            self.power / self.max_power,
            
            self.backlog_size / self.max_backlog_size,
            self.processed / self.max_processed,
            self.insert_rate / self.max_processed,
        ], dtype=np.float32)
    

    def step(self, action, insert_backlog=False, inference=False):
        v_idx, f_idx = action
        self.current_voltage = self.voltage_levels[v_idx]
        self.current_frequency = self.frequency_levels[f_idx]

        self.latency = self.get_latency()
        self.power = self.get_power()

        reward = 0
        if self.prev_freq != self.current_frequency:
            reward -= 0.5
        self.prev_freq = self.current_frequency
        
        backlog_before_process = self.backlog_size
        self.processed = self.update_process()
        self.random_backlog(inference)
        
        power_penalty = -1 * self.power / self.max_power
        # backlog_penalty = -0.5 * ((self.inference_delay / self.max_inference_delay)**0.5)
        # backlog_penalty = 0.35 * (self.processed / (backlog_before_process or 1))
        # backlog_penalty = 1 * (self.processed / (self.insert_rate or 1))
        backlog_penalty = 1 * (self.processed / self.max_processed)
        
        if self.backlog_size > self.max_backlog_size*0.75:
            backlog_penalty -= self.backlog_size * 0.1

        # reward = power_penalty + backlog_reward
        reward = power_penalty + backlog_penalty
        
        ret = {}
        if inference:
            print(f"[{self.current_step:4}] reward [{reward:.3f}] ({power_penalty:6.3f}, {backlog_penalty:6.3f}) f[{self.current_frequency}]")
            ret = {
                'pow': power_penalty,
                'bkl': backlog_penalty,
            }

        obs = np.array([
            self.current_voltage / self.max_voltage,
            self.current_frequency / self.max_frequency,
            self.latency / self.max_latency,
            self.power / self.max_power,
            
            self.backlog_size / self.max_backlog_size,
            self.processed / self.max_processed,
            self.insert_rate / self.max_processed
        ], dtype=np.float32)


        self.current_step += 1
        done = self.current_step >= MAX_STEP or self.backlog_size > self.max_backlog_size
        return obs, reward, done, ret
    
    def random_backlog(self, inference):
        # if self.current_step % 100 == 0:
        #     self.backlog_size += int(self.max_backlog_size * (self.current_step / 900))
        
        if not inference:
            # add = random.randrange(0, 25+1)
            add = random.choice([0]*25 + list(range(0, 26)))
            # add = abs(int(25 * (1 - self.current_step / 500)))
        else:
            # add = random.randrange(0, 25+1)
            add = random.choice([0]*25 + list(range(0, 26)))
        self.insert_rate = add
        self.backlog_size += add
        return self.backlog_size
    
    # def potential_processed = 

    def update_process(self):
        processed = self.current_frequency // 40
        processed = min(self.backlog_size, processed)
        self.backlog_size -= processed
        return processed

    # ----------------- COMPUTE
    def get_latency(self, f=False):
        if not f:
            f = self.current_frequency
        if f == 0:
            return 1000
        return 1000 / f
    
    def get_power(self, v=False, f=False):
        if not v:
            v = self.current_voltage
        if not f:
            f = self.current_frequency
        return 100 * (f+1) * (v**2)
    
    # def get_inference_delay(self, latency=False, queue=False):
    #     d = self.latency if not latency else latency
    #     backlog = self.backlog_size if not queue else queue
    #     return backlog * d
    
        
    




