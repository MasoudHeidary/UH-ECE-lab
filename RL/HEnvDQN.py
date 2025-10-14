# HEnv_fixed.py
import gym
from gym import spaces
import numpy as np
import random
import math
from typing import List
from copy import deepcopy

from tool.log import Log
import cadence

MAX_STEP = 1000
MAX_BACKLOG_SIZE = 1000
log = Log(f"{__file__}.log", terminal=True)


class Instruction():
    def __init__(self, start_step, exe_step):
        self.start_step = start_step                # timestep that instruction enters backlog
        self.end_step = start_step + exe_step       # timestep that instruction execution should finish
        self.needed_FLOPS = 0                       # remaining FLOPs to finish instruction
        self.running = False                        # is instruction executing/waiting for execution
        
    def __repr__(self):
        return f"{(self.start_step, self.running, self.needed_FLOPS)}"
    
    def is_running(self):
        return self.running
    
    def is_crashed(self, current_step):
        """is current_step is more than end_step that means instruction has been waiting over-period"""
        return current_step > self.end_step
    
    def set(self, needed_FLOPS):
        """based on the model loaded, FLOPS will be different"""
        if self.running:
            raise RuntimeError("cannot re-set the task")
        if needed_FLOPS <= 0:
            raise ValueError("needed FLOPs should be positive")
        
        self.needed_FLOPS = needed_FLOPS
        self.running = True
    
    def render(self, FLOPS) -> bool:
        """return True when Instruction is done"""
        if not self.running:
            raise RuntimeError("can not render task which is not running")
        
        if self.needed_FLOPS < 0:
            raise RuntimeError(f"this task has already been finished")
        
        self.needed_FLOPS -= FLOPS
        # if self.needed_FLOPS <= 0:
        #     self.running = False
        return self.needed_FLOPS <= 0
    
    def get_propagation(self, current_step, FLOPs_rate):
        # return max(0, current_step - self.arrival_step)
        if not self.running:
            return 0
        return self.needed_FLOPS / FLOPs_rate #+ (current_step - self.start_step)
    
    
class Backlog():
    def __init__(self, instruction_lst: List[Instruction]):
        self.inst_lst = instruction_lst
        
    def __repr__(self):
        return f"{[self.inst_lst]}"
    
    def count_running(self):
        count = 0
        for inst in self.inst_lst:
            count += int(inst.is_running())
        return count
    
    def count_crashed(self, current_step: int) -> int:
        counter = 0
        for inst in self.inst_lst:
            counter += int(inst.is_crashed(current_step))
        return counter
            
    def get_propagation(self, current_step, FLOPs_rate):
        if FLOPs_rate == 0:
            if self.count_running() == 0:
                return 0
            FLOPs_rate = 1
        
        sum_needed_flops = 0    # needed FLOPS to empty the backlog
        sum_steps = 0
        for inst in self.inst_lst:
            sum_steps += inst.get_propagation(current_step, FLOPs_rate)
        return sum_steps
    
    def activate_tasks(self, current_step, task_FLOPS):
        """return number of activated tasks"""
        counter = 0
        for inst in self.inst_lst:
            if (not inst.is_running()) and (inst.start_step <= current_step):
                inst.set(task_FLOPS)
                counter += 1
        return counter
                
    
    def render(self, current_step, FLOPS, task_FLOPS):
        if len(self.inst_lst) == 0:
            return False
        
        self.activate_tasks(current_step, task_FLOPS)
        
        inst = self.inst_lst[0]
        if not inst.is_running():
            # inst.set(task_FLOPS)
            return False
        done = inst.render(FLOPS)
        if done:
            self.inst_lst.pop(0)
        return True


def random_backlog(inst_length):
    inst_lst = []
    for i in range(inst_length):
        start_step = random.randrange(0, MAX_STEP-100)
        inst = Instruction(start_step, 5)  
        inst_lst.append(inst)
    inst_lst.sort(key = lambda inst: inst.start_step)
    return Backlog(inst_lst)

backlog = random_backlog(10)
log.println(f"backlog: {backlog}")





class SystolicArrayEnv(gym.Env):
    """
    Discrete-action DVFS/backlog env compatible with DQN.
    Action: single Discrete(Nv * Nf) integer -> mapped to (voltage_index, freq_index)
    Observation: normalized finite 7-vector (safe for DQN).
    """

    def __init__(self):
        super().__init__()

        self.voltage_levels = [0.60, 0.70, 0.80, 0.90]
        self.frequency_levels = list(range(100, 1000+1, 100))

        self.Nv = len(self.voltage_levels)
        self.Nf = len(self.frequency_levels)
        self.action_space = spaces.Discrete(self.Nv * self.Nf)

        # observation: volt_norm, freq_norm, latency_norm, power_norm, backlog_norm, processed_norm, insert_rate_norm
        low = np.array([0.]*6, dtype=np.float32)
        high = np.array([1.]*6, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # parameters
        # self.max_backlog = MAX_BACKLOG_SIZE
        self.max_backlog = 5
        self.max_step = MAX_STEP
        self.max_processed = 25

        self.max_voltage = float(self.voltage_levels[-1])
        self.max_frequency = float(self.frequency_levels[-1])
        # for normalization stable nonzero denominators
        self.max_latency = self._latency_from_freq(max(1, int(self.max_frequency)))
        self.max_power = self._power_from_freq(self.max_voltage, self.max_frequency)

        # internal state
        self.current_voltage = 0.0
        self.current_frequency = 0.0

        self.latency = 0.0
        self.power = 0.0

        self.backlog = 0
        self.backlog_running = 0
        self.backlog_crashed = 0
        # self.processed = 0
        # self.insert_rate = 0

        self.prev_frequency = 0.0
        self.current_step = 0

        # random seed
        self.seed()

    # ----------------- gym API -----------------
    def seed(self, s=None):
        random.seed(s)
        np.random.seed(s)

    def reset(self, insert_rate=None):
        # choose start volt/freq (middle)
        self.current_voltage = self.voltage_levels[0]
        self.current_frequency = self.frequency_levels[0]
        self.prev_frequency = self.current_frequency

        # backlog random start small to encourage learning
        self.backlog = random_backlog(100)
        self.backlog_running = self.backlog.count_running() #random.randint(0, max(1, self.max_backlog // 10))
        self.backlog_crashed = self.backlog.count_crashed(0)
        # self.processed = 0

        # choose fixed insert_rate for the whole episode (so agent can learn to adapt)
        # if insert_rate is None:
        #     # choose between 0..max_insert
        #     max_insert = max(1, self.max_backlog // 40)
        #     self.insert_rate = random.randint(0, max_insert)
        # else:
        #     self.insert_rate = int(insert_rate)
        # self.insert_rate = 0

        self.current_step = 0
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
        v_idx = action // self.Nf
        f_idx = action % self.Nf
        # clamp indices
        v_idx = int(np.clip(v_idx, 0, self.Nv - 1))
        f_idx = int(np.clip(f_idx, 0, self.Nf - 1))

        # apply action
        self.current_voltage = float(self.voltage_levels[v_idx])
        self.current_frequency = float(self.frequency_levels[f_idx])
        self._update_derived()

        # 1) process before arrivals (causal)
        # prev_backlog = int(self.backlog_size)
        # self.processed = self._process_capacity(self.current_frequency)
        # self.processed = min(self.processed, self.backlog_size)
        # self.backlog_size = max(0, self.backlog_size - self.processed)

        self.backlog.render(self.current_step, self.current_frequency, 2000)
        # self.backlog_size = self.backlog.get_propagation(self.current_step, self.current_frequency)
        self.backlog_running = self.backlog.count_running()
        self.backlog_crashed = self.backlog.count_crashed(self.current_step)

        # self.insert_rate = arrived

        # 3) compute derived measures

        # 4) reward design
        # difference-based term normalizes by max_processed (clear credit)
        # diff_term = (self.processed - arrived) / max(1.0, self.max_processed)
        diff_term = 0
        # diff_term = -0.01 * (self.backlog_size / self.max_backlog)**2
        # if self.backlog_size > 5:
        #     diff_term -= 2 * self.backlog.count_running()
        diff_term += -7 * self.backlog_crashed
        
        # self.backlog_size = int(min(self.max_backlog, self.backlog_size + arrived))

        power_penalty = (self.power / (self.max_power + 1e-9))
        # switch_penalty = 0.2 * (abs(self.current_frequency - self.prev_frequency) / (self.max_frequency + 1e-9))
        switch_penalty = 0
        if self.current_frequency != self.prev_frequency:
            switch_penalty += 0.1
        self.prev_frequency = self.current_frequency
        
        reward = float(diff_term - power_penalty - switch_penalty)

        if self.backlog.count_running() == 0:
            if self.current_frequency == self.frequency_levels[0]:
                reward += 0.01

        # update prev frequency for next step
        if inference:
            # print(f"step [{self.current_step}]({reward:5.3f}): {diff_term:.3f}({self.backlog_crashed}) {power_penalty:.3f} {switch_penalty:.3f} [f={self.current_frequency}]")
            log.println(
                f"[{self.current_step}]({reward:6.3f}), " +\
                f"[f:{self.current_frequency:6}], [running:{self.backlog_running:3}, crashed:{self.backlog_crashed}]] " +\
                f""
            )

        # bookkeeping

            
        # info handy for debugging/inference
        info = {
            "processed": 0,
            "arrived": 0,
            "f_applied": float(self.current_frequency),
            "power": float(self.power),
            "backlog": int(self.backlog_running),
            "insert_rate": self.backlog_crashed,
        }
        
        self.current_step += 1
        done = (self.current_step >= self.max_step) #or (self.backlog_size >= self.max_backlog)

        obs = self._get_obs()
        return obs, reward, done, info

    # ----------------- helpers -----------------
    def _process_capacity(self, freq):
        # processing capacity linear with freq; tune scale so f_max can process some backlog
        # At max freq we can process around self.max_processed tasks per step
        cap = int((float(freq) / max(1.0, self.max_frequency)) * float(self.max_processed))
        return max(0, cap)

    def _latency_from_freq(self, f):
        # prevent divide by zero: use large latency when f==0
        if f <= 0:
            return 20.0
        return 1000.0 / float(f)

    def _power_from_freq(self, v, f):
        # simple approximate power model, scale to keep numbers reasonable
        frac = float(f) / max(1.0, self.max_frequency)
        # P ~ V^2 * f but we keep V independently from freq here
        return (v) * (f**1.5) * 1000

    def _update_derived(self):
        # update latency & power for normalization / observation
        self.latency = self._latency_from_freq(self.current_frequency)
        self.power = self._power_from_freq(self.current_voltage, self.current_frequency)

    def _get_obs(self):
        # normalized observation 7-vector, clipped into [0,1]
        obs = np.array([
            float(self.current_voltage) / (self.max_voltage + 1e-9),
            float(self.current_frequency) / (self.max_frequency + 1e-9),
            
            float(self.latency) / (self.max_latency + 1e-9),
            float(self.power) / (self.max_power + 1e-9),
            
            float(self.backlog_running) / (self.max_backlog + 1e-9),
            float(self.backlog_crashed) / (self.max_backlog + 1e-9),
            # float(self.processed) / (self.max_processed + 1e-9),
            # float(self.insert_rate) / (max(1, self.max_backlog // 40) + 1e-9),
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    # convenient render for debug
    def render(self, mode='human'):
        print(f"step {self.current_step:4d} freq {self.current_frequency:4.0f}MHz backlog {self.backlog_size:4d} insert {self.insert_rate} proc {self.processed} power {self.power:.2f}")
