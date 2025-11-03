
from typing import List, Literal
import random
from copy import deepcopy


# Instruction
# -------------------------------------------------------------------------------
# |    no penalty    |    linear penalty    |    crashed - high flat penalty    |
# -------------------------------------------------------------------------------
# t0                 t1                     t2

class Instruction():
    def __init__(self, t0, t1, t2):
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        
        self.done = False
        self.sett = False
        self.FLOPs = 0
        
    def __repr__(self):
        return f"Inst{(self.t0, self.t1, self.t2, self.sett, self.done, self.FLOPs)}"
    
    def is_done(self):
        return self.done
    def is_set(self):
        return self.sett
    
    def is_active(self, t):
        """current time is pass start point, but not done yet"""
        return (self.t0 <= t) and (not self.done)
    
    def get_status(self, t):
        """what stage of penalty we are"""
        if self.is_done():
            return (3, 1)
        elif not self.is_active(t):
            return (-1, 1)
        elif (self.t0 <= t < self.t1):
            return (0, t - self.t0 + 1)
        elif self.t1 <= t < self.t2:
            return (1, t - self.t1 + 1) #linear penalty
        elif self.t2 <= t:
            return (2, t - self.t2 + 1) #crashed
        raise RuntimeError("unexpected status!")
            
    def set_flops(self, flops, re_set_allowed=False):
        if self.done:
            raise RuntimeError("can not set FLOPs in a finished instruction")
        if (self.sett) and (not re_set_allowed):
            raise RuntimeError("re setting FLOPs is not allowed")
        self.FLOPs = flops
        self.sett = True
    
    def render(self, t, flops):
        """return number of flops that will be used"""
        if not self.is_active(t):
            raise RuntimeError("can not render inactive (or done) instruction")
        if not self.sett:
            raise RuntimeError("instruction FLOPs was never set to be rendered")
        use_flops = min(self.FLOPs, flops)
        self.FLOPs -= use_flops
        if self.FLOPs <= 0:
            self.done = True
        return use_flops
    
    
class Backlog():
    def __init__(self, inst_lst: List[Instruction]):
        self.inst_lst = inst_lst.copy()
        self.inst_lst.sort(key = lambda inst: inst.t0)
        self.inst_lst: List[Instruction]
        
    def __repr__(self):
        return f"{self.inst_lst}"
    
    def get_active_size(self, t):
        count = 0
        for inst in self.inst_lst:
            count += int(inst.is_active(t))
        return count
    
    def get_status_size(self, t, status: Literal["not_active", "ok", "linear", "crash", "done"]):
        _ = {"not_active": -1, "ok": 0, "linear": 1, "crash": 2, "done": 3}
        inst_status_code = _[status]
        
        count = 0
        for inst in self.inst_lst:
            status = inst.get_status(t)
            if status[0] == inst_status_code:
                # count += status[1]
                count += 1
        return count
            
    def render(self, t, flops, set_flops):
        for inst in self.inst_lst:
            if inst.is_active(t) and not inst.is_set():
                inst.set_flops(set_flops)
        
        remain_flops = flops
        for inst in self.inst_lst:
            if inst.is_active(t) and not inst.is_done():
                used_flops = inst.render(t, remain_flops)
                remain_flops -= used_flops
                if remain_flops <= 0:
                    return
    

def random_backlog(inst_length, t1=5, t2=10, distance=0, max_rate=10, force_length=False, max_step=1000):
    """in distance, only max_rate instruction exist concurently"""
    inst_lst = []
    step_set = dict()
    for i in range(inst_length):
        t0 = random.randrange(0, max_step)
        inst = Instruction(t0, t0 + t1, t0 + t2)
        
        # exist = False
        # counter = 0
        # for dis in range(-distance, distance+1):
        #     if step_set.get(inst.start_step + dis, -1) == -1:
        #         step_set[inst.start_step + dis] = 0
        #     counter += step_set[inst.start_step + dis]
        #     if counter == max_rate:
        #         exist = True
        #         break
        
        # if not exist:
        #     inst_lst.append(inst)
        #     step_set[inst.start_step] += 1
        # elif force_length:
        #     i -= 1
        inst_lst.append(inst)
    return Backlog(inst_lst)
 
    
if __name__ == "__main__":
    
    if False:
        inst = Instruction(10, 15, 20)
        for step in range(40):
            if step == 5:
                inst.set_flops(1000)
            
            if inst.is_active(step) and not inst.is_done():
                inst.render(step, 50)
            
            print(f"[{step:2}] ### {inst}")
            print(f"active: {inst.is_active(step)}, done: {inst.is_done()}")
            print(f"status: {inst.get_status(step)}")
            print()
            
    if True:
        # backlog = Backlog([Instruction(10, 15, 20)])
        backlog = random_backlog(2, MAX_STEP=10)
        print(backlog)
        
        for step in range(40):
            print(f"[{step:2}] ### {backlog}")
            backlog.render(step, 100, 1000)
            
            not_active = backlog.get_status_size(step, "not_active")
            ok         = backlog.get_status_size(step, "ok")
            linear     = backlog.get_status_size(step, "linear")
            crashed    = backlog.get_status_size(step, "crash")
            print(f"not_active [{not_active}], ok [{ok}], linear [{linear}], crashed [{crashed}]")
            print()