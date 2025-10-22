
from typing import List


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
    
    def is_active(self, t):
        """current time is pass start point, but not done yet"""
        return (self.t0 <= t) and (not self.done)
    
    def get_status(self, t):
        """what stage of penalty we are"""
        if not self.is_active(t):
            return (0, 0)
        elif (self.t0 <= t < self.t1):
            return (0, t - self.t0 + 1)
        elif self.t1 <= t < self.t2:
            return (1, t - self.t1 + 1) #linear penalty
        elif self.t2 <= t:
            return (2, t - self.t2 + 1) #crashed
        raise RuntimeError("unexpected status!")
            
    def set_flops(self, flops, re_set_allowed=True):
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
        self.inst_lst = inst_lst.copy().sort(key = lambda inst: inst.t0)
        self.inst_lst: List[Instruction]
        
    def __repr__(self):
        return f"{[self.inst_lst]}"
    
    def get_active_size(self, t):
        count = 0
        for inst in self.inst_lst:
            count += int(inst.is_active(t))
        return count
    
    def get_linear_size(self, t):
        count = 0
        for inst in self.inst_lst:
            status = inst.get_status(t)
            if status[0] == 1:
                count += status[1]
        return count
            
    # def current_status(self, t):
    
    
    
if __name__ == "__main__":
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