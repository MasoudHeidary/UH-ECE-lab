from .sim import *

class Not:

    def __init__(self) -> None:
        self._input = N
        self._output = N

        self.win = Wire()
        self.p = PMOS()
        self.n = NMOS()
        self.wout = Wire(2)
        self.elements = [self.win, self.p, self.n, self.wout]

    def netlist(self):
        self.win[0] = self.input
        self.p.input = H
        self.p.gate = self.win.output
        self.n.input = L
        self.n.gate = self.win.output
        self.wout[0] = self.p.output
        self.wout[1] = self.n.output
        self.output = self.wout.output

    @property
    def change_flag(self):
        return any([i.change_flag for i in self.elements])

    # input pins
    @property
    def input(self):
        return self._input
    
    @input.setter
    def input(self, value):
        self._input = value

    @property
    def output(self):
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self._output
    
    @output.setter
    def output(self, value):
        self._output = value

    def __repr__(self) -> str:
        return f"{self.input} -> {self.output}"


class eNot:
    def __init__(self):
        self.__input = N
        self.__output = N
    
        self.__change_flag = True

    @property
    def change_flag(self):
        return self.__change_flag
    
    @property
    def input(self):
        return self.__input
    
    @input.setter
    def input(self, value):
        if(self.__input != value):
            self.__change_flag = True
            self.__input = value
            
    def __netlist(self):
        self.__output = H if self.__input==L else L
    
    @property
    def output(self):
        self.__change_flag = False
        self.__netlist()
        return self.__output
    
class And:

    def __init__(self) -> None:
        self.A = N
        self.B = N
        self.__output = N

        self.p = [PMOS() for _ in range(2)]
        self.n = [NMOS() for _ in range(2)]
        self.w = Wire(3)
        self.ngate = Not()

        self.elements = self.p + self.n + [self.w, self.ngate]

    def netlist(self):

        self.p[0].input = H
        self.p[0].gate = self.A
        self.w[0] = self.p[0].output

        self.p[1].input = H
        self.p[1].gate = self.B
        self.w[1] = self.p[1].output

        self.w[2] = self.n[0].output
        self.n[0].gate = self.A
        self.n[0].input = self.n[1].output

        self.n[1].gate = self.B
        self.n[1].input = L

        self.ngate.input = self.w.output
        self.__output = self.ngate.output
        
    @property
    def change_flag(self):
        return any([i.change_flag for i in self.elements])

    @property
    def output(self):
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self.__output

class eAnd:
    def __init__(self):
        self.__A = N
        self.__B = N
        
        self.__change_flag = True
        self.__output = N
    
    @property
    def change_flag(self):
        return self.__change_flag
    
    @property
    def A(self):
        return self.__A
    
    @A.setter
    def A(self, value):
        if(self.__A != value):
            self.__change_flag = True
            self.__A = value
    
    @property
    def B(self):
        return self.__B
    
    @B.setter
    def B(self, value):
        if(self.__B != value):
            self.__change_flag = True
            self.__B = value
    
    def __netlist(self):
        self.__output = self.A and self.B
    
    @property
    def output(self):
        self.__change_flag = False
        self.__netlist()
        return self.__output
    
    

def __test__and():
    g_and = And()

    g_and.A = L
    g_and.B = L
    if (g_and.output != L):
        raise RuntimeError()
    print(g_and.output)

    g_and.A = H
    if (g_and.output != L):
        raise RuntimeError()
    print(g_and.output)

    g_and.B = H
    if(g_and.output != H):
        raise RuntimeError()
    print(g_and.output)

    print(g_and.n[0].gate)


if __name__ == "__main__":
    __test__and()



