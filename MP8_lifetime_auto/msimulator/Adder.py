

from .sim import *
from .logic import *
from .bin_func import *
import time

from .Multiplier import eFA


class RippleAdder:

    def __init__(self, A:list[int], B:list[int], bit_len=8) -> None:
        self.bit_len = bit_len
        
        # input output register states
        self.A = A.copy()
        self.B = B.copy()
        self.__sum = [N for _ in range(bit_len)]
        self.__overflow = N

        # adding gates in block
        self.gfa = [eFA() for _ in range(bit_len)]

        # list of all elements to watch for
        self.elements = self.gfa

    @property
    def change_flag(self) -> bool:
        """if block is steady state"""
        return any([i.change_flag for i in self.elements])
    
    @property
    def sum(self):
        """run block till steady state position"""
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self.__sum

    @property
    def overflow(self):
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self.__overflow
    
    def netlist(self) -> None:
        """circuit wiring and connection between gates"""
        for i in range(self.bit_len):
            self.gfa[i].A = self.A[i]
            self.gfa[i].B = self.B[i]

            if i == 0:
                self.gfa[i].C = L
            else:
                self.gfa[i].C = self.gfa[i-1].carry
            
        
            # output mapping
            self.__sum[i] = self.gfa[i].sum
        self.__overflow = self.gfa[-1].carry



class CarrySkipAdder:

    def __init__(self, A:list[int], B:list[int], bit_len=8, block_size=4) -> None:
        self.bit_len = bit_len
        self.block_size = block_size
        if (self.bit_len % self.block_size) != 0:
            raise ValueError(f"bit_len does not fit in block_size")

        # input and output register states
        self.A = A.copy()
        self.B = B.copy()
        self.__sum = [N for _ in range(self.bit_len)]
        self.__overflow = N

        # adding gates to the circuit
        self.gfa = [eFA() for _ in range(self.bit_len)]

        # adding middle values
        self.P = [0 for _ in range(self.bit_len)]

        # list of all elements to watch for
        self.elements = self.gfa

    @property
    def change_flag(self) -> bool:
        """if block is steady state"""
        return any([i.change_flag for i in self.elements])
    
    @property
    def sum(self):
        """run block till steady state position"""
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self.__sum
    
    @property
    def overflow(self):
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self.__overflow
    
    def netlist(self) -> None:
        """circuit wiring and connection between gates"""

        # A, B input connections
        for i in range(self.bit_len):
            self.gfa[i].A = self.A[i]
            self.gfa[i].B = self.B[i]

            self.__sum[i] = self.gfa[i].sum

        # carry wiring
        for i in range(self.bit_len):
            # making P's
            self.P[i] = self.A[i] ^ self.B[i]

            if i == 0:
                self.gfa[i].C = L
            
            elif i % self.block_size != 0:
                self.gfa[i].C = self.gfa[i-1].carry
            
            elif i % self.block_size == 0:
                # skip carry logic
                # p_and = all([self.gand[j].output for j in range(i-self.block_size, i)])
                p_and = all([self.P[j] for j in range(i-self.block_size, i)])

                if p_and == 0:
                    # carry of previous gate
                    self.gfa[i].C = self.gfa[i-1].carry
                else:
                    # carry in of first FA of previous block
                    self.gfa[i].C = self.gfa[i-self.block_size].C

            else:
                raise SystemError("float carry-pin!")


        # output mapping
        self.__overflow = self.gfa[-1].carry


    
class CarrySaveAdder:
    """this adder, designed to add three numbers, so output register is one-bit bigger"""

    def __init__(self, A:list[int], B:list[int], C:list[int]=False, bit_len=8) -> None:
        self.bit_len = bit_len

        # input and output register states
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy() if C else [0 for _ in range(self.bit_len)]

        self.__sum = [N for _ in range(self.bit_len + 1)]
        self.__overflow = N

        # adding gates to the circuit
        self.gfa = [[eFA() for _ in range(self.bit_len)] for _ in range(2)]

        # list of all elements to watch for
        self.elements = self.gfa[0] + self.gfa[1]

    @property
    def change_flag(self) -> bool:
        """if block is steady state"""
        return any([i.change_flag for i in self.elements])
    
    @property
    def sum(self):
        """run block till steady state position"""
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self.__sum
    
    @property
    def overflow(self):
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self.__overflow
    

    def netlist(self) -> None:
        """circuit wiring and connection between gates"""

        # A, B input connections, first layer
        for i in range(self.bit_len):
            self.gfa[0][i].A = self.A[i]
            self.gfa[0][i].B = self.B[i]
            self.gfa[0][i].C = self.C[i]

            if i == 0:
                self.__sum[i] = self.gfa[0][i].sum

        # second layer input wiring
        for i in range(self.bit_len):
            self.gfa[1][i].A = self.gfa[0][i].carry
            self.gfa[1][i].B = self.gfa[0][i+1].sum     if i != (self.bit_len-1)    else    L
            self.gfa[1][i].C = self.gfa[1][i-1].carry   if i != 0                   else    L

            self.__sum[i+1] = self.gfa[1][i].sum
        self.__overflow = self.gfa[1][-1].carry