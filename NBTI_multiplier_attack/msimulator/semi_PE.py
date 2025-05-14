# from msimulator.Multiplier import *
# from msimulator.bin_func import *
from .Multiplier import *
from .bin_func import *

class SemiPEArray:
    def __init__(self, A_x=[], A_y=[], bit_len=8, x_len=1, y_len=1):
        self.A_x = A_x.copy()
        self.A_y = A_y.copy()
        self.bit_len = bit_len
        self.x_len = x_len
        self.y_len = y_len
        self.__output = [[N for i_y in range(self.y_len)] for i_x in range(self.x_len)]

        self.mp = [
            [MPn_v3(A=A_x[i_x], B=A_y[i_y], in_len=bit_len) for i_y in range(y_len)]
            for i_x in range(x_len)
        ]

        self.elements = []
        for lay_mp in self.mp:
            self.elements += lay_mp

    def netlist(self):
        for i_x in range(self.x_len):
            for i_y in range(self.y_len):
                self.mp[i_x][i_y].A = self.A_x[i_x]
                self.mp[i_x][i_y].B = self.A_y[i_y]
                self.__output[i_x][i_y] = self.mp[i_x][i_y].output

    @property
    def change_flag(self):
        return any([i.change_flag for i in self.elements])

    @property
    def output(self):
        self.netlist()
        while self.change_flag:
            self.netlist()
        return self.__output


if __name__ == "__main__":

    x = [
        [1, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 1, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 1, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 1, 0, 0, 0, 0],  # 8
    ]
    y = [
        [0, 0, 0, 0, 1, 0, 0, 0],  # 16
        [0, 0, 0, 0, 0, 1, 0, 0],  # 32
    ]


    x_len = len(x)
    y_len = len(y)

    pe = SemiPEArray(A_x=x, A_y=y, x_len=x_len, y_len=y_len)

    for i_y in range(y_len):
        for i_x in range(x_len):
            # print(f"[{i_x}][{i_y}]: {pe.output[i_x][i_y]}")
            print(f"[{i_x}][{i_y}]: {reverse_signed_b(pe.A_x[i_x])} x {reverse_signed_b(pe.A_y[i_y])} = {reverse_signed_b(pe.output[i_x][i_y])}")
