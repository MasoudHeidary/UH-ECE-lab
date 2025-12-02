
import random

from .bin_func import *
from .Adder import RippleAdder




class AdderShrinked:
    def __init__(self, bit_len, raw_mp):
        self.bit_len = bit_len
        self.raw_mp = raw_mp


    def get_total_stress(self, sample=False, log_obj=False):
        """supports both sampling, and both full iteration"""
        sample_counter = 0
        limit = 2 ** self.bit_len
        stress_counter = [{'T0': 0, 'T1': 0} for _ in range(self.bit_len)]

        if sample == False:
            """full iteration"""
            for A in range(0, limit):
                for B in range(0, limit):
                    a_bin = ubin(A, self.bit_len)
                    b_bin = ubin(B, self.bit_len)

                    # adder = RippleAdder(a_bin, b_bin, self.bit_len)
                    adder = self.raw_mp(A=a_bin, B=b_bin, bit_len=self.bit_len)

                    """take overflow bit as carry-out bit"""
                    # if adder.overflow:
                    #     "this input combination raises overflow case"
                    #     continue

                    sample_counter += 1
                    out_bin = adder.sum + [adder.overflow]
                    for fa_i in range(self.bit_len):
                        T0 = adder.gfa[fa_i].p[0]
                        T1 = adder.gfa[fa_i].p[2]

                        stress_counter[fa_i]['T0'] += (not T0)
                        stress_counter[fa_i]['T1'] += (not T1)
        
        else:
            """sampling iteration"""
            for sample_i in range(sample):
                A = random.randint(0, limit)
                B = random.randint(0, limit)

                # adder = RippleAdder(a_bin, b_bin, self.bit_len)
                adder = self.raw_mp(a_bin, b_bin, self.bit_len)

                # if adder.overflow:
                #     "this input combination raises overflow case"
                #     continue

                sample_counter += 1
                out_bin = adder.sum + [adder.overflow]
                for fa_i in range(self.bit_len):
                    T0 = adder.gfa[fa_i].p[0]
                    T1 = adder.gfa[fa_i].p[2]

                    stress_counter[fa_i]['T0'] += (not T0)
                    stress_counter[fa_i]['T1'] += (not T1)

        # uncompress the shrinked alpha
        alpha_lst = stress_counter
        for i in range(self.bit_len):
            alpha_lst[i] = [v / sample_counter for v in alpha_lst[i].values()]

            alpha_lst[i] = [
                alpha_lst[i][0],
                1 - alpha_lst[i][0],
                alpha_lst[i][1],
                1 - alpha_lst[i][1],
                1 - alpha_lst[i][1],
                alpha_lst[i][1],
                ]

        return alpha_lst





class CarrySaveAdderShrinked:
    def __init__(self, bit_len, raw_mp):
        self.bit_len = bit_len
        self.raw_mp = raw_mp

    def get_total_stress(self, sample=False, log_obj=False):
        """supports both sampling, and both full iteration"""
        sample_counter = 0
        limit = 2 ** self.bit_len
        stress_counter = [[{'T0': 0, 'T1': 0} for _ in range(self.bit_len)] for _ in range(2)]

        if sample == False:
            """full iteration"""
            for A in range(0, limit):
                for B in range(0, limit):
                    a_bin = ubin(A, self.bit_len)
                    b_bin = ubin(B, self.bit_len)

                    # adder = RippleAdder(a_bin, b_bin, self.bit_len)
                    adder = self.raw_mp(A=a_bin, B=b_bin, bit_len=self.bit_len)

                    """take overflow bit as carry-out bit"""
                    # if adder.overflow:
                    #     "this input combination raises overflow case"
                    #     continue

                    sample_counter += 1
                    out_bin = adder.sum + [adder.overflow]
                    for row in range(2):
                        for fa_i in range(self.bit_len):
                            T0 = adder.gfa[row][fa_i].p[0]
                            T1 = adder.gfa[row][fa_i].p[2]

                            stress_counter[row][fa_i]['T0'] += (not T0)
                            stress_counter[row][fa_i]['T1'] += (not T1)
        
        else:
            """sampling iteration"""
            raise NotImplementedError()
            for sample_i in range(sample):
                A = random.randint(0, limit)
                B = random.randint(0, limit)

                # adder = RippleAdder(a_bin, b_bin, self.bit_len)
                adder = self.raw_mp(a_bin, b_bin, self.bit_len)

                # if adder.overflow:
                #     "this input combination raises overflow case"
                #     continue

                sample_counter += 1
                out_bin = adder.sum + [adder.overflow]
                for row in range(2):
                    for fa_i in range(self.bit_len):
                        T0 = adder.gfa[row][fa_i].p[0]
                        T1 = adder.gfa[row][fa_i].p[2]

                        stress_counter[row][fa_i]['T0'] += (not T0)
                        stress_counter[row][fa_i]['T1'] += (not T1)

        # uncompress the shrinked alpha
        alpha_lst = stress_counter
        for row in range(2):
            for i in range(self.bit_len):
                alpha_lst[row][i] = [v / sample_counter for v in alpha_lst[row][i].values()]

                alpha_lst[row][i] = [
                    alpha_lst[row][i][0],
                    1 - alpha_lst[row][i][0],
                    alpha_lst[row][i][1],
                    1 - alpha_lst[row][i][1],
                    1 - alpha_lst[row][i][1],
                    alpha_lst[row][i][1],
                    ]

        return alpha_lst