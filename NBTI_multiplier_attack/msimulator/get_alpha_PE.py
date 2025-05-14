import time
import multiprocessing
import os
import random
import math
import time
from itertools import product

# from .Multiplier import MPn_v3, L, H
from msimulator.semi_PE import *

MAX_PROCESSES = 20  # multiprocessing.cpu_count()


class SemiPEArrayStressTest:
    def __init__(
        self,
        x_len,
        y_len,
        bit_len,
        optimizer_trigger,
        optimizer_accept,
        optimizer_enable=True,
        queue_size=100_000_000,
    ):
        self.x_len = x_len
        self.y_len = y_len
        self.bit_len = bit_len

        self.input_len = 2 ** (bit_len * (x_len + y_len))

        self.optimizer_enable = optimizer_enable
        self.queue = multiprocessing.Queue(maxsize=queue_size)

        self.optimizer_trigger = optimizer_trigger
        self.optimizer_accept = optimizer_accept

    def process_batch(self, batch, log_obj=False):
        _multiplier_stress_counter = [
            [
                {"T0": 0, "T0p": 0, "T1": 0, "T1p": 0, "T2": 0, "T2p": 0}
                for _ in range(self.bit_len)
            ]
            for _ in range(self.bit_len - 1)
        ]
        pe_stress_counter = [
            [_multiplier_stress_counter.copy() for _ in range(self.y_len)]
            for _ in range(self.x_len)
        ]

        for A_x, A_y in batch:
            A_x_b = [signed_b(i, self.bit_len) for i in A_x]
            A_y_b = [signed_b(i, self.bit_len) for i in A_y]
            # mp = MPn_v3(A_b, B_b, self.bit_len)
            # mp.output
            pe = SemiPEArray(A_x_b, A_y_b, self.bit_len, self.x_len, self.y_len)
            pe.output

            # ====================================== OPTIMIZER
            # optimize_flag = False
            # if self.optimizer_enable:
            #     if (A != -1 * 2**(self.bit_len - 1)) and (B != -1 * 2**(self.bit_len - 1)):
            #         if (self.optimizer_trigger(mp, A, B)):
            #             neg_A = -A
            #             neg_B = -B
            #             neg_mp = MPn_v3(self.signed_b(neg_A, self.bit_len), self.signed_b(neg_B, self.bit_len), self.bit_len)
            #             neg_mp.output

            #             if self.optimizer_accept(neg_mp, neg_A, neg_B):
            #                 optimize_flag = True
            #                 mp = neg_mp
            # if log_obj:
            #     log_obj.println(f"{A_b}, {B_b}, [compliment: {optimize_flag}]")

            optimize_flag = False
            if self.optimizer_enable:
                if (-1 * 2 ** (self.bit_len - 1)) not in (A_x + A_y):
                    if(self.optimizer_trigger(pe)):
                        neg_A_x = [-1*i for i in A_x]
                        neg_A_y = [-1*i for i in A_y]

                        neg_A_x_bin = [signed_b(i, self.bit_len) for i in neg_A_x]
                        neg_A_y_bin = [signed_b(i, self.bit_len) for i in neg_A_y]
                        neg_pe = SemiPEArray(neg_A_x_bin, neg_A_y_bin, self.bit_len, self.x_len, self.y_len)
                        neg_pe.output

                        if self.optimizer_accept(neg_pe):
                            optimize_flag = True
                            pe = neg_pe

            if log_obj:
                log_obj.println(f"{A_x_b}, {A_y_b}, [compliment: {optimize_flag}]")

            # ====================================== OPTIMIZER [END]

            output = pe.output
            # if output != self.signed_b(A * B, self.bit_len * 2):
            #     raise RuntimeError(f"Multiplier test failed for {A} x {B}. Output: {output} != Expected: {A * B}")

            for pe_x in range(self.x_len):
                for pe_y in range(self.y_len):
                    for lay in range(self.bit_len - 1):
                        for index in range(self.bit_len):
                            T0 = pe.mp[pe_x][pe_y].gfa[lay][index].tgate[0].p0.gate
                            T0p = pe.mp[pe_x][pe_y].gfa[lay][index].tgate[0].p1.gate
                            T1 = pe.mp[pe_x][pe_y].gfa[lay][index].tgate[1].p0.gate
                            T1p = pe.mp[pe_x][pe_y].gfa[lay][index].tgate[1].p1.gate
                            T2 = pe.mp[pe_x][pe_y].gfa[lay][index].tgate[2].p0.gate
                            T2p = pe.mp[pe_x][pe_y].gfa[lay][index].tgate[2].p1.gate

                            pe_stress_counter[pe_x][pe_y][lay][index]["T0"] += not T0
                            pe_stress_counter[pe_x][pe_y][lay][index]["T0p"] += not T0p
                            pe_stress_counter[pe_x][pe_y][lay][index]["T1"] += not T1
                            pe_stress_counter[pe_x][pe_y][lay][index]["T1p"] += not T1p
                            pe_stress_counter[pe_x][pe_y][lay][index]["T2"] += not T2
                            pe_stress_counter[pe_x][pe_y][lay][index]["T2p"] += not T2p

        self.queue.put(pe_stress_counter)

    # def generate_batches(self, batch_size):
    #     _range = range(-1 * 2**(bit_len - 1), 2**(bit_len - 1))
    #     batch = []
    #     for A in _range:
    #         for B in _range:
    #             batch.append((A, B))
    #             if len(batch) >= batch_size:
    #                 yield batch
    #                 batch = []
    #     if batch:
    #         yield batch

    def generate_batches(self, batch_size):
        element_range = range(-1 * 2 ** (self.bit_len - 1), 2 ** (self.bit_len - 1))
        A_patterns = product(element_range, repeat=self.x_len)

        total_patterns = len(element_range)**(self.x_len + self.y_len)
        print("total pattern", total_patterns)
        processed_count = 0

        batch = []
        for A in A_patterns:
            B_patterns = product(element_range, repeat=self.y_len)
            for B in B_patterns:
                # print(f"{A}, {B}")
                batch.append((list(A), list(B)))

                processed_count += 1
                #print("batch count\t", processed_count);
                if processed_count % (total_patterns // 100) == 0:
                    percent_complete = (processed_count / total_patterns) * 100
                    print(f"Progress: {math.ceil(percent_complete)}%")
                    print("count:", processed_count, "/", total_patterns)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def process_inputs_in_batches(self, batch_size, log_obj=False):
        _multiplier_stress_counter = [
            [
                {"T0": 0, "T0p": 0, "T1": 0, "T1p": 0, "T2": 0, "T2p": 0}
                for _ in range(self.bit_len)
            ]
            for _ in range(self.bit_len - 1)
        ]
        pe_total_stress_counter = [
            [_multiplier_stress_counter.copy() for _ in range(self.y_len)]
            for _ in range(self.x_len)
        ]

        processes = []
        batch_generator = self.generate_batches(batch_size)
        for batch in batch_generator:
            p = multiprocessing.Process(
                target=self.process_batch, args=(batch,), kwargs={"log_obj": log_obj}
            )
            processes.append(p)
            p.start()

            if len(processes) >= MAX_PROCESSES:
                for p in processes:
                    p.join()
                while not self.queue.empty():
                    stress_counter = self.queue.get()

                    for pe_x in range(self.x_len):
                        for pe_y in range(self.y_len):
                            for lay in range(self.bit_len - 1):
                                for index in range(self.bit_len):
                                    for key in pe_total_stress_counter[pe_x][pe_y][lay][
                                        index
                                    ]:
                                        pe_total_stress_counter[pe_x][pe_y][lay][index][
                                            key
                                        ] += stress_counter[pe_x][pe_y][lay][index][key]
                processes = []

        for p in processes:
            p.join()
        while not self.queue.empty():
            stress_counter = self.queue.get()
            for pe_x in range(self.x_len):
                for pe_y in range(self.y_len):
                    for lay in range(self.bit_len - 1):
                        for index in range(self.bit_len):
                            for key in pe_total_stress_counter[pe_x][pe_y][lay][index]:
                                pe_total_stress_counter[pe_x][pe_y][lay][index][
                                    key
                                ] += stress_counter[pe_x][pe_y][lay][index][key]

        return pe_total_stress_counter

    def run(self, batch_size=4096, log_obj=False):
        start_time = time.time()  # Record end time

        alpha_lst = self.process_inputs_in_batches(batch_size, log_obj=log_obj)

        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate total time taken
        print(f"Execution time: {execution_time:.4f} seconds")

        for pe_x in range(self.x_len):
            for pe_y in range(self.y_len):
                for i in range(self.bit_len - 1):
                    for j in range(self.bit_len):
                        alpha_lst[pe_x][pe_y][i][j] = [
                            v / self.input_len
                            for v in alpha_lst[pe_x][pe_y][i][j].values()
                        ]

        return alpha_lst


# from multiplier_lib import MultiplierStressTest
import time

if __name__ == "__main__":
    exit()

    bit_len = 8
    optimizer_enable = True
    batch_size = 1000

    def optimizer_trigger(mp: MPn_v3):
        # return mp.gfa[2][3].tgate[0].p0.gate == L
        return False

    def optimizer_accept(mp: MPn_v3):
        # return mp.gfa[2][3].tgate[0].p0.gate == H
        return False

    start_time = time.time()

    # Create an instance of MultiplierStressTest
    stress_test = SemiPEArrayStressTest(
        1,
        1,
        bit_len,
        optimizer_trigger=optimizer_trigger,
        optimizer_accept=optimizer_accept,
        optimizer_enable=True,
    )

    # Run the stress test
    stress_result = stress_test.run(batch_size)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # print(f"Stress result: {stress_result}")
    # for lay in range(bit_len-1):
    #     print(f"{[[i/(2**(bit_len*2)) for i in stress.values()] for stress in stress_result[lay]]}, ")
    print(stress_result)

    print(f"Total execution time: {elapsed_time:.2f} seconds")
