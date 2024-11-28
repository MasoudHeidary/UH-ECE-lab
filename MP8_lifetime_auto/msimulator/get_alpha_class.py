import time
import multiprocessing
import os
import random
from .Multiplier import MPn_v3, L, H

MAX_PROCESSES = 20 #multiprocessing.cpu_count()

class MultiplierStressTest:
    def __init__(self, bit_len, optimizer_trigger, optimizer_accept, optimizer_enable=True, queue_size=100_000_000):
        self.bit_len = bit_len
        self.input_len = 2**(self.bit_len * 2)
        self.optimizer_enable = optimizer_enable
        self.queue = multiprocessing.Queue(maxsize=queue_size)

        self.optimizer_trigger = optimizer_trigger
        self.optimizer_accept = optimizer_accept

    @staticmethod
    def signed_b(num: int, bit_len: int):
        num_cpy = num
        if num < 0:
            num_cpy = 2**bit_len + num
        bit_num = list(map(int, reversed(format(num_cpy, f'0{bit_len}b'))))

        if (num > 0) and (bit_num[-1] != 0):
            raise OverflowError(f"number {num} can't fit in signed #{bit_len} bits")
        if (num < 0) and (bit_num[-1] != 1):
            raise OverflowError(f"number {num} can't fit in signed #{bit_len} bits")
        return bit_num

    @staticmethod
    def reverse_signed_b(binary_list):
        binary_str = ''.join(map(str, reversed(binary_list)))
        num = int(binary_str, 2)

        if binary_list[-1] == 1:
            num = num - (2**len(binary_list))
        return num

    def process_batch(self, batch, log_obj=False):
        stress_counter = [
            [{'T0': 0, 'T0p': 0, 'T1': 0, 'T1p': 0, 'T2': 0, 'T2p': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]

        for A, B in batch:
            A_b = self.signed_b(A, self.bit_len)
            B_b = self.signed_b(B, self.bit_len)
            mp = MPn_v3(A_b, B_b, self.bit_len)
            mp.output

            # ====================================== OPTIMIZER
            optimize_flag = False
            if self.optimizer_enable:
                if (A != -1 * 2**(self.bit_len - 1)) and (B != -1 * 2**(self.bit_len - 1)):
                    if (self.optimizer_trigger(mp, A, B)):
                        neg_A = -A
                        neg_B = -B
                        neg_mp = MPn_v3(self.signed_b(neg_A, self.bit_len), self.signed_b(neg_B, self.bit_len), self.bit_len)
                        neg_mp.output

                        if self.optimizer_accept(neg_mp, neg_A, neg_B):
                            optimize_flag = True
                            mp = neg_mp
                else:
                    optimize_flag = "edge"
                    
            if log_obj:
                log_obj.println(f"{A_b}, {B_b}, [compliment: {optimize_flag}]")
                # log_obj.println(f"{A}, {B}, [compliment: {optimize_flag}]")
            
            # ====================================== OPTIMIZER [END]

            output = mp.output
            if output != self.signed_b(A * B, self.bit_len * 2):
                raise RuntimeError(f"Multiplier test failed for {A} x {B}. Output: {output} != Expected: {A * B}")

            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):
                    T0 = mp.gfa[lay][index].tgate[0].p0.gate
                    T0p = mp.gfa[lay][index].tgate[0].p1.gate
                    T1 = mp.gfa[lay][index].tgate[1].p0.gate
                    T1p = mp.gfa[lay][index].tgate[1].p1.gate
                    T2 = mp.gfa[lay][index].tgate[2].p0.gate
                    T2p = mp.gfa[lay][index].tgate[2].p1.gate

                    stress_counter[lay][index]['T0'] += (not T0)
                    stress_counter[lay][index]['T0p'] += (not T0p)
                    stress_counter[lay][index]['T1'] += (not T1)
                    stress_counter[lay][index]['T1p'] += (not T1p)
                    stress_counter[lay][index]['T2'] += (not T2)
                    stress_counter[lay][index]['T2p'] += (not T2p)

        self.queue.put(stress_counter)

    @staticmethod
    def generate_batches(bit_len, batch_size):
        _range = range(-1 * 2**(bit_len - 1), 2**(bit_len - 1))
        batch = []
        for A in _range:
            for B in _range:
                batch.append((A, B))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def process_inputs_in_batches(self, batch_size, log_obj=False):
        total_stress_counter = [
            [{'T0': 0, 'T0p': 0, 'T1': 0, 'T1p': 0, 'T2': 0, 'T2p': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]

        processes = []
        batch_generator = self.generate_batches(self.bit_len, batch_size)
        for batch in batch_generator:
            p = multiprocessing.Process(target=self.process_batch, args=(batch, ), kwargs={'log_obj': log_obj})
            processes.append(p)
            p.start()

            if len(processes) >= MAX_PROCESSES:
                for p in processes:
                    p.join()
                while not self.queue.empty():
                    stress_counter = self.queue.get()
                    for lay in range(self.bit_len - 1):
                        for index in range(self.bit_len):
                            for key in total_stress_counter[lay][index]:
                                total_stress_counter[lay][index][key] += stress_counter[lay][index][key]
                processes = []

        for p in processes:
            p.join()
        while not self.queue.empty():
            stress_counter = self.queue.get()
            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):
                    for key in total_stress_counter[lay][index]:
                        total_stress_counter[lay][index][key] += stress_counter[lay][index][key]

        return total_stress_counter

    def run(self, batch_size=4096, log_obj=False):
        start_time = time.time()  # Record end time

        alpha_lst = self.process_inputs_in_batches(batch_size, log_obj=log_obj)

        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate total time taken
        print(f"Execution time: {execution_time:.4f} seconds") 

        for i in range(self.bit_len - 1):
            for j in range(self.bit_len):
                alpha_lst[i][j] = [v / self.input_len for v in alpha_lst[i][j].values()]

        return alpha_lst







# from multiplier_lib import MultiplierStressTest
import time

if __name__ == "__main__":
    bit_len = 8
    optimizer_enable = True
    batch_size = 1000 


    def optimizer_trigger(mp: MPn_v3):
        return mp.gfa[2][3].tgate[0].p0.gate == L
    def optimizer_accept(mp: MPn_v3):
        return mp.gfa[2][3].tgate[0].p0.gate == H


    start_time = time.time()

    # Create an instance of MultiplierStressTest
    stress_test = MultiplierStressTest(bit_len, optimizer_trigger=optimizer_trigger, optimizer_accept=optimizer_accept, optimizer_enable=True)

    # Run the stress test
    stress_result = stress_test.run(batch_size)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # print(f"Stress result: {stress_result}")
    for lay in range(bit_len-1):
        print(f"{[[i/(2**(bit_len*2)) for i in stress.values()] for stress in stress_result[lay]]}, ")
       

    print(f"Total execution time: {elapsed_time:.2f} seconds")
