import time
import random as rnd
import multiprocessing

from .Multiplier import Wallace_comp, L, H


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


class AlphaWallaceShrinked:
    def __init__(self, bit_len, optimizer_trigger, optimizer_accept, optimizer_enable=True):
        self.bit_len = bit_len
        self.optimizer_enable = optimizer_enable

        self.optimizer_trigger = optimizer_trigger
        self.optimizer_accept = optimizer_accept


    def process_input_pair(self, input_pair, log_obj=False):


        A, B = input_pair[0], input_pair[1]
        A_b = signed_b(A, self.bit_len)
        B_b = signed_b(B, self.bit_len)
        mp = Wallace_comp(A_b, B_b, self.bit_len)
        mp.output

        # ====================================== OPTIMIZER
        optimize_flag = False
        if self.optimizer_enable:
            if (A != -1 * 2**(self.bit_len - 1)) and (B != -1 * 2**(self.bit_len - 1)):
                if (self.optimizer_trigger(mp)):
                    neg_A = -A
                    neg_B = -B
                    neg_mp = Wallace_comp(signed_b(neg_A, self.bit_len), signed_b(neg_B, self.bit_len), self.bit_len)
                    neg_mp.output

                    if self.optimizer_accept(neg_mp):
                        optimize_flag = True
                        mp = neg_mp
                
        # if log_obj:
            # log_obj.println(f"{A_b}, {B_b}, [compliment: {optimize_flag}]")
            # log_obj.println(f"{A}, {B}, [compliment: {optimize_flag}]")
        
        # ====================================== OPTIMIZER [END]

        output = mp.output
        # if output != signed_b(A * B, self.bit_len * 2):
            # raise RuntimeError(f"Multiplier test failed for {A} x {B}. Output: {output} != Expected: {A * B}")

        stress_counter = [
            [{'T0': 0, 'T1': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]
        for lay in range(self.bit_len - 1):
            for index in range(self.bit_len):
                    T0 = mp.gfa[lay][index].p[0]
                    T1 = mp.gfa[lay][index].p[2]

                    stress_counter[lay][index]['T0'] += (not T0)
                    stress_counter[lay][index]['T1'] += (not T1)
        return stress_counter


    def get_total_stress(self, sample_count, rnd_seed, log_obj=False):
        total_stress_counter = [
            [{'T0': 0, 'T1': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]
        range_min = -1 * 2**(self.bit_len - 1)
        range_max = 2**(self.bit_len - 1) - 1

        rnd.seed(rnd_seed)
        # seen = list()
        sample_i = 0

        while sample_i < sample_count:
            sample_i += 1

            A, B = rnd.randint(range_min, range_max), rnd.randint(range_min, range_max)
            # if (A, B) in seen:
                # sample_i -= 1
                # continue
            # seen.append((A, B)) 

            if sample_i % 10_000 == 0:
                print(f"{sample_i} / {sample_count}")

            stress_counter = self.process_input_pair((A, B), log_obj=log_obj)
            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):
                    for key in total_stress_counter[lay][index]:
                        total_stress_counter[lay][index][key] += stress_counter[lay][index][key]

        return total_stress_counter
    
    
    
    ############# MULTI PROCESS WORK
    def _stress_worker(self, sample_count, rnd_seed, proc_id=None):
        total_stress_counter = [
            [{'T0': 0, 'T1': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]
        range_min = -1 * 2**(self.bit_len - 1)
        range_max = 2**(self.bit_len - 1) - 1

        rnd.seed(rnd_seed)
        sample_i = 0

        while sample_i < sample_count:
            A, B = rnd.randint(range_min, range_max), rnd.randint(range_min, range_max)
            sample_i += 1
            
            if sample_i % 10_000 == 0:
                print(f"{sample_i} / {sample_count} ({sample_i/sample_count*100:.2f}%)", flush=True)

            stress_counter = self.process_input_pair((A, B))
            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):
                    for key in total_stress_counter[lay][index]:
                        total_stress_counter[lay][index][key] += stress_counter[lay][index][key]

        print(f"Process {proc_id} finished with {sample_count} samples.")
        return total_stress_counter


    def merge_counters(self, counters_list):
        merged = [
            [{'T0': 0, 'T1': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]

        for counter in counters_list:
            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):
                    merged[lay][index]['T0'] += counter[lay][index]['T0']
                    merged[lay][index]['T1'] += counter[lay][index]['T1']
        return merged
    #############

    def run(self, sample_count, rnd_seed, log_obj=False):
        start_time = time.time()

        alpha_lst = self.get_total_stress(sample_count, rnd_seed, log_obj=log_obj)

        end_time = time.time()
        if log_obj:
            log_obj.println(f"Execution time: {end_time - start_time:.4f} seconds") 

        # uncompress the shrinked alpha
        for i in range(self.bit_len - 1):
            for j in range(self.bit_len):
                alpha_lst[i][j] = [v / sample_count for v in alpha_lst[i][j].values()]

                alpha_lst[i][j] = [
                    alpha_lst[i][j][0],
                    1-alpha_lst[i][j][0],
                    alpha_lst[i][j][1],
                    1-alpha_lst[i][j][1],
                    1-alpha_lst[i][j][1],
                    alpha_lst[i][j][1],
                    ]

        return alpha_lst
    
    
    def run_multi(self, sample_count, rnd_seed, log_obj=False, proc_count=20):
        start_time = time.time()

        sample_per_proc = sample_count // proc_count
        # seeds = [rnd_seed * i for i in range(proc_count)]
        seeds = [rnd_seed * (self.bit_len**i) for i in range(proc_count)]

        with multiprocessing.Pool(proc_count) as pool:
            results = pool.starmap(self._stress_worker, [(sample_per_proc, seeds[i], i) for i in range(proc_count)])

        alpha_lst = self.merge_counters(results)

        end_time = time.time()
        if log_obj:
            log_obj.println(f"Execution time: {end_time - start_time:.4f} seconds") 

        # uncompress the shrinked alpha
        for i in range(self.bit_len - 1):
            for j in range(self.bit_len):
                alpha_lst[i][j] = [v / sample_count for v in alpha_lst[i][j].values()]

                alpha_lst[i][j] = [
                    alpha_lst[i][j][0],
                    1-alpha_lst[i][j][0],
                    alpha_lst[i][j][1],
                    1-alpha_lst[i][j][1],
                    1-alpha_lst[i][j][1],
                    alpha_lst[i][j][1],
                    ]

        return alpha_lst


