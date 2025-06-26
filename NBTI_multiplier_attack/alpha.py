import time
import multiprocessing

from msimulator.bin_func import signed_b, reverse_signed_b

INPUT_PIN_ALPHA = True
MAX_PROCESSES = 15 #multiprocessing.cpu_count()

class AlphaMultiprocess:
    def __init__(self, raw_mp, bit_len, log=False, rew_lst=[], queue_size=100_000_000):
        self.raw_mp = raw_mp
        self.bit_len = bit_len
        self.input_len = 2**(self.bit_len * 2)
        
        self.log_obj = log
        self.rew_lst = rew_lst
        self.queue = multiprocessing.Queue(maxsize=queue_size)

    def process_batch(self, batch, log_obj=False):
        if INPUT_PIN_ALPHA:
            stress_counter = [
                [{'T0': 0, 'T0p': 0, 'T1': 0, 'T1p': 0, 'T2': 0, 'T2p': 0, 'A':0, 'B':0, 'C':0, 'sum':0, 'carry':0} for _ in range(self.bit_len)] 
                for _ in range(self.bit_len - 1)
            ]
        else:
            stress_counter = [
                [{'T0': 0, 'T0p': 0, 'T1': 0, 'T1p': 0, 'T2': 0, 'T2p': 0} for _ in range(self.bit_len)] 
                for _ in range(self.bit_len - 1)
            ]

        for A, B in batch:
            A_b = signed_b(A, self.bit_len)
            B_b = signed_b(B, self.bit_len)
            mp = self.raw_mp(A_b, B_b, self.bit_len, self.rew_lst)

            output = mp.output
            # if output != signed_b(A * B, self.bit_len * 2):
            #     raise RuntimeError(f"Multiplier test failed for {A} x {B}. Output: {output} != Expected: {A * B}")

            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):

                    # if eFA
                    T0 = mp.gfa[lay][index].p[0]
                    T0p = mp.gfa[lay][index].p[1]
                    T1 = mp.gfa[lay][index].p[2]
                    T1p = mp.gfa[lay][index].p[3]
                    T2 = mp.gfa[lay][index].p[4]
                    T2p = mp.gfa[lay][index].p[5]
                    
                    stress_counter[lay][index]['T0'] += (not T0)
                    stress_counter[lay][index]['T0p'] += (not T0p)
                    stress_counter[lay][index]['T1'] += (not T1)
                    stress_counter[lay][index]['T1p'] += (not T1p)
                    stress_counter[lay][index]['T2'] += (not T2)
                    stress_counter[lay][index]['T2p'] += (not T2p)
                    
                    if INPUT_PIN_ALPHA:
                        A_st = mp.gfa[lay][index].A
                        B_st = mp.gfa[lay][index].B
                        C_st = mp.gfa[lay][index].C
                        sum_st = mp.gfa[lay][index].sum
                        carry_st = mp.gfa[lay][index].carry

                        
                        stress_counter[lay][index]['A'] += (not A_st)
                        stress_counter[lay][index]['B'] += (not B_st)
                        stress_counter[lay][index]['C'] += (not C_st)
                        stress_counter[lay][index]['sum'] += (not sum_st)
                        stress_counter[lay][index]['carry'] += (not carry_st)

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
        if INPUT_PIN_ALPHA:
            total_stress_counter = [
                [{'T0': 0, 'T0p': 0, 'T1': 0, 'T1p': 0, 'T2': 0, 'T2p': 0, 'A':0, 'B':0, 'C':0, 'sum':0, 'carry':0} for _ in range(self.bit_len)] 
                for _ in range(self.bit_len - 1)
            ]
        else:
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

    def run(self, batch_size=2**12):
        start_time = time.time()  # Record end time

        alpha_lst = self.process_inputs_in_batches(batch_size, log_obj=self.log_obj)

        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate total time taken
        if self.log_obj:
            self.log_obj.println(f"Execution time: {execution_time:.4f} seconds") 

        for i in range(self.bit_len - 1):
            for j in range(self.bit_len):
                alpha_lst[i][j] = [v / self.input_len for v in alpha_lst[i][j].values()]

        return alpha_lst



# single process alpha
def get_alpha(raw_mp, bit_len, log=False, rew_lst=[], verify=False):
    alpha_row = bit_len-1
    alpha_index = bit_len
    alpha = [
        [
            [0 for _ in range(6)] 
            for _ in range(alpha_index)
        ]
        for _ in range(alpha_row)
    ]

    limit = 2 ** (bit_len - 1)
    for a in range(-limit, limit):

        for b in range(-limit, limit):

            a_bin = signed_b(a, bit_len)
            b_bin = signed_b(b, bit_len)

            # mp: MPn_rew
            mp = raw_mp(a_bin, b_bin, bit_len, rew_lst)
            mp.output

            if verify:
                out = reverse_signed_b(mp.output)
                if a * b != out:
                    raise ValueError(f"output verification failed, {a} * {b} != {out}")

            for row in range(alpha_row):
                for index in range(alpha_index):
                    for t in range(6):
                        alpha[row][index][t] += (not mp.gfa[row][index].p[t])

    # alpha couter -> alpha probability
    for row in range(alpha_row):
        for index in range(alpha_index):
            for t in range(6):
                alpha[row][index][t] /= ((2*limit)**2)

                # intercorrection, alpha 0 OR 1 -> 0.5
                if alpha[row][index][t] in [0, 1]:
                    alpha[row][index][t] = 0.5

    return alpha

