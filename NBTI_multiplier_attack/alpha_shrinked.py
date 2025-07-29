import time
import multiprocessing
from msimulator.bin_func import signed_b, reverse_signed_b
import random as rnd


MAX_PROCESSES = 10

class AlphaShrinkedMultiprocess:
    def __init__(self, raw_mp, bit_len, log=False, rew_lst=[], queue_size=100_000_000):
        self.raw_mp = raw_mp
        self.bit_len = bit_len
        self.input_len = 2**(self.bit_len * 2)
        
        self.log_obj = log
        self.rew_lst = rew_lst
        self.queue = multiprocessing.Queue(maxsize=queue_size)

    def process_batch(self, batch, conn):
        stress_counter = [
            [{'T0': 0, 'T1': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]

        for A, B in batch:
            A_b = signed_b(A, self.bit_len)
            B_b = signed_b(B, self.bit_len)
            mp = self.raw_mp(A_b, B_b, self.bit_len, self.rew_lst)
            output = mp.output

            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):

                    # if eFA
                    T0 = mp.gfa[lay][index].p[0]
                    T1 = mp.gfa[lay][index].p[2]
                    
                    stress_counter[lay][index]['T0'] += (not T0)
                    stress_counter[lay][index]['T1'] += (not T1)
    
        # self.queue.put(stress_counter)
        conn.send(stress_counter)
        conn.close()
        # return stress_counter

    def generate_batches(self, batch_size):
        _range = range(-1 * 2**(self.bit_len - 1), 2**(self.bit_len - 1))
        batch = []
        for A in _range:
            if self.log_obj:
                self.log_obj.println(f"A [{A:7,}]")
            for B in _range:
                batch.append((A, B))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def process_inputs_in_batches(self, batch_size):
        total_stress_counter = [
            [{'T0': 0, 'T1': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]

        # return total_stress_counter
        batch_generator = self.generate_batches(batch_size)
        processes = []
        conns = []

        for batch in batch_generator:
            parent_conn, child_conn = multiprocessing.Pipe()
            p = multiprocessing.Process(target=self.process_batch, args=(batch, child_conn))
            processes.append(p)
            conns.append(parent_conn)
            p.start()

            if len(processes) >= MAX_PROCESSES:
                for p in processes:
                    p.join()

                for conn in conns:
                    stress_counter = conn.recv()
                    for lay in range(self.bit_len - 1):
                        for index in range(self.bit_len):
                            for key in total_stress_counter[lay][index]:
                                total_stress_counter[lay][index][key] += stress_counter[lay][index][key]

                processes = []
                conns = []

        # Final flush for any remaining processes
        for p in processes:
            p.join()
        for conn in conns:
            stress_counter = conn.recv()
            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):
                    for key in total_stress_counter[lay][index]:
                        total_stress_counter[lay][index][key] += stress_counter[lay][index][key]

        return total_stress_counter

    def run(self, batch_size=2**12):
        start_time = time.time()  # Record end time

        alpha_lst = self.process_inputs_in_batches(batch_size)

        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate total time taken
        if self.log_obj:
            self.log_obj.println(f"Execution time: {execution_time:.4f} seconds") 

        for i in range(self.bit_len - 1):
            for j in range(self.bit_len):
                alpha_lst[i][j] = [v / self.input_len for v in alpha_lst[i][j].values()]

                alpha_lst[i][j] = [
                    alpha_lst[i][j][0],
                    1-alpha_lst[i][j][0],
                    alpha_lst[i][j][1],
                    1-alpha_lst[i][j][1],
                    1-alpha_lst[i][j][1],
                    alpha_lst[i][j][1],
                    ]
                
        return alpha_lst


"""using random samples to calculate the alpha"""
class AlphaSampled:
    def __init__(self, raw_mp, bit_len, rew_lst=[]):
        self.raw_mp = raw_mp
        self.bit_len = bit_len
        self.rew_lst = rew_lst

    def process_input_pair(self, input_pair):
        A, B = input_pair[0], input_pair[1]
        A_b = signed_b(A, self.bit_len)
        B_b = signed_b(B, self.bit_len)
        mp = self.raw_mp(A_b, B_b, self.bit_len, self.rew_lst)
        mp.output

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


    def get_total_stress(self, sample_count, rnd_seed, log=False):
        total_stress_counter = [
            [{'T0': 0, 'T1': 0} for _ in range(self.bit_len)] 
            for _ in range(self.bit_len - 1)
        ]
        range_min = -1 * 2**(self.bit_len - 1)
        range_max = 2**(self.bit_len - 1) - 1

        rnd.seed(rnd_seed)
        sample_i = 0

        while sample_i < sample_count:
            sample_i += 1

            A, B = rnd.randint(range_min, range_max), rnd.randint(range_min, range_max)
            if log and (sample_i % 10_000 == 0):
                log.println(f"{sample_i} / {sample_count}")

            stress_counter = self.process_input_pair((A, B))
            for lay in range(self.bit_len - 1):
                for index in range(self.bit_len):
                    for key in total_stress_counter[lay][index]:
                        total_stress_counter[lay][index][key] += stress_counter[lay][index][key]

        return total_stress_counter
    
    ############# MULTI PROCESS WORK
    def _stress_worker(self, sample_count, rnd_seed, proc_id=None, log=False):
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
            
            if log and (sample_i % 10_000 == 0):
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

    def run_single(self, sample_count, rnd_seed, log=False):
        start_time = time.time()

        alpha_lst = self.get_total_stress(sample_count, rnd_seed)

        end_time = time.time()
        if log:
            log.println(f"Execution time: {end_time - start_time:.4f} seconds") 

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
    
    
    def run_multi(self, sample_count, rnd_seed, log=False, proc_count=20):
        if log:
            start_time = time.time()

        sample_per_proc = sample_count // proc_count
        seeds = [rnd_seed * (self.bit_len**i) for i in range(proc_count)]
        with multiprocessing.Pool(proc_count) as pool:
            results = pool.starmap(self._stress_worker, [(sample_per_proc, seeds[i], i, bool(log)) for i in range(proc_count)])
        alpha_lst = self.merge_counters(results)

        if log:
            end_time = time.time()
            log.println(f"Execution time: {end_time - start_time:.4f} seconds") 

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

