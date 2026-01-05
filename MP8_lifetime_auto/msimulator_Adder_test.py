from msimulator.Adder import *

def __test__ripple_adder(bit_len=8, debug=False):
    # NOTE: adder output register is one-bit bigger than input-range
    st = time.time()
    
    limit = 2 ** (bit_len)
    sample_count = 0
    for A in range(0, limit):
        for B in range(0, limit):
            a_bin = ubin(A, bit_len)
            b_bin = ubin(B, bit_len)
            adder = RippleAdder(a_bin, b_bin, bit_len)

            """take overflow bit as carry-out bit"""
            # if adder.overflow:
            #     continue

            out_bin = adder.sum + [adder.overflow]
            sum = rev_ubin(out_bin)

            expected_sum = A+B
            if debug:
                print(f"{A:5} + {B:5} = {sum:7} | {a_bin} + {b_bin} = {out_bin}")
            
            if sum != expected_sum:
                raise RuntimeError(f" {A}+{B} = {sum} [!={expected_sum}] - TEST FAILED")
            sample_count += 1


    et = time.time()
    print(f"__test__ >> ripple_adder[{bit_len}] >> sample[{sample_count}] >> True [{et-st}s]")
    return True

def __test__carry_skip_adder(bit_len=8, debug=False):
    # NOTE: adder output register is one-bit bigger than input-range
    st = time.time()
    
    limit = 2 ** (bit_len)
    sample_count = 0
    for A in range(0, limit):
        for B in range(0, limit):
            a_bin = ubin(A, bit_len)
            b_bin = ubin(B, bit_len)
            adder = CarrySkipAdder(a_bin, b_bin, bit_len)

            """take overflow bit as carry-out bit"""
            # if adder.overflow:
            #     continue

            out_bin = adder.sum + [adder.overflow]
            sum = rev_ubin(out_bin)

            expected_sum = A+B
            if debug:
                print(f"{A:5} + {B:5} = {sum:7} | {a_bin} + {b_bin} = {out_bin}")
            
            if sum != expected_sum:
                raise RuntimeError(f" {A}+{B} = {sum} [!={expected_sum}] - TEST FAILED")
            sample_count += 1


    et = time.time()
    print(f"__test__ >> carry_skip_adder[{bit_len}] >> sample[{sample_count}] >> True [{et-st}s]")
    return True


def __test__carry_save_adder(bit_len=8, debug=False):
    # NOTE: adder output register is one-bit bigger than input-range
    st = time.time()
    
    limit = 2 ** (bit_len)
    sample_count = 0
    for A in range(0, limit):
        for B in range(0, limit):
            a_bin = ubin(A, bit_len)
            b_bin = ubin(B, bit_len)
            adder = CarrySaveAdder(A=a_bin, B=b_bin, bit_len=bit_len)

            """take overflow bit as carry-out bit"""
            # if adder.overflow:
            #     continue

            out_bin = adder.sum + [adder.overflow]
            sum = rev_ubin(out_bin)

            expected_sum = A+B
            if debug:
                print(f"{A:5} + {B:5} = {sum:7} | {a_bin} + {b_bin} = {out_bin}")
            
            if sum != expected_sum:
                raise RuntimeError(f" {A}+{B} = {sum} [!={expected_sum}] - TEST FAILED")
            sample_count += 1


    et = time.time()
    print(f"__test__ >> carry_save_adder[{bit_len}] >> sample[{sample_count}] >> True [{et-st}s]")
    return True


if __name__ == "__main__":
    __test__ripple_adder(bit_len = 8, debug = False)
    __test__carry_skip_adder(bit_len = 8, debug = False)
    __test__carry_save_adder(bit_len = 8, debug = True)