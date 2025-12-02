

from msimulator.get_alpha_adder import AdderShrinked, CarrySaveAdderShrinked
from msimulator.Adder import RippleAdder, CarrySkipAdder, CarrySaveAdder


from tool.log import Log

BIT_LEN = 8

log = Log(f"{__file__}.log", terminal=True)

if True and (__name__ == "__main__"):
    log.println(f"RippleAdder - getting alpha for bit_len [{BIT_LEN}]")
    alpha = AdderShrinked(BIT_LEN, RippleAdder).get_total_stress(log_obj=False)
    log.println(f"alpha, bit [{BIT_LEN}]: \n{alpha}")

    log.println(f"CarrySkipAdder - getting alpha for bit_len [{BIT_LEN}]")
    alpha = AdderShrinked(BIT_LEN, CarrySkipAdder).get_total_stress(log_obj=False)
    log.println(f"alpha, bit [{BIT_LEN}]: \n{alpha}")

    log.println(f"CarrySaveAdder - getting alpha for bit_len [{BIT_LEN}]")
    alpha = CarrySaveAdderShrinked(BIT_LEN, CarrySaveAdder).get_total_stress(log_obj=False)
    log.println(f"alpha, bit [{BIT_LEN}]: \n{alpha}")