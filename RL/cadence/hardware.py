
from map_pmos_vbody_vth import vdd_vth_to_vbody
from map_array_MP8_vbody_delay_power import vdd_vbody_to_delay_power
from NBTI_formula import delta_vth

class Hardware:
    def __init__(self, init_vth=0.442, aging_alpha=0.1):
        self.initial_vth = init_vth
        self.vth = init_vth
        self.alpha = aging_alpha

        self.vdd_levels = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    def __str__(self):
        return f"vth({self.vth:.3f}) max_freq({self.get_max_freq():8.1f}MHz)"
    
    @staticmethod
    def to_freqMHz(delay_ps):
        return 1_000_000 / delay_ps
    
    @staticmethod
    def to_delayps(freq_MHz):
        return 1_000_000 / freq_MHz

    def get_vbody(self, vdd):
        """v_body, based on vdd and internal states"""
        return vdd_vth_to_vbody(vdd, self.vth)

    def get_delay_power(self, vdd):
        """delay[ps], power[uw]"""
        vbody = self.get_vbody(vdd)
        return vdd_vbody_to_delay_power(vdd, vbody)
    
    def get_vdd(self, freq_MHz, look_up_error=True):
        delay_ps = self.to_delayps(freq_MHz)
        for vdd in self.vdd_levels:
            delay_power = self.get_delay_power(vdd)
            if delay_power and (delay_power[0] < delay_ps):
                return vdd
        if look_up_error:
            raise LookupError("too high frequency")
        return False
    
    def apply_aging(self, vdd, t0, t1, T=348.15):
        vdef = vdd + self.initial_vth

        t0_vth = delta_vth(vdef, T, self.alpha, 1E-9, t0)
        t1_vth = delta_vth(vdef, T, self.alpha, 1E-9, t1)
        self.vth += (t1_vth - t0_vth)
        print(f"{t0_vth} -> {t1_vth} == {t1_vth - t0_vth}")

    def get_max_freq(self):
        """maximum frequency possible"""
        delay_ps = self.get_delay_power(self.vdd_levels[-1])[0]
        freq_MHz = self.to_freqMHz(delay_ps)
        return freq_MHz


def hardware_debug(hardware, t, freq, vdd, delay_power):
    return f"[t:{t:10}s] [freq:{freq:5}MHz]: {hardware} vdd({vdd:.2f}) delay({delay_power[0]:5}ps) power({delay_power[1]:5}uw)"


if __name__ == "__main__":



    ### constant 1000MHz running
    freq = 1000 #MHz
    hardware = Hardware()

    t_0m = 0
    vdd = hardware.get_vdd(freq)
    delay_power = hardware.get_delay_power(vdd)
    print(hardware_debug(hardware, t_0m, freq, vdd, delay_power))

    t_12m = 12 * 30 * 24 * 60 * 60  # 12 months
    hardware.apply_aging(vdd, t_0m, t_12m)
    vdd = hardware.get_vdd(freq)
    delay_power = hardware.get_delay_power(vdd)
    print(hardware_debug(hardware, t_12m, freq, vdd, delay_power))


    print("="*30)

    ### 100MHz -> 1000MHz running
    freq = 100 #MHz
    hardware = Hardware()

    t_0m = 0
    vdd = hardware.get_vdd(freq)
    delay_power = hardware.get_delay_power(vdd)
    print(hardware_debug(hardware, t_0m, freq, vdd, delay_power))

    t_6m = 6 * 30 * 24 * 60 * 60  # 12 months
    hardware.apply_aging(vdd, t_0m, t_6m)
    vdd = hardware.get_vdd(freq)
    delay_power = hardware.get_delay_power(vdd)
    print(hardware_debug(hardware, t_6m, freq, vdd, delay_power))

    freq = 1000 #MHz
    t_12m = 12 * 30 * 24 * 60 * 60  # 12 months
    hardware.apply_aging(vdd, t_6m, t_12m)
    vdd = hardware.get_vdd(freq)
    delay_power = hardware.get_delay_power(vdd)
    print(hardware_debug(hardware, t_12m, freq, vdd, delay_power))
