
from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.Multiplier import eFA

from random import random
import matplotlib.pyplot as plt
from datetime import datetime

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body


log = Log(f"{__file__}.log", terminal=True)

ALPHA_A = 0.25
ALPHA_B = 0.75
ALPHA_C = 0
SAMPLE = 1_000_000

input_alpha = [0, 0, 0]
output_alpha = [0, 0]
pmos_alpha = [0, 0, 0, 0, 0, 0]

def get_input():
    r = random()
    a = 0 if r < ALPHA_A else 1

    r = random()
    b = 0 if r < ALPHA_B else 1

    r = random()
    c = 0 if r < ALPHA_C else 1

    return (a, b, c)



for sample in range(SAMPLE):

    fa = eFA()
    fa.A, fa.B, fa.C = get_input()
    
    output_alpha[0] += not fa.sum
    output_alpha[1] += not fa.carry

    input_alpha[0] += not fa.A
    input_alpha[1] += not fa.B
    input_alpha[2] += not fa.C

    for i, p in enumerate(fa.p):
        pmos_alpha[i] += not p


output_alpha[0] /= SAMPLE
output_alpha[1] /= SAMPLE

input_alpha[0] /= SAMPLE
input_alpha[1] /= SAMPLE
input_alpha[2] /= SAMPLE

pmos_alpha[0] /= SAMPLE
pmos_alpha[1] /= SAMPLE
pmos_alpha[2] /= SAMPLE
pmos_alpha[3] /= SAMPLE
pmos_alpha[4] /= SAMPLE
pmos_alpha[5] /= SAMPLE


log.println(f"output alpha: {output_alpha}")
log.println(f"input alpha: {input_alpha}")
log.println(f"pmos alpha: {pmos_alpha}")




def get_FA_delay(fa_alpha, temp, sec):
    tg1_alpha = max(fa_alpha[0], fa_alpha[1])
    tg1_vth = abs(BTI.Vth) + BTI.delta_vth(BTI.Vdef, temp, tg1_alpha, BTI.Tclk, sec)
    tg1_pb = pmos_vth_to_body(tg1_vth)

    tg2_alpha = max(fa_alpha[2], fa_alpha[3])
    tg2_vth = abs(BTI.Vth) + BTI.delta_vth(BTI.Vdef, temp, tg2_alpha, BTI.Tclk, sec)
    tg2_pb = pmos_vth_to_body(tg2_vth)

    return tgate_pb_to_delay(tg1_pb) + tgate_pb_to_delay(tg2_pb)


if False:
    temp = 273.15 + 30
    _zero_delay = get_FA_delay(pmos_alpha, temp, 0)
    
    res_week = []
    res_delay = []

    for week in range(0, 200):
        delay = get_FA_delay(pmos_alpha, temp, week * 7 *24*60*60)
        aging_rate = (delay - _zero_delay) / _zero_delay

        log.println(f"week {week:03}: {delay: 8.2f} [{aging_rate * 100 :4.2f}%]")
        res_week.append(week)
        res_delay.append(delay)

    plt.plot(res_week, res_delay)
    plt.title(f"singleFA-TEMP-{temp}")

    timestamp = datetime.now().strftime('%m,%d-%H:%M:%S.%f')
    fig_name = f"fig-{timestamp}.jpg"
    plt.legend()
    plt.savefig(fig_name)
    plt.clf()
    log.println(f"plot saved in {fig_name}")