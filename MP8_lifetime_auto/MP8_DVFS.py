import tool.NBTI_formula as NBTI
import tool.vth_body_map as VTH
from tool.map_pb_to_current import get_current_from_pb, get_pb_from_current
import random
import numpy as np


def initial_vth_base(bit_len=8, base_vth=abs(NBTI.Vth), faulty_transistor=False):
    vth = [
        [[base_vth for _ in range(6)] for _ in range(bit_len)]
        for _ in range(bit_len - 1)
    ]

    if faulty_transistor:
        fa_i = faulty_transistor["fa_i"]
        fa_j = faulty_transistor["fa_j"]
        t_index = faulty_transistor["t_index"]
        x_vth_base = faulty_transistor["x_vth_base"]
        # x_vth_growth = faulty_transistor['x_vth_growth']

        vth[fa_i][fa_j][t_index] *= x_vth_base

    return vth


def generate_body_voltage_from_base(
    alpha_lst, t_sec, bit_len, vth_matrix, faulty_transistor=False
):
    if t_sec <= 10:
        t_sec = 10

    body_voltage = [
        [[0 for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len - 1)
    ]

    for fa_i in range(bit_len - 1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                vth_growth = NBTI.delta_vth(
                    NBTI.Vdef, NBTI.T, alpha_lst[fa_i][fa_j][t_index], NBTI.Tclk, t_sec
                )

                if faulty_transistor:
                    if (
                        (fa_i == faulty_transistor["fa_i"])
                        and (fa_j == faulty_transistor["fa_j"])
                        and (t_index == faulty_transistor["t_index"])
                    ):
                        vth_growth *= faulty_transistor["x_vth_growth"]

                body_voltage[fa_i][fa_j][t_index] = VTH.get_body_voltage(
                    vth_matrix[fa_i][fa_j][t_index] + vth_growth
                )

    return body_voltage


def get_body_voltage(alpha, t_sec):
    if t_sec <= 10:
        t_sec = 10

    vth_base = abs(NBTI.Vth)
    vth_growth = NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha, NBTI.Tclk, t_sec)

    vth = vth_base + vth_growth
    return VTH.get_body_voltage(vth)


def get_max_alpha(alpha_lst):
    alpha = np.array(alpha_lst)
    alpha = sorted(alpha.flatten(), reverse=True)

    for a in alpha:
        if a != 1:
            return a


alpha_max = 0.853515


initial_v_base = VTH.get_body_voltage(vth=abs(NBTI.Vth))
current_base = get_current_from_pb(initial_v_base)
current_fail = current_base * (1 - 0.5)
vb_fail = get_pb_from_current(current_fail)

for t_week in range(200):
    t_sec = t_week * 7 * 24 * 60 * 60

    body_voltage = get_body_voltage(alpha_max, t_sec)

    if body_voltage > vb_fail:
        print(f"FAILED {t_week} weeks")
        break
