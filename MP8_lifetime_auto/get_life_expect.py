import tool.NBTI_formula as NBTI
import tool.vth_body_map as VTH
from tool.map_pb_to_current import get_current_from_pb, get_pb_from_current
import random

def generate_body_voltage(alpha_lst, t_sec, bit_len, faulty_transistor):
    # equation starts from 1s
    if t_sec <= 0:
        t_sec = 10

    body_voltage = []

    for layer in range(bit_len - 1):
        lay_v = []
        for fa_index in range(bit_len):
            tmp_v_T = []
            for t_index in range(6):
                _vth_base = abs(NBTI.Vth)
                _vth_growth = NBTI.delta_vth(
                    NBTI.Vdef,
                    NBTI.T,
                    alpha_lst[layer][fa_index][t_index],
                    NBTI.Tclk,
                    t_sec,
                )
                if faulty_transistor:
                    if (
                        (layer == faulty_transistor["fa_i"])
                        and (fa_index == faulty_transistor["fa_j"])
                        and (t_index == faulty_transistor["t_index"])
                    ):
                        _vth_base *= faulty_transistor["x_vth_base"]
                        _vth_growth *= faulty_transistor["x_vth_growth"]

                _vth = _vth_base + _vth_growth
                _vb = VTH.get_body_voltage(_vth)
                tmp_v_T.append(_vb)

            lay_v.append(tmp_v_T)
        body_voltage.append(lay_v)
    return body_voltage


def get_life_expect(alpha_lst, bit_len, faulty_transistor=False):
    initial_v_base = VTH.get_body_voltage(vth=abs(NBTI.Vth))
    current_base = get_current_from_pb(initial_v_base)
    current_fail = current_base * (1 - 0.5)
    vb_fail = get_pb_from_current(current_fail)

    for t_week in range(0, 200, 1):
        t_sec = t_week * 7 * 24 * 60 * 60

        body_voltage = generate_body_voltage(
            alpha_lst, t_sec, bit_len, faulty_transistor
        )

        for fa_i in range(bit_len - 1):
            for fa_j in range(bit_len):
                for t_index in range(6):

                    if body_voltage[fa_i][fa_j][t_index] >= vb_fail:
                        # log.println("alpha: [NORMAL]")
                        # log.println(f"FA[{fa_i}][{fa_j}], transistor[{t_index}] failed at week [{t_week}], {normal_body_voltage[fa_i][fa_j][t_index]} >= {vb_fail}")
                        return {
                            "fa_i": fa_i,
                            "fa_j": fa_j,
                            "t_index": t_index,
                            "t_week": t_week,
                        }




def generate_random_vth_base(bit_len=8, neg_factor=-0.1, pos_factor=0.1, base_vth=abs(NBTI.Vth)):
    vth = [
        [[base_vth for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len-1)
    ]

    for fa_i in range(bit_len-1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                vth[fa_i][fa_j][t_index] *= (1 + random.uniform(neg_factor, pos_factor))

    return vth


def generate_body_voltage_from_base(alpha_lst, t_sec, bit_len, vth_matrix):
    if t_sec <= 10:
        t_sec = 10

    body_voltage = [
        [[0 for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len-1)
    ]

    for fa_i in range(bit_len-1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                vth_growth = NBTI.delta_vth(
                    NBTI.Vdef,
                    NBTI.T,
                    alpha_lst[fa_i][fa_j][t_index],
                    NBTI.Tclk,
                    t_sec
                )

                body_voltage[fa_i][fa_j][t_index] = VTH.get_body_voltage(
                    vth_matrix[fa_i][fa_j][t_index] + vth_growth
                )

    return body_voltage
