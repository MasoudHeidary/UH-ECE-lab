import tool.NBTI_formula as NBTI
import tool.vth_body_map as VTH
from tool.map_pb_to_current import get_current_from_pb, get_pb_from_current


def generate_body_voltage(alpha_lst, t_sec, x_len, y_len, bit_len, faulty_transistor):
    # equation starts from 1s
    if t_sec <= 0:
        t_sec = 10

    # the body_voltage pre define structure is from inside to outside
    # body_voltage = [[[0 for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len - 1)]
    # body_voltage = []
    body_voltage = [
        [
            [
                [[0 for _ in range(6)] for _ in range(bit_len)]
                for _ in range(bit_len-1)
            ]
            for _ in range(y_len)
        ]
        for _ in range(x_len)
    ]

    for pe_x in range(x_len):
        for pe_y in range(y_len):

            for fa_i in range(bit_len - 1):
                for fa_j in range(bit_len):

                    for t_index in range(6):

                        _vth_base = abs(NBTI.Vth)
                        _vth_growth = NBTI.delta_vth(
                            NBTI.Vdef,
                            NBTI.T,
                            alpha_lst[pe_x][pe_y][fa_i][fa_j][t_index],
                            NBTI.Tclk,
                            t_sec,
                        )

                        if faulty_transistor:
                            if (
                                (pe_x == faulty_transistor["pe_x"])
                                and (pe_y == faulty_transistor["pe_y"])
                                and (fa_i == faulty_transistor["fa_i"])
                                and (fa_j == faulty_transistor["fa_j"])
                                and (t_index == faulty_transistor["t_index"])
                            ):
                                _vth_base *= faulty_transistor["x_vth_base"]
                                _vth_growth *= faulty_transistor["x_vth_growth"]

                        _vth = _vth_base + _vth_growth
                        _vb = VTH.get_body_voltage(_vth)

                        body_voltage[pe_x][pe_y][fa_i][fa_j][t_index] = _vb

    return body_voltage


def get_life_pe(alpha_lst, x_len, y_len, bit_len, faulty_transistor=False):
    initial_v_base = VTH.get_body_voltage(vth=abs(NBTI.Vth))
    current_base = get_current_from_pb(initial_v_base)
    current_fail = current_base * (1 - 0.5)
    vb_fail = get_pb_from_current(current_fail)

    for t_week in range(0, 200, 1):
        t_sec = t_week * 7 * 24 * 60 * 60

        body_voltage = generate_body_voltage(
            alpha_lst, t_sec, x_len, y_len, bit_len, faulty_transistor
        )

        for pe_x in range(x_len):
            for pe_y in range(y_len):
                for fa_i in range(bit_len - 1):
                    for fa_j in range(bit_len):
                        for t_index in range(6):

                            if body_voltage[pe_x][pe_y][fa_i][fa_j][t_index] >= vb_fail:
                                # log.println("alpha: [NORMAL]")
                                # log.println(f"FA[{fa_i}][{fa_j}], transistor[{t_index}] failed at week [{t_week}], {normal_body_voltage[fa_i][fa_j][t_index]} >= {vb_fail}")
                                return {
                                    "pe_x": pe_x,
                                    "pe_y": pe_y,
                                    "fa_i": fa_i,
                                    "fa_j": fa_j,
                                    "t_index": t_index,
                                    "t_week": t_week,
                                }
