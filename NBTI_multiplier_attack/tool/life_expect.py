
import tool.NBTI_formula as BTI
import random


def generate_guassian_vth_base(bit_len=8, mu=0, sigma=0.02, base_vth=abs(BTI.Vth), seed=False):
    if seed:
        random.seed(seed)

    vth = [
        [[base_vth for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len-1)
    ]

    for fa_i in range(bit_len-1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                vth_variation = random.gauss(mu, sigma)
                vth[fa_i][fa_j][t_index] *= (1 + vth_variation)

    return vth


def generate_body_from_base(bit_len, vth_matrix, alpha_lst, t_sec, tmp, pmos_map = None):
    body_voltage = [
        [[0 for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len-1)
    ]

    for fa_i in range(bit_len-1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                vth_growth = BTI.delta_vth(
                    BTI.Vdef,
                    tmp,
                    alpha_lst[fa_i][fa_j][t_index],
                    BTI.Tclk,
                    t_sec
                )

                body_voltage[fa_i][fa_j][t_index] = pmos_map(
                    vth_matrix[fa_i][fa_j][t_index] + vth_growth
                )

    return body_voltage