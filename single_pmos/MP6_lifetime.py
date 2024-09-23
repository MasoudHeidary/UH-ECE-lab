import tool.NBTI_formula as NBTI 
import tool.vth_body_map as VTH
from map_pb_to_current import get_current_from_pb, get_pb_from_current


# normal alpha
normal_alpha = \
    [
        [
            [0.75, 0.625, 0.625], [0.75, 0.625, 0.625], [0.75, 0.625, 0.625], 
            [0.75, 0.625, 0.625], [0.75, 0.625, 0.625], [0.5, 0.75, 0.75],  # [1,...] => [0.5, ...]
        ], 
        [
            [0.625, 0.5625, 0.5625], [0.625, 0.5625, 0.5625], [0.625, 0.5625, 0.5625],
            [0.625, 0.5625, 0.5625], [0.79296875, 0.64453125, 0.64453125], [0.91796875, 0.75, 0.75],
        ],
        [
            [0.5625, 0.53125, 0.53125], [0.5625, 0.53125, 0.53125], [0.5625, 0.53125, 0.53125], 
            [0.66796875, 0.58203125, 0.58203125], [0.751953125, 0.646484375, 0.646484375], [0.8828125, 0.75, 0.75],
        ],
        [
            [0.53125, 0.515625, 0.515625], [0.53125, 0.515625, 0.515625], [0.59375, 0.541015625, 0.541015625],
            [0.646484375, 0.5810546875, 0.5810546875], [0.736328125, 0.646484375, 0.646484375], [0.8662109375, 0.75, 0.75],
        ],
        [
            [0.515625, 0.5078125, 0.5078125], [0.552734375, 0.5224609375, 0.5224609375], [0.5849609375, 0.54248046875, 0.54248046875],
            [0.6416015625, 0.5830078125, 0.5830078125], [0.7275390625, 0.6474609375, 0.6474609375], [0.85791015625, 0.75, 0.75],
        ],
    ]


# modifing XOR FA[4][5] T1
improved_alpha = \
[
    [
        [0.75, 0.625, 0.625], [0.75, 0.625, 0.625], [0.75, 0.625, 0.625],
        [0.75, 0.625, 0.625], [0.625, 0.5625, 0.5625], [0.5, 0.625, 0.625],
    ],
    [
        [0.625, 0.5625, 0.5625], [0.625, 0.5625, 0.5625], [0.625, 0.5625, 0.5625], 
        [0.54296875, 0.5234375, 0.5234375], [0.728515625, 0.591796875, 0.591796875], [0.876953125, 0.625, 0.625], 
    ],
    [
        [0.5625, 0.53125, 0.53125], [0.5625, 0.53125, 0.53125], [0.51171875, 0.5078125, 0.5078125], 
        [0.625, 0.55078125, 0.55078125], [0.6845703125, 0.5947265625, 0.5947265625], [0.82421875, 0.625, 0.625]
    ],
    [
        [0.53125, 0.515625, 0.515625], [0.5, 0.505859375, 0.505859375], [0.56640625, 0.521484375, 0.521484375], 
        [0.6015625, 0.54931640625, 0.54931640625], [0.6689453125, 0.5947265625, 0.5947265625], [0.79931640625, 0.625, 0.625]
    ],
    [
        [0.5, 0.503662109375, 0.503662109375], [0.53515625, 0.516845703125, 0.516845703125], [0.556640625, 0.540283203125, 0.540283203125], 
        [0.5966796875, 0.562255859375, 0.562255859375], [0.66015625, 0.598388671875, 0.598388671875], [0.786865234375, 0.5, 0.5]
    ]
]



def generate_body_voltage(alpha_lst, t_sec):
    # equation starts from 1s
    if t_sec <= 0:
        t_sec = 100

    body_voltage = []
    
    for layer in range(5):
        lay_v = []
        for fa_index in range(6):
            if (layer==4) and (fa_index==5):
                v0 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
                v1 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
                tmp_v = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec)
                v2 = VTH.get_body_voltage(tmp_v * 1.2)
                v3 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec))
                v4 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec))
                v5 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec))
            else:
                v0 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
                v1 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
                v2 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec))
                v3 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec))
                v4 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec))
                v5 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec))
            # v0= v1= v2= v3= v4= v5= 2.5

            lay_v.append([v0, v1, v2, v3, v4, v5])
        body_voltage.append(lay_v)
    return body_voltage
    


current_base = get_current_from_pb(0.8)
current_fail = current_base * 0.3            # 30% in current will result in failure
vb_fail = get_pb_from_current(current_fail)

if __name__ == "__main__":

    if True:
        for t_week in range(0, 200, 1):
            t_sec = t_week * (30/2) * 24 * 60 * 60



            # normal aging
            body_voltage = generate_body_voltage(normal_alpha, t_sec)

            for fa_i in range(5):
                for fa_j in range(6):
                    for t_index in range(6):
                    
                        if body_voltage[i][j][t_index] >= vb_fail:
                            print(f"FA[{fa_i}][{fa_j}], transistor[{t_index}] failed, {body_voltage[fa_i][fa_j]} >= {vb_fail}")
