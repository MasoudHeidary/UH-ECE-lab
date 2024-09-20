import tool.NBTI_formula as NBTI 
# import tool.vth_body_map as VTH
import matplotlib.pyplot as plt

from ocean_MP6_improving import normal_alpha, improved_alpha


def generate_vth_voltage(alpha_lst, t_sec):
    # equation starts from 1s
    if t_sec <= 0:
        t_sec = 1

    body_voltage = []
    
    for layer in range(5):
        lay_v = []
        for fa_index in range(6):
            if (layer==4) and (fa_index==5):
                #vth base is 10% higher
                v0 = abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec)
                v1 = abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec)
                tmp_v = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec)
                v2 = tmp_v * 1.1
                v3 = abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec)
                v4 = abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec)
                v5 = abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec)
            else:
                v0 = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec)
                v1 = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec)
                v2 = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec)
                v3 = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec)
                v4 = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec)
                v5 = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec)
            # v0= v1= v2= v3= v4= v5= 2.5

            lay_v.append([v0, v1, v2, v3, v4, v5])
        body_voltage.append(lay_v)
    return body_voltage




fig, axes = plt.subplots(5, 6, figsize=(10, 15))


for lay in range(5):
    for fa_index in range(6):

        x_time = []
        vth_T0_normal = []
        vth_T1_normal = []
        vth_T1o_normal = []
        vth_T0_improved = []
        vth_T1_improved = []
        vth_T1o_improved = []

        
        for t_week in range(0, 200, 1):
            t_sec = t_week * (30/2) * 24 * 60 * 60

            x_time += [t_week]

            # normal_body_voltage = generate_vth_voltage(normal_alpha, t_sec)
            # improved_body_voltage = generate_vth_voltage(improved_alpha, t_sec)
            _normal_vth = generate_vth_voltage(normal_alpha, t_sec)
            vth_T0_normal += [_normal_vth[lay][fa_index][0*2]]
            vth_T1_normal += [_normal_vth[lay][fa_index][1*2]]
            vth_T1o_normal += [_normal_vth[lay][fa_index][1*2 + 1]]

            _improved_vth = generate_vth_voltage(improved_alpha, t_sec)
            vth_T0_improved += [_improved_vth[lay][fa_index][0*2]]
            vth_T1_improved += [_improved_vth[lay][fa_index][1*2]]
            vth_T1o_improved += [_improved_vth[lay][fa_index][1*2 + 1]]


        # print needed data
        if (lay==4) and (fa_index==5):
            print(vth_T1_normal[0], vth_T1_normal[200-1])
            print(vth_T1_improved[0], vth_T1_improved[200-1])

            initial = vth_T1_normal[0]
            normal_age = vth_T1_normal[200-1] - initial
            improved_age = vth_T1_improved[200-1] - initial
            print(
                (normal_age-improved_age)/normal_age*100
            )

        # plot data
        ax = axes[lay, fa_index]

        # ax.plot(x_time, vth_T0_normal, label = "T0 Normal")
        ax.plot(x_time, vth_T1_normal, label = "T1 Normal")
        # ax.plot(x_time, vth_T1o_normal, label = "T1o Normal")


        # ax.plot(x_time, vth_T0_improved, label = "T0 improved")
        ax.plot(x_time, vth_T1_improved, label = "T1 improved")
        # ax.plot(x_time, vth_T1o_improved, label = "T1o improved")
    
plt.legend()
plt.tight_layout()
plt.show()

    
