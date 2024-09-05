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
                v2 = abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec)
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
        vth_T0_improved = []
        vth_T1_improved = []
        for t_week in range(0, 200, 1):
            t_sec = t_week * (30/2) * 24 * 60 * 60

            x_time += [t_week]

            # normal_body_voltage = generate_vth_voltage(normal_alpha, t_sec)
            # improved_body_voltage = generate_vth_voltage(improved_alpha, t_sec)

            # vth_T0_normal += [abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, normal_alpha[lay][fa_index][0], NBTI.Tclk, t_sec)]
            vth_T1_normal += [abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, normal_alpha[lay][fa_index][1], NBTI.Tclk, t_sec)]

            # vth_T0_improved += [abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, improved_alpha[lay][fa_index][0], NBTI.Tclk, t_sec)]
            vth_T1_improved += [abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, improved_alpha[lay][fa_index][0], NBTI.Tclk, t_sec)]

        ax = axes[lay, fa_index]
        # ax.plot(x_time, vth_T0_normal, label = "T0 Normal")
        ax.plot(x_time, vth_T1_normal, label = "T1 Normal")
        # ax.plot(x_time, vth_T0_improved, label = "T0 improved")
        ax.plot(x_time, vth_T1_improved, label = "T1 improved")
    
plt.legend()
plt.tight_layout()
plt.show()

    
