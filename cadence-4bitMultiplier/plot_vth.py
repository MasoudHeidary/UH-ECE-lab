import matplotlib.pyplot as plt
import lib.NBTI_formula as NBTI


x = []
normal_vth = []
modified_vth = []

for t_week in range(1, 100, 1):
    x += [t_week]
    t_sec = t_week * 30 * 24 * 60 * 60

    alpha = [160, 160, 160, 192, 144, 160, 164, 192, 136, 148, 164, 192]
    normal_vth += [
        [(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, a/256, NBTI.Tclk, t_sec)) for a in alpha]
    ]
    
    # rewired aging
    alpha = [192, 208, 160, 192, 192, 232, 164, 192, 192, 196, 198, 202]
    modified_vth += [
        [(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, a/256, NBTI.Tclk, t_sec)) for a in alpha]
    ]


# 12 FAs in Multipliere
for fa in range(12):
    plt.plot(x, [i[fa] for i in normal_vth], label=f"Normal FA-{fa}", color='red', linewidth=2.5, linestyle=':')
    plt.plot(x, [i[fa] for i in modified_vth], label=f"Modified FA-{fa}", color='blue', linewidth=2.5, linestyle="-.")
plt.xlabel('time(weeks)', fontsize=16)
plt.ylabel('Vth(V)', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
