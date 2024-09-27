import matplotlib.pyplot as plt
import tool.NBTI_formula as NBTI



if False:
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


# MAIN
if False:
    time = []
    best_normal_vth = []
    worst_modified_vth = []

    for t_week in range(0, 100, 1):
        time += [t_week]
        t_sec = t_week * 30 * 24 * 60 * 60



        alpha = 136
        best_normal_vth += [
            (abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, t_sec))
        ]
        
        # rewired aging
        alpha = 232
        worst_modified_vth += [
            (abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, t_sec))
        ]

    
    # expand time
    time = [i*2 for i in time]

    plt.figure(figsize=(13, 10))

    plt.plot(time, worst_modified_vth, label="Tampered Vth", color='red', linewidth=5)
    plt.plot(time, best_normal_vth, label="Normal Vth", linewidth=5)

    plt.xlabel('Time(weeks)', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28, fontweight='bold')

    plt.ylabel('Vth(v)', fontsize=28, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')

    plt.legend(fontsize=28)
    plt.grid(True)
    plt.show()



# 4 years
t_sec = 4 * 365/7/2 *30*24*60*60
alpha = 136
nor = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, t_sec)
alpha = 232
mod = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, t_sec)
print(nor)
print(mod)
print((mod-nor)/nor*100)

# growth 
alpha = 136
start = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, 1)



# compare to initial value
t_week = 100
t_sec = t_week * (30/2) * 24 * 60 * 60
end = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 232/256, NBTI.Tclk, t_sec)
print(f"compare to initial value: {(end - abs(NBTI.Vth))/abs(NBTI.Vth)*100}")


t_list = [
    24 * 30 * 24* 60 * 60,
    49 * 30 * 24* 60 * 60,
    74 * 30 * 24* 60 * 60,
    99 * 30 * 24* 60 * 60,
]

# for t in t_list:
#     alpha = 136
#     b = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, t)
#     alpha = 232
#     c = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, t)
#     print((c-b)/b*100)

# for t in t_list:
#     alpha = 136
#     b = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, t)
#     alpha = 232
#     c = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, t)
#     print(f"{c-start}, {b-start} : {start}")
#     print((c-start-(b-start))/(b-start)*100)



# alpha = 232
# end = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, 99 * 30 * 24 * 60 * 60)
# start = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha/256, NBTI.Tclk, 1)
# print(f"{start} => {end}")
# print((end-start)/start * 100)
