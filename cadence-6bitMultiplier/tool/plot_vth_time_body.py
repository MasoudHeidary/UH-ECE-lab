import matplotlib.pyplot as plt


from NBTI_formula import *
from vth_body_map import *


x = []
y = []

for i in range(1, 200):
    x += [i]
    t = i * 7 * 24 * 60 * 60

    y += [get_body_voltage(
        0.448 + delta_vth(Vdef, T, alpha, Tclk, t)
    )]

plt.plot(x, y)
plt.show()