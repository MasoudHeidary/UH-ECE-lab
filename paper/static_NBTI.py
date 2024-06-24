from math import exp, sqrt, pow


# physics
q =  1.60217663E-19
ev = q
eps0 = 8.854187817E-12
k = 8.6173303E-5 * ev


# transistor constant variables
n = 1/4
Ea = 0.49 * ev
E0 = 0.335E9
delta = 0.5
K = 8E22
xi1 = 0.9
xi2 = 0.5
T0 = 1E-8
tox = 4E-9 #oxide thickness
epsox = 3.9 * eps0 #change from 3.9 to 200
Cox = epsox / tox

Vgs = 0.8
Vth = -0.45
Vdef = Vgs - Vth

Tclk = 1/(1E9) #4GHz
alpha = 0.5 
T = 300 
t = 30 * 24 * 60 * 60


def C(T):
    return 1/T0 * exp(-Ea / (k * T))


def A(Vdef):
    Eox = Vdef / tox
    return (q*tox)/epsox * (K**2 * Cox * Vdef * exp(Eox/E0)**2)**(1/3)

def DVth(Vdef, T, t):
    return A(Vdef) * ((1+delta)*tox + sqrt(C(T)*t))**(2*n)


x = []
y_T1 = []
y_T2 = []
y_V1 = []
y_V2 = []

for i in range(10, 40+1, 1):
    t = i/10

    x += [t]
    y_T1 += [DVth(1.9-Vth, 300, t)]
    y_T2 += [DVth(1.9-Vth, 320, t)]
    y_V1 += [DVth(2.1-Vth, 300, t)]
    y_V2 += [DVth(2.1-Vth, 320, t)]


import matplotlib.pyplot as plt
plt.plot(x, y_T1, label="Vgs=1.9, T=300", linewidth=3)
plt.plot(x, y_T2, label="Vgs=1.9, T=320", linewidth=3)
plt.plot(x, y_V1, label="Vgs=2.1, T=300", linewidth=3)
plt.plot(x, y_V2, label="Vgs=2.1, T=220", linewidth=3)

plt.xlabel('time(seconds)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')

plt.ylabel('ΔVth(v)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.legend(fontsize=14)
plt.grid(True)
plt.show()
