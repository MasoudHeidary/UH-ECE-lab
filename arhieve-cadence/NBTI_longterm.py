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


def C(T):
    _value = 1/T0 * exp(-Ea / (k * T))
    return _value

def Kv(Vdef, T):
    _Cox = epsox / tox
    _Eox = Vdef / tox
    _value = (q * tox / epsox)**3 * K**2 * _Cox * Vdef * sqrt(C(T)) * exp(2 * _Eox / E0)

    return _value

def Bt(T, alpha, Tclk, t):
    _numerator = 2 * xi1 * tox + sqrt(xi2 * C(T) * (1-alpha) * Tclk)
    _denominator = 2 * tox + sqrt(C(T) * t)

    return 1 - (_numerator / _denominator)

def delta_vth(Vdef, T, alpha, Tclk, t):
    _Kv = Kv(Vdef, T)
    _Bt = Bt(T, alpha, Tclk, t)
    _numerator =  sqrt(_Kv**2 * alpha * Tclk)
    _denominator = 1 - _Bt**(0.5/n)
    
    return (_numerator / _denominator)**(2*n)



# enter data
Vgs = 0.8
Vth = -0.45
Vdef = Vgs - Vth

Tclk = 1/(1E9) #4GHz

# duty cycle
alpha = 0.5 

# temperature
T = 300 

# time
t = 30 * 24 * 60 * 60


import matplotlib.pyplot as plt
import numpy as np

#change of vdef
"""
x = []
y = []
for _vdef in range(-45, 45, 1):
    x += [_vdef / 100]
    y += [delta_vth(_vdef/10, T, alpha, Tclk, t)]

plt.plot(x, y)
plt.xlabel('vdef')
plt.ylabel('Vth change')
plt.show()
"""

#change of alpha
"""
x = []
y = []
for _alpha in range(1, 10, 1):
    x += [_alpha/10]
    y += [delta_vth(Vdef, T, _alpha/10, Tclk, t)]
plt.plot(x, y)
plt.xlabel("alpha")
plt.ylabel("Vth change")
plt.show()
"""

#change of frequency
"""
x = []
y = [] 

for _freq in range(100_000, 1_000_000, 10):
    x += [1/_freq]
    y += [delta_vth(Vdef, T, alpha, 1/_freq, t)]
plt.plot(x, y)
plt.xlabel("frequency")
plt.ylabel("Vth change")
plt.show()
"""


x = []
y_fa1_normal = []
y_fa1_rewired = []
y_fa3_normal = []
y_fa3_rewired = []
for _t_month in range(0, 120, 1):
    t_month = _t_month / 10
    x += [t_month]
    y_fa1_normal += [delta_vth(Vdef, T, 160/256, Tclk, t_month*30*24*60*60)]
    y_fa1_rewired += [delta_vth(Vdef, T, 208/256, Tclk, t_month*30*24*60*60)]
    y_fa3_normal += [delta_vth(Vdef, T, 192/256, Tclk, t_month*30*24*60*60)]
    y_fa3_rewired += [delta_vth(Vdef, T, 228/256, Tclk, t_month*30*24*60*60)]

plt.plot(x, y_fa1_normal, label="FA1-normal")
plt.plot(x, y_fa1_rewired, label="FA1-rewired")
plt.plot(x, y_fa3_normal, label="FA3-normal")
plt.plot(x, y_fa3_rewired, label="FA3-rewired")
plt.xlabel("time (month)")
plt.ylabel("Vth change")
plt.legend()
plt.show()



# from pandas import DataFrame
# df = DataFrame({'Vgs - Vth': x, 'delta Vth': y})
# df
# df.to_excel('data.xlsx', sheet_name='sheet1', index=False)