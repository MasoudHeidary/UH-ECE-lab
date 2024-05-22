from math import exp, sqrt


# user input
t = 1 * 365 * 24 * 60 * 60
T = 300
duty_cycle = 0.5
Tclk = 1 / (4E9) # 4GHz


# physical constants
ev = 1.6021E-19
epsilon0 = 8.85E-12




# transistor parameter
tox = 4E-9
epsilonox = 3.9 * epsilon0
Cox = epsilonox / tox
Vdef = 0.8 - 0.4




# C
T0 = 1E-8
Ea = 0.49 * ev
k = 8.617E-5 * ev


# Kv
Eox =  Vdef / tox
E0 = 0.335E9
K = 8E22


#Bt
te = tox
epsilon1 = 0.9
epsilon2 = 0.5


# delta Vth
n = 1/4






def C(T: float):
   return 1/T0 * exp(-Ea/(k*T))


def Kv(T: float):
   return (ev * tox / epsilonox)**3 * K**2 * Cox * Vdef * C(T)**0.5 * exp(2*Eox/E0)


def  Bt(T: float, t: int):
   _res = (2 * epsilon1 * te + sqrt(epsilon2 * C(T) * (1-duty_cycle) * Tclk)) / (2 * tox + sqrt(C(T) * t))
   return 1 - _res


def deltaVth(T, t):
   _res = sqrt(Kv(T)**2 * duty_cycle * Tclk) / (1 - Bt(T, t)**(1/(2*n)))
   return _res ** (2*n)


print(deltaVth(T, t))

