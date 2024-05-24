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


# example of using
if __name__ == "__main__":
    

    print(
        delta_vth(Vdef, T, 160/256, Tclk, t)
    )
