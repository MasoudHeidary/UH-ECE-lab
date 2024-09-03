# UH-ECE-lab

University of Houston Property.




## interesting bug

**PMOS under constant stress:**
this happen in this section of code

```python
v0 = VTH.get_body_voltage(abs(NBTI.Vth)*1.1 + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
```

reason will be the 
```python
def Bt(T, alpha, Tclk, t):
    _numerator = 2 * xi1 * tox + sqrt(xi2 * C(T) * (1-alpha) * Tclk)
    _denominator = 2 * tox + sqrt(C(T) * t)

    return 1 - (_numerator / _denominator)
```
section of code and will cause `(1-alpha)` to be zero