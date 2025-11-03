# UH-ECE-lab

## Overview

Welcome to the UH-ECE-lab repository, This repository contains the code and resources related to my research in Electrical and Computer Engineering (ECE) at the University of Houston. It serves as a collection of experimental scripts, utilities, and prototypes used throughout various projects and studies.

The nature of research in this field often requires flexibility, rapid iteration, and frequent modifications to the code. Therefore, many of the functions, classes, and utilities are duplicated across multiple projects. Instead of abstracting these functions into reusable modules, they are often **copied and pasted** directly into different scripts. This approach ensures that code remains adaptable and can be quickly adjusted without breaking dependencies between projects.

## Key Points

- Repetitive Code: Many functions and classes are repeated or copied between files rather than being referenced via imports. This is a deliberate decision due to the constantly evolving nature of the research, where frequent changes can render dependencies fragile.

- Flexibility Over Reusability: The focus is on rapid changes and adaptability. Modularization has been deprioritized in favor of ensuring that each piece of code can be modified independently to fit the unique needs of each specific research project.

- Code Evolution: As this repository serves as an ongoing resource for various projects, you will find that some pieces of code may evolve differently across projects. This is by design to accommodate the constantly changing requirements of the research.

- Preventing Breakages: One of the main goals of this structure is to prevent code from breaking when moving between projects or testing new ideas. By copying code and adjusting it for each scenario, there is less risk of an issue arising due to dependencies or module changes.




### Bug 

#### PMOS under constant stress (NBTI aging):

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