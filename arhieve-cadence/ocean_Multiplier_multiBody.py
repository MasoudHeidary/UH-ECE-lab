import subprocess

from tool.log import *

def generate_netlist(pb:list, vdd:float):
    _netlist = \
    f"""

// Library name: prjage
// Cell name: andnb
// View name: schematic
subckt andnb A B GND O PBody VDD
    R4 (PBody pbody) resistor r=0
    R53 (out O) resistor r=0
    R2 (GND gnd) resistor r=0
    R0 (VDD vdd) resistor r=0
    N2 (net10 A net16 gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N0 (net16 B gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N1 (out net10 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P1 (net10 A vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P0 (net10 B vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (out net10 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
ends andnb
// End of subcircuit definition.

// Library name: prjage
// Cell name: 8andnb
// View name: schematic
subckt prjage_8andnb_schematic A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\>\\
        A\<6\> A\<7\> B\<0\> B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\>\\
        B\<7\> GND PBody VDD y\<0\> y\<1\> y\<2\> y\<3\> y\<4\> y\<5\>\\
        y\<6\> y\<7\>
    R4 (PBody pbody) resistor r=0
    R2 (GND gnd) resistor r=0
    R0 (VDD vdd) resistor r=0
    I2 (A\<1\> B\<1\> gnd y\<1\> pbody vdd) andnb
    I4 (A\<3\> B\<3\> gnd y\<3\> pbody vdd) andnb
    I3 (A\<2\> B\<2\> gnd y\<2\> pbody vdd) andnb
    I6 (A\<5\> B\<5\> gnd y\<5\> pbody vdd) andnb
    I5 (A\<4\> B\<4\> gnd y\<4\> pbody vdd) andnb
    I1 (A\<0\> B\<0\> gnd y\<0\> pbody vdd) andnb
    I7 (A\<6\> B\<6\> gnd y\<6\> pbody vdd) andnb
    I8 (A\<7\> B\<7\> gnd y\<7\> pbody vdd) andnb
ends prjage_8andnb_schematic
// End of subcircuit definition.

// Library name: prjage
// Cell name: FAnb2
// View name: schematic
subckt FAnb2 A B C carry gnd pb0 pb1 sum vdd
    N1 (nA A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n\\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0\\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N2 (A nB xx gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n\\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0\\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N3 (nA B xx gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n\\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0\\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N32 (net16 dxx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N31 (net20 nxx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N39 (C dxx carry gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N38 (net6 nxx carry gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N33 (net20 net18 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N34 (net18 net9 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N36 (net16 net8 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N26 (nB B gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n\\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0\\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N27 (nC C gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n\\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0\\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N28 (nxx xx gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N29 (net11 xx gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N30 (dxx net11 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N35 (net9 nC gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N37 (net8 nC gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N40 (net6 net4 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N41 (net4 net1 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N42 (net2 A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    N43 (net1 net2 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (nA A vdd pb1) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P26 (nB B vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P1 (A B xx pb0) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n\\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0\\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P4 (nA nB xx pb0) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n\\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0\\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P27 (nC C vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P28 (nxx xx vdd net3) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P39 (C nxx carry pb1) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P38 (net6 dxx carry pb0) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P32 (net16 nxx sum pb1) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P31 (net20 dxx sum pb0) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n\\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P33 (net20 net18 vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f\\
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0\\
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0\\
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P34 (net18 net9 vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P36 (net16 net8 vdd pb1) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P29 (net11 xx vdd pb1) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P30 (dxx net11 vdd pb1) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P35 (net9 nC vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P37 (net8 nC vdd pb1) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P40 (net6 net4 vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P41 (net4 net1 vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P42 (net2 A vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    P43 (net1 net2 vdd pb0) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n\\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1\\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n\\
        tie_orient=0 swrfmhc_local=0 analog=0
    C0 (sum gnd) capacitor c=5f
    C1 (carry gnd) capacitor c=5f
ends FAnb2
// End of subcircuit definition.

// Library name: prjage
// Cell name: 8ADnb2
// View name: schematic
subckt prjage_8ADnb2_schematic A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\>\\
        A\<6\> A\<7\> B\<0\> B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\>\\
        B\<7\> S\<0\> S\<1\> S\<2\> S\<3\> S\<4\> S\<5\> S\<6\> S\<7\>\\
        carry cin gnd pb0 pb1 vdd
    I25 (A\<7\> B\<7\> _c\<6\> carry gnd pb1 pb0 S\<7\> vdd) FAnb2
    I24 (A\<6\> B\<6\> _c\<5\> _c\<6\> gnd pb1 pb0 S\<6\> vdd) FAnb2
    I23 (A\<5\> B\<5\> _c\<4\> _c\<5\> gnd pb1 pb0 S\<5\> vdd) FAnb2
    I22 (A\<4\> B\<4\> _c\<3\> _c\<4\> gnd pb1 pb0 S\<4\> vdd) FAnb2
    I21 (A\<3\> B\<3\> _c\<2\> _c\<3\> gnd pb1 pb0 S\<3\> vdd) FAnb2
    I20 (A\<2\> B\<2\> _c\<1\> _c\<2\> gnd pb1 pb0 S\<2\> vdd) FAnb2
    I19 (A\<1\> B\<1\> _c\<0\> _c\<1\> gnd pb1 pb0 S\<1\> vdd) FAnb2
    I18 (A\<0\> B\<0\> cin _c\<0\> gnd pb1 pb0 S\<0\> vdd) FAnb2
ends prjage_8ADnb2_schematic
// End of subcircuit definition.

// Library name: prjage
// Cell name: 8MPnb2
// View name: schematic
subckt prjage_8MPnb2_schematic A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\>\\
        A\<6\> A\<7\> B\<0\> B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\>\\
        B\<7\> GND M\<0\> M\<10\> M\<11\> M\<12\> M\<13\> M\<14\> M\<15\>\\
        M\<1\> M\<2\> M\<3\> M\<4\> M\<5\> M\<6\> M\<7\> M\<8\> M\<9\> VDD\\
        pb\<0\> pb\<10\> pb\<11\> pb\<12\> pb\<13\> pb\<1\> pb\<2\>\\
        pb\<3\> pb\<4\> pb\<5\> pb\<6\> pb\<7\> pb\<8\> pb\<9\>
    R0 (VDD vdd) resistor r=0
    R1 (gnd GND) resistor r=0
    R124 (gnd zero) resistor r=0
    I151 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<1\>\\
        B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> gnd pb\<13\> vdd\\
        net16 net5 net7 net76 net3 net4 net8 net1) prjage_8andnb_schematic
    I163 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<7\>\\
        B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> gnd pb\<13\> vdd\\
        net292 net291 net290 net289 net288 net287 net286 net285)\\
        prjage_8andnb_schematic
    I161 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<6\>\\
        B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> gnd pb\<13\> vdd\\
        net318 net317 net316 net315 net314 net313 net312 net311)\\
        prjage_8andnb_schematic
    I159 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<5\>\\
        B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> gnd pb\<13\> vdd\\
        net344 net343 net342 net341 net340 net339 net338 net337)\\
        prjage_8andnb_schematic
    I157 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<4\>\\
        B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> gnd pb\<13\> vdd\\
        net454 net453 net452 net451 net450 net449 net448 net447)\\
        prjage_8andnb_schematic
    I155 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<3\>\\
        B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> gnd pb\<13\> vdd\\
        net428 net427 net426 net425 net424 net423 net422 net421)\\
        prjage_8andnb_schematic
    I154 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<2\>\\
        B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> gnd pb\<13\> vdd\\
        net420 net419 net418 net417 net416 net415 net414 net413)\\
        prjage_8andnb_schematic
    I150 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\>\\
        B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> gnd pb\<13\> vdd\\
        M\<0\> net10 net12 net14 net15 net2 net11 net13)\\
        prjage_8andnb_schematic
    I176 (net10 net12 net14 net15 net2 net11 net13 zero net16 net5 net7\\
        net76 net3 net4 net8 net1 M\<1\> net24 net23 net22 net21 net20\\
        net19 net18 net17 zero gnd pb\<1\> pb\<0\> vdd)\\
        prjage_8ADnb2_schematic
    I180 (net352 net351 net350 net349 net348 net347 net346 net345 net344\\
        net343 net342 net341 net340 net339 net338 net337 M\<5\> net326\\
        net325 net324 net323 net322 net321 net320 net319 zero gnd pb\<9\>\\
        pb\<8\> vdd) prjage_8ADnb2_schematic
    I181 (net326 net325 net324 net323 net322 net321 net320 net319 net318\\
        net317 net316 net315 net314 net313 net312 net311 M\<6\> net300\\
        net299 net298 net297 net296 net295 net294 net293 zero gnd pb\<11\>\\
        pb\<10\> vdd) prjage_8ADnb2_schematic
    I182 (net300 net299 net298 net297 net296 net295 net294 net293 net292\\
        net291 net290 net289 net288 net287 net286 net285 M\<7\> M\<8\>\\
        M\<9\> M\<10\> M\<11\> M\<12\> M\<13\> M\<14\> M\<15\> zero gnd\\
        pb\<13\> pb\<12\> vdd) prjage_8ADnb2_schematic
    I179 (net479 net478 net477 net476 net475 net474 net473 net472 net454\\
        net453 net452 net451 net450 net449 net448 net447 M\<4\> net352\\
        net351 net350 net349 net348 net347 net346 net345 zero gnd pb\<7\>\\
        pb\<6\> vdd) prjage_8ADnb2_schematic
    I178 (net445 net444 net443 net442 net441 net440 net439 net438 net428\\
        net427 net426 net425 net424 net423 net422 net421 M\<3\> net479\\
        net478 net477 net476 net475 net474 net473 net472 zero gnd pb\<5\>\\
        pb\<4\> vdd) prjage_8ADnb2_schematic
    I177 (net24 net23 net22 net21 net20 net19 net18 net17 net420 net419\\
        net418 net417 net416 net415 net414 net413 M\<2\> net445 net444\\
        net443 net442 net441 net440 net439 net438 zero gnd pb\<3\> pb\<2\>\\
        vdd) prjage_8ADnb2_schematic
ends prjage_8MPnb2_schematic
// End of subcircuit definition.

// Library name: prjage
// Cell name: test_MPnb2
// View name: schematic
I0 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\> B\<1\>\\
        B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> B\<7\> gnd net3 net41 net40\\
        net39 net38 net37 _net0 net2 net1 net48 net47 net46 net45 net44\\
        net43 net42 vdd pb\<0\> pb\<5\> pb\<5\> pb\<6\> pb\<6\> pb\<0\>\\
        pb\<1\> pb\<1\> pb\<2\> pb\<2\> pb\<3\> pb\<3\> pb\<4\> pb\<4\>)\\
        prjage_8MPnb2_schematic
V3 (step gnd) vsource type=pulse val0=0 val1={vdd} period=16n delay=1p\\
        rise=1p fall=1p
V17 (pb\<6\> gnd) vsource dc={pb[6]} type=dc
V8 (pb\<5\> gnd) vsource dc={pb[5]} type=dc
V7 (pb\<4\> gnd) vsource dc={pb[4]} type=dc
V6 (pb\<3\> gnd) vsource dc={pb[3]} type=dc
V5 (pb\<2\> gnd) vsource dc={pb[2]} type=dc
V4 (pb\<1\> gnd) vsource dc={pb[1]} type=dc
V2 (pb\<0\> gnd) vsource dc={pb[0]} type=dc
V1 (gnd 0) vsource dc=0 type=dc
V0 (vdd 0) vsource dc={vdd} type=dc
C15 (net3 gnd) capacitor c=1f
C14 (net1 gnd) capacitor c=1f
C13 (net47 gnd) capacitor c=1f
C12 (net45 gnd) capacitor c=1f
C11 (net43 gnd) capacitor c=1f
C10 (net41 gnd) capacitor c=1f
C9 (net39 gnd) capacitor c=1f
C8 (net37 gnd) capacitor c=1f
C7 (net2 gnd) capacitor c=1f
C6 (net48 gnd) capacitor c=1f
C5 (net46 gnd) capacitor c=1f
C4 (net44 gnd) capacitor c=1f
C3 (net42 gnd) capacitor c=1f
C2 (net40 gnd) capacitor c=1f
C1 (net38 gnd) capacitor c=1f
C0 (_net0 gnd) capacitor c=1f
R15 (step B\<7\>) resistor r=0
R14 (step B\<6\>) resistor r=0
R13 (step B\<5\>) resistor r=0
R12 (step B\<4\>) resistor r=0
R11 (step B\<3\>) resistor r=0
R10 (step B\<2\>) resistor r=0
R9 (step B\<1\>) resistor r=0
R8 (step B\<0\>) resistor r=0
R7 (step A\<7\>) resistor r=0
R6 (step A\<6\>) resistor r=0
R5 (step A\<5\>) resistor r=0
R4 (step A\<4\>) resistor r=0
R3 (step A\<3\>) resistor r=0
R2 (step A\<2\>) resistor r=0
R1 (step A\<1\>) resistor r=0
R0 (step A\<0\>) resistor r=0
    """

    return _netlist
def update_netlist_file(file_name, code):
    _file = open(file_name, "w")
    _file.write(code)
    _file.flush()
    _file.close()



def generate_ocean_script(log_file, pb, vdd):
    _ocean_code = \
    f"""
simulator( 'spectre )
design(	 "/home/mheidary/simulation/test_MPnb2/spectre/schematic/netlist/netlist")
resultsDir( "/home/mheidary/simulation/test_MPnb2/spectre/schematic" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "9n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
save( 'i "/V0/PLUS" )
temp( 27 ) 
run()
selectResult( 'tran )

/*
plot(getData("/end") getData("/vdd") getData("/V0/PLUS") getData("/step") )
hardCopyOptions(?hcOutputFile "{log_file}.png")
hardCopy()
*/

reportFile = outfile("{log_file}.txt")
_delay = delay(?wf1 VT("/step"), ?value1 {vdd/2}, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/end"), ?value2 {vdd/2}, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
_power = average(abs((VT("/vdd") * IT("/V0/PLUS"))))
fprintf(reportFile "{{\\\"vdd\\\":%f, \\\"pb0\\\":%f, \\\"pb1\\\":%f, \\\"pb2\\\":%f, \\\"pb3\\\":%f, \\\"pb4\\\":%f, \\\"pb5\\\":%f, \\\"pb6\\\":%f, \\\"delay\\\":%f, \\\"power\\\":%f }}" {vdd} {pb[0]} {pb[1]} {pb[2]} {pb[3]} {pb[4]} {pb[5]} {pb[6]} _delay*1e9 _power*1e6)
close(reportFile)

exit
    """
    return _ocean_code

    
def update_ocean_script_file(file_address, ocean_script):
    _file = open(file_address, "w")
    _file.write(ocean_script)
    _file.flush()
    _file.close()


def run_ocean_script(script_address):
    subprocess.call(["ocean", "-replay", script_address])



log = Log("terminal-log.txt")
counter = 0
counter_t = 0


# for vdd in [i/100 for i in range(60, 90+1, 10)]:
#     vb_base = int(vdd*100)
#     # vb_base = 200
#     vb_max = 250+1
#     for pb0 in [i/100 for i in range(vb_base, vb_max, 50)]:
#         for pb1 in [i/100 for i in range(vb_base, vb_max, 50)]:
#             for pb2 in [i/100 for i in range(vb_base, vb_max, 50)]:
#                 for pb3 in [i/100 for i in range(vb_base, vb_max, 50)]:
#                     for pb4 in [i/100 for i in range(vb_base, vb_max, 50)]:
#                         for pb5 in [i/100 for i in range(vb_base, vb_max, 50)]:
#                             for pb6 in [i/100 for i in range(vb_base, vb_max, 50)]:

# for vdd in [i/100 for i in range(80, 90+5, 10)]:
#     # vb_base = int(vdd*100)
#     vb_base = 200
#     vb_max = 350+1
#     for pb0 in [i/100 for i in range(vb_base, vb_max, 50)]:
#         for pb1 in [i/100 for i in range(vb_base, vb_max, 50)]:
#             for pb2 in [i/100 for i in range(vb_base, vb_max, 50)]:
#                 for pb3 in [i/100 for i in range(vb_base, vb_max, 50)]:
#                     for pb4 in [i/100 for i in range(vb_base, vb_max, 50)]:
#                         for pb5 in [i/100 for i in range(vb_base, vb_max, 50)]:
#                             for pb6 in [i/100 for i in range(vb_base, vb_max, 50)]:

                                # if (8592 == counter):
                                #     pb = [pb0, pb1, pb2, pb3, pb4, pb5, pb6]

                                #     # short circuit pb6 to pb5
                                #     pb[-1] = pb[-2]
                                #     if counter % 4 != 0:
                                #         log.println(f"counter #{counter} (!SKIPPED), data: {pb} vdd({vdd})")
                                #     else:
                                #         netlist = generate_netlist(pb, vdd)
                                #         update_netlist_file("/home/mheidary/simulation/test_MPnb2/spectre/schematic/netlist/netlist", netlist)

                                #         script = generate_ocean_script(f"./log/{counter}", pb, vdd)
                                #         update_ocean_script_file("./multibody.ocn", script)
                                #         run_ocean_script("./multibody.ocn")
                                        
                                #         log.println(f"counter #{counter}, data: {pb} vdd({vdd})")

                                # counter += 1
                                # print(counter)





for vdd in [i/100 for i in range(60, 90+1, 1)]:
    vb_base = int(vdd*100)
    vb_max = 390+1
    for pb0 in [i/100 for i in range(vb_base, vb_max, 10)]:


        if (0 <= counter):
            # pb = [pb0, pb1, pb2, pb3, pb4, pb5, pb6]
            pb = [pb0, pb0, pb0, pb0, pb0, pb0, pb0]

            
            netlist = generate_netlist(pb, vdd)
            update_netlist_file("/home/mheidary/simulation/test_MPnb2/spectre/schematic/netlist/netlist", netlist)

            _log_name = f"vdd-{vdd}-pb-{pb[0]}"
            script = generate_ocean_script(f"./log-multi-body-range-0.6-0.9-step-0.01/{_log_name}", pb, vdd)
            update_ocean_script_file("./multibody.ocn", script)
            run_ocean_script("./multibody.ocn")
            
            log.println(f"counter #{counter}, data: {pb} vdd({vdd})")

        counter += 1
        print(counter)