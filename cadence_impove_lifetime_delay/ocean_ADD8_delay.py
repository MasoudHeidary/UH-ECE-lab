import subprocess
import random
from tool.log import Log


log = Log("tlog.txt")

NETLIST = "/home/mheidary/simulation/delay_ADD8/spectre/schematic/netlist/netlist"
SCHEMATIC = "/home/mheidary/simulation/delay_ADD8/spectre/schematic"
OCEAN = "tmp.ocn"


def update_netlist(filename, A_bin, B_bin):
    netlist = \
    f"""

// Library name: prjage
// Cell name: FAnb
// View name: schematic
subckt FAnb A B Cin Cout GND PBody S VDD
    N1 (nA A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N2 (A nB xx gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N3 (nA B xx gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N32 (net16 dxx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N31 (net20 nxx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N39 (C dxx carry gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N38 (net6 nxx carry gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N33 (net20 net18 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N34 (net18 net9 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N36 (net16 net8 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N26 (nB B gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N27 (nC C gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N28 (nxx xx gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N29 (net11 xx gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N30 (dxx net11 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N35 (net9 nC gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N37 (net8 nC gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N40 (net6 net4 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N41 (net4 net1 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N42 (net2 A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N43 (net1 net2 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (nA A vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P26 (nB B vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P1 (A B xx pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P4 (nA nB xx pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P27 (nC C vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P28 (nxx xx vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P39 (C nxx carry pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P38 (net6 dxx carry pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P32 (net16 nxx sum pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P31 (net20 dxx sum pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P33 (net20 net18 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P34 (net18 net9 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P36 (net16 net8 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P29 (net11 xx vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P30 (dxx net11 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P35 (net9 nC vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P37 (net8 nC vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P40 (net6 net4 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P41 (net4 net1 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P42 (net2 A vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P43 (net1 net2 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    R5 (Cin C) resistor r=0
    R55 (PBody pbody) resistor r=0
    R53 (sum S) resistor r=0
    R54 (carry Cout) resistor r=0
    R2 (GND gnd) resistor r=0
    R0 (VDD vdd) resistor r=0
    C0 (sum gnd) capacitor c=5f
    C1 (carry gnd) capacitor c=5f
ends FAnb
// End of subcircuit definition.

// Library name: prjage
// Cell name: 8ADnb
// View name: schematic
subckt prjage_8ADnb_schematic A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> \
        A\<6\> A\<7\> B\<0\> B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> \
        B\<7\> Cin Cout GND O\<0\> O\<1\> O\<2\> O\<3\> O\<4\> O\<5\> \
        O\<6\> O\<7\> PBody VDD
    R12 (pbody PBody) resistor r=0
    R10 (Cout C\<7\>) resistor r=0
    R9 (S\<7\> O\<7\>) resistor r=0
    R8 (S\<6\> O\<6\>) resistor r=0
    R7 (S\<5\> O\<5\>) resistor r=0
    R6 (S\<4\> O\<4\>) resistor r=0
    R5 (S\<3\> O\<3\>) resistor r=0
    R4 (S\<2\> O\<2\>) resistor r=0
    R3 (S\<1\> O\<1\>) resistor r=0
    R2 (S\<0\> O\<0\>) resistor r=0
    R0 (vdd VDD) resistor r=0
    R1 (gnd GND) resistor r=0
    R24 (B\<5\> B_5) resistor r=0
    R27 (A\<1\> A_1) resistor r=0
    R25 (B\<6\> B_6) resistor r=0
    R18 (A\<0\> A_0) resistor r=0
    R26 (B\<7\> B_7) resistor r=0
    R19 (B\<0\> B_0) resistor r=0
    R28 (A\<2\> A_2) resistor r=0
    R29 (A\<3\> A_3) resistor r=0
    R20 (B\<1\> B_1) resistor r=0
    R21 (B\<2\> B_2) resistor r=0
    R30 (A\<4\> A_4) resistor r=0
    R22 (B\<3\> B_3) resistor r=0
    R31 (A\<5\> A_5) resistor r=0
    R32 (A\<6\> A_6) resistor r=0
    R23 (B\<4\> B_4) resistor r=0
    R11 (A\<7\> A_7) resistor r=0
    R17 (Cin Cin) resistor r=0
    I8 (A_0 B_0 Cin C\<0\> gnd pbody S\<0\> vdd) FAnb
    I15 (A_7 B_7 C\<6\> C\<7\> gnd pbody S\<7\> vdd) FAnb
    I14 (A_6 B_6 C\<5\> C\<6\> gnd pbody S\<6\> vdd) FAnb
    I13 (A_5 B_5 C\<4\> C\<5\> gnd pbody S\<5\> vdd) FAnb
    I12 (A_4 B_4 C\<3\> C\<4\> gnd pbody S\<4\> vdd) FAnb
    I11 (A_3 B_3 C\<2\> C\<3\> gnd pbody S\<3\> vdd) FAnb
    I10 (A_2 B_2 C\<1\> C\<2\> gnd pbody S\<2\> vdd) FAnb
    I9 (A_1 B_1 C\<0\> C\<1\> gnd pbody S\<1\> vdd) FAnb
ends prjage_8ADnb_schematic
// End of subcircuit definition.

// Library name: prjage
// Cell name: delay_ADD8
// View name: schematic
I183 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\> \
        B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> B\<7\> Cin cout gnd net1 \
        net28 net29 net30 net31 net32 net33 o7 pbody vdd) \
        prjage_8ADnb_schematic
C9 (cout gnd) capacitor c=1f
C8 (o7 gnd) capacitor c=1f
C7 (net33 gnd) capacitor c=1f
C6 (net32 gnd) capacitor c=1f
C5 (net31 gnd) capacitor c=1f
C4 (net30 gnd) capacitor c=1f
C2 (net29 gnd) capacitor c=1f
C1 (net28 gnd) capacitor c=1f
C0 (net1 gnd) capacitor c=1f
V1 (net3 0) vsource dc=0 type=dc
V0 (net4 0) vsource dc=800.0m type=dc
R20 (gnd clk) resistor r=100
R75 (pbody vdd) resistor r=0
R17 (vdd net4) resistor r=0
R16 (gnd net3) resistor r=0
V9 (Cin gnd) vsource type=pulse val0=0 val1=800m period=5n delay=100p \
        rise=1p fall=1p
V17 (B\<7\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[7] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V8 (B\<6\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[6] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V16 (B\<5\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[5] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V7 (B\<4\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[4] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V15 (B\<3\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[3] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V6 (B\<2\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[2] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V14 (B\<1\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[1] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V5 (B\<0\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[0] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V13 (A\<7\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[7] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V4 (A\<6\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[6] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V12 (A\<5\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[5] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V3 (A\<4\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[4] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V11 (A\<3\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[3] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V2 (A\<2\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[2] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V10 (A\<1\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[1] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V100 (A\<0\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[0] else "0"} period=5n delay=100p \
        rise=1p fall=1p
V18 (clk gnd) vsource type=pulse val0=0 val1=800m period=5n delay=100p \
        rise=1p fall=1p
"""
    
    with open(filename, "w") as f:
        f.write(netlist)
        f.flush()
        f.close()

    return None


def update_ocean(filename, log_file):
    script = \
    f"""
simulator( 'spectre )
design(	 "/home/mheidary/simulation/delay_ADD8/spectre/schematic/netlist/netlist")
resultsDir( "/home/mheidary/simulation/delay_ADD8/spectre/schematic" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "1n"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
temp( 27 ) 
run()
selectResult( 'tran )
plot(getData("/o7") getData("/cout") getData("/clk") )
hardCopyOptions(?hcOutputFile "{log_file}.png" ?hcResolution 500 ?hcFontSize 18 ?hcOutputFormat "png" ?hcImageWidth 3000 ?hcImageHeight 2000)
hardCopy()

o7_delay = delay(?wf1 VT("/clk"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/o7"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
cout_delay = delay(?wf1 VT("/clk"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/cout"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)

reportFile = outfile("{log_file}")
ocnPrint(?output reportFile "o7" o7_delay)
ocnPrint(?output reportFile "cout" cout_delay)
close(reportFile)

exit
"""
    
    with open(filename, "w") as f:
        f.write(script)
        f.flush()
        f.close()

    return None


def run_ocean_script(script_address):
    subprocess.call(["ocean", "-replay", script_address])


def b(num: int, bit_len: int):
    num_cpy = num
    bit_num = list(map(int, reversed(format(num_cpy, f'0{bit_len}b'))))
    return bit_num


if __name__ == "__main__":

    random_count = 0

    while random_count < 1000:
        A = random.randint(0, 255)
        B = random.randint(0, 255)
        if b(A+B, 8+1)[-2:] == [0,0]:
            log.println(f"{A}+{B} ignored!")
            continue
        random_count += 1
        
        A_bin = b(A, 8)
        B_bin = b(B, 8)

        log.println(f"{A} {B} DONE.")
        update_netlist(NETLIST, A_bin, B_bin)

        log_name = f"log/log-ADD8-delay-{A}-{B}.txt"
        update_ocean(OCEAN, log_name)
        run_ocean_script(OCEAN)

    
        


