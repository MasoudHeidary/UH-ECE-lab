import subprocess

def generate_netlist(vdd, pbody, period):
    _netlist = \
    f"""

// Library name: prjage
// Cell name: FAnb
// View name: schematic
subckt FAnb A B Cin Cout GND PBody S VDD
    N1 (nA A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N2 (A nB xx gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N3 (nA B xx gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N32 (net16 dxx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N31 (net20 nxx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N39 (C dxx carry gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N38 (net6 nxx carry gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N33 (net20 net18 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N34 (net18 net9 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N36 (net16 net8 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N26 (nB B gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N27 (nC C gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N28 (nxx xx gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N29 (net11 xx gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N30 (dxx net11 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N35 (net9 nC gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N37 (net8 nC gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N40 (net6 net4 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N41 (net4 net1 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N42 (net2 A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    N43 (net1 net2 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (nA A vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P26 (nB B vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P1 (A B xx pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P4 (nA nB xx pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P27 (nC C vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P28 (nxx xx vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P39 (C nxx carry pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P38 (net6 dxx carry pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P32 (net16 nxx sum pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P31 (net20 dxx sum pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \\
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P33 (net20 net18 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \\
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \\
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \\
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P34 (net18 net9 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \\
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \\
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \\
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P36 (net16 net8 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \\
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \\
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \\
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P29 (net11 xx vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P30 (dxx net11 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \\
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \\
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \\
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P35 (net9 nC vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P37 (net8 nC vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P40 (net6 net4 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \\
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \\
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \\
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P41 (net4 net1 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \\
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \\
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \\
        lle_pcpc=84n tie_orient=0 swrfmhc_local=0 analog=0
    P42 (net2 A vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P43 (net1 net2 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f \\
        ps=488n pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 \\
        acv_opt=-1 ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 \\
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
subckt prjage_8ADnb_schematic A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> \\
        A\<6\> A\<7\> B\<0\> B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> \\
        B\<7\> Cin Cout GND O\<0\> O\<1\> O\<2\> O\<3\> O\<4\> O\<5\> \\
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
    P1 (net10 A vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P0 (net10 B vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (out net10 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
ends andnb
// End of subcircuit definition.

// Library name: prjage
// Cell name: 8andnb
// View name: schematic
subckt prjage_8andnb_schematic A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> \\
        A\<6\> A\<7\> B\<0\> B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> \\
        B\<7\> GND PBody VDD y\<0\> y\<1\> y\<2\> y\<3\> y\<4\> y\<5\> \\
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
// Cell name: 8MPnb
// View name: schematic
subckt prjage_8MPnb_schematic A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> \\
        A\<6\> A\<7\> B\<0\> B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> \\
        B\<7\> M\<0\> M\<10\> M\<11\> M\<12\> M\<13\> M\<14\> M\<15\> \\
        M\<1\> M\<2\> M\<3\> M\<4\> M\<5\> M\<6\> M\<7\> M\<8\> M\<9\> gnd \\
        pb vdd
    I183 (net10 net12 net14 net15 net2 net11 net13 zero net16 net5 net7 \\
        net76 net3 net4 net8 net1 zero net17 gnd M\<1\> net24 net23 net22 \\
        net21 net20 net19 net18 vdd pb) prjage_8ADnb_schematic
    I186 (net62 net61 net60 net59 net58 net57 net56 net55 net454 net453 \\
        net452 net451 net450 net449 net448 net447 zero net345 gnd M\<4\> \\
        net352 net351 net350 net349 net348 net347 net346 vdd pb) \\
        prjage_8ADnb_schematic
    I185 (net52 net51 net50 net49 net48 net47 net46 net45 net428 net427 \\
        net426 net425 net424 net423 net422 net421 zero net55 gnd M\<3\> \\
        net62 net61 net60 net59 net58 net57 net56 vdd pb) \\
        prjage_8ADnb_schematic
    I184 (net24 net23 net22 net21 net20 net19 net18 net17 net420 net419 \\
        net418 net417 net416 net415 net414 net413 zero net45 gnd M\<2\> \\
        net52 net51 net50 net49 net48 net47 net46 vdd pb) \\
        prjage_8ADnb_schematic
    I187 (net352 net351 net350 net349 net348 net347 net346 net345 net344 \\
        net343 net342 net341 net340 net339 net338 net337 zero net319 gnd \\
        M\<5\> net326 net325 net324 net323 net322 net321 net320 vdd pb) \\
        prjage_8ADnb_schematic
    I188 (net326 net325 net324 net323 net322 net321 net320 net319 net318 \\
        net317 net316 net315 net314 net313 net312 net311 zero net293 gnd \\
        M\<6\> net300 net299 net298 net297 net296 net295 net294 vdd pb) \\
        prjage_8ADnb_schematic
    I189 (net300 net299 net298 net297 net296 net295 net294 net293 net292 \\
        net291 net290 net289 net288 net287 net286 net285 zero M\<15\> gnd \\
        M\<7\> M\<8\> M\<9\> M\<10\> M\<11\> M\<12\> M\<13\> M\<14\> vdd \\
        pb) prjage_8ADnb_schematic
    R124 (gnd zero) resistor r=0
    I151 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<1\> \\
        B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> gnd pb vdd net16 \\
        net5 net7 net76 net3 net4 net8 net1) prjage_8andnb_schematic
    I163 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<7\> \\
        B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> gnd pb vdd net292 \\
        net291 net290 net289 net288 net287 net286 net285) \\
        prjage_8andnb_schematic
    I161 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<6\> \\
        B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> gnd pb vdd net318 \\
        net317 net316 net315 net314 net313 net312 net311) \\
        prjage_8andnb_schematic
    I159 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<5\> \\
        B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> gnd pb vdd net344 \\
        net343 net342 net341 net340 net339 net338 net337) \\
        prjage_8andnb_schematic
    I157 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<4\> \\
        B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> gnd pb vdd net454 \\
        net453 net452 net451 net450 net449 net448 net447) \\
        prjage_8andnb_schematic
    I155 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<3\> \\
        B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> gnd pb vdd net428 \\
        net427 net426 net425 net424 net423 net422 net421) \\
        prjage_8andnb_schematic
    I154 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<2\> \\
        B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> gnd pb vdd net420 \\
        net419 net418 net417 net416 net415 net414 net413) \\
        prjage_8andnb_schematic
    I150 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\> \\
        B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> gnd pb vdd M\<0\> \\
        net10 net12 net14 net15 net2 net11 net13) prjage_8andnb_schematic
ends prjage_8MPnb_schematic
// End of subcircuit definition.

// Library name: prjage
// Cell name: del
// View name: schematic
C17 (net3 gnd) capacitor c=1f
C0 (_net0 gnd) capacitor c=1f
C24 (net12 gnd) capacitor c=1f
C25 (net11 gnd) capacitor c=1f
C16 (net4 gnd) capacitor c=1f
C19 (net5 gnd) capacitor c=1f
C20 (net8 gnd) capacitor c=1f
C18 (net6 gnd) capacitor c=1f
C26 (net14 gnd) capacitor c=1f
C27 (net13 gnd) capacitor c=1f
C22 (net10 gnd) capacitor c=1f
C28 (net1 gnd) capacitor c=1f
C23 (net9 gnd) capacitor c=1f
C15 (net2 gnd) capacitor c=1f
C21 (net7 gnd) capacitor c=1f
C29 (net15 gnd) capacitor c=1f
I5416 (gnd step step gnd step step gnd step gnd step step gnd step step \\
        gnd step net1 net6 net5 net4 net3 net2 _net0 net15 net14 net13 \\
        net12 net11 net10 net9 net8 net7 gnd pb vdd) \\
        prjage_8MPnb_schematic
V0 (vdd 0) vsource dc={vdd} type=dc
V1 (gnd 0) vsource dc=0 type=dc
V3 (pb 0) vsource dc={pbody} type=dc
V2 (step gnd) vsource type=pulse val0=0 val1={vdd} period={period} delay=1p \\
        rise=1p fall=1p
        """
    
    return _netlist


def update_netlist_file(file_name, code):
    _file = open(file_name, "w")
    _file.write(code)
    _file.flush()
    _file.close()


def generate_ocean_script(log_file, vdd, pbody, period):
    _script = \
    f"""
simulator( 'spectre )
design(	 "/home/mheidary/simulation/del/spectre/schematic/netlist/netlist")
resultsDir( "/home/mheidary/simulation/del/spectre/schematic" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "10n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
save( 'v "/end" "/step" "/vdd" )
save( 'i "/V0/PLUS" )
temp( 27 ) 
run()
selectResult( 'tran )
plot(getData("/end") getData("/step") getData("/vdd") getData("/V0/PLUS") )
hardCopyOptions(?hcOutputFile "{log_file}.png")
hardCopy()

reportFile = outfile("{log_file}.txt")
_delay = delay(?wf1 VT("/step") ?value1 0.4 ?edge1 "rising" ?nth1 1 ?td1 0.0 ?tol1 nil ?wf2 VT("/end") ?value2 0.4 ?edge2 "rising" ?nth2 1 ?tol2 nil ?td2 nil ?stop nil ?multiple nil)
_power = average(abs((VT("/vdd") * IT("/V0/PLUS"))))
_max = ymax(VT("/end"))
/*
fprintf(reportFile "LOG FILE HEADER(vdd, pbody, period) (delay(ns), power(uw), max_voltage(mv))\\n")
fprintf(reportFile "%f %f %f\\n" {vdd} {pbody} {period})
fprintf(reportFile "%f\\n" _delay*1e9)
fprintf(reportFile "%f\\n" _power*1e6)
fprintf(reportFile "%f\\n" _max*1e3)
*/
fprintf(reportFile "{{\\\"vdd\\\":%f, \\\"pbody\\\":%f, \\\"period\\\":%f, \\\"delay\\\":%f, \\\"power\\\":%f, \\\"max\\\":%f }}" {vdd} {pbody} {period} _delay*1e9 _power*1e6 _max)
close(reportFile)

exit
    """

    return _script
    

def update_ocean_script_file(file_address, ocean_script):
    _file = open(file_address, "w")
    _file.write(ocean_script)
    _file.flush()
    _file.close()


def run_ocean_script(script_address):
    subprocess.call(["ocean", "-replay", script_address])




###
"""
vdd = 0.8
pbody = 0.8
period = 15.0
netlist = generate_netlist(vdd, pbody, f"{period}n")
update_netlist_file("/home/mheidary/simulation/del/spectre/schematic/netlist/netlist", netlist)

script = generate_ocean_script(f"./log/000.txt", vdd, pbody, period)
update_ocean_script_file("./ocean.ocn", script)

run_ocean_script("./ocean.ocn")
exit()
"""
###

pylog = open("pylog.txt", "w")
pylog.write("INIT\n")
pylog.flush()
log_counter = 0

period = 19.0
#for period in [1/(i/1000) for i in range(100, 500+40, 40)]:
for vdd in [i/100 for i in range(60, 150-4, 4)]:
        for pbody in [i/100 for i in range(int(vdd*100), int(vdd*100)+100, 2)]:
            netlist = generate_netlist(vdd, pbody, f"{period}n")
            update_netlist_file("/home/mheidary/simulation/del/spectre/schematic/netlist/netlist", netlist)

            script = generate_ocean_script(f"./log/{log_counter}", vdd, pbody, period)
            update_ocean_script_file("./ocean.ocn", script)
            run_ocean_script("./ocean.ocn")

            log_counter += 1
            pylog.write(f"{period}ns - {vdd}v - {pbody}v\n")
            pylog.flush()
            #print(f"{period}ns - {vdd} - {pbody}")
        
        print(f"### {period}ns - {vdd}v - DONE ###")

pylog.close()

