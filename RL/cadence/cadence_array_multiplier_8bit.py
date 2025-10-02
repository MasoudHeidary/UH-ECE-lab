import re
import subprocess

from tool.log import Log

CADENCE_SERVER = False
NETLIST_DIR = "/home/mheidary/simulation/array_multiplier_8bit/spectre/schematic/netlist/netlist" if CADENCE_SERVER else "./netlist"
RESULT_DIR = "/home/mheidary/simulation/array_multiplier_8bit/spectre/schematic" if CADENCE_SERVER else "./result"
OCEAN_DIR = "./tmp.ocn"



def update_netlist(netlist_name, vdd, pbody, inputA, inputB, period):
    netlist = \
    f"""

// Library name: prjage
// Cell name: andnb
// View name: schematic
subckt andnb A B GND O PBody VDD
    R4 (PBody pbody) resistor r=0
    R53 (out O) resistor r=0
    R2 (GND gnd) resistor r=0
    R0 (VDD vdd) resistor r=0
    N2 (net10 A net16 gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N0 (net16 B gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N1 (out net10 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P1 (net10 A vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P0 (net10 B vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (out net10 vdd pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
ends andnb
// End of subcircuit definition.

// Library name: prjage
// Cell name: 8andnb
// View name: schematic
subckt prjage_8andnb_schematic A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> \
        A\<6\> A\<7\> B\<0\> B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> \
        B\<7\> GND PBody VDD y\<0\> y\<1\> y\<2\> y\<3\> y\<4\> y\<5\> \
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
// Cell name: FAnb_simp
// View name: schematic
subckt FAnb_simp A B C carry gnd pbody sum vdd
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
    N32 (nC xx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N31 (C nxx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N39 (C xx carry gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N38 (A nxx carry gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
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
    P38 (A xx carry pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P32 (nC nxx sum pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P31 (C xx sum pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
ends FAnb_simp
// End of subcircuit definition.

// Library name: prjage
// Cell name: ADD8
// View name: schematic
subckt ADD8 A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\> \
        B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> B\<7\> S\<0\> S\<1\> \
        S\<2\> S\<3\> S\<4\> S\<5\> S\<6\> S\<7\> carry carryin gnd pbody \
        vdd
    I16 (A\<0\> B\<0\> carryin C\<0\> gnd pbody S\<0\> vdd) FAnb_simp
    I23 (A\<7\> B\<7\> C\<6\> carry gnd pbody S\<7\> vdd) FAnb_simp
    I22 (A\<6\> B\<6\> C\<5\> C\<6\> gnd pbody S\<6\> vdd) FAnb_simp
    I21 (A\<5\> B\<5\> C\<4\> C\<5\> gnd pbody S\<5\> vdd) FAnb_simp
    I20 (A\<4\> B\<4\> C\<3\> C\<4\> gnd pbody S\<4\> vdd) FAnb_simp
    I19 (A\<3\> B\<3\> C\<2\> C\<3\> gnd pbody S\<3\> vdd) FAnb_simp
    I18 (A\<2\> B\<2\> C\<1\> C\<2\> gnd pbody S\<2\> vdd) FAnb_simp
    I17 (A\<1\> B\<1\> C\<0\> C\<1\> gnd pbody S\<1\> vdd) FAnb_simp
ends ADD8
// End of subcircuit definition.

// Library name: prjage
// Cell name: MP8
// View name: schematic
subckt MP8 A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\> \
        B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> B\<7\> M\<0\> M\<10\> \
        M\<11\> M\<12\> M\<13\> M\<14\> M\<15\> M\<1\> M\<2\> M\<3\> \
        M\<4\> M\<5\> M\<6\> M\<7\> M\<8\> M\<9\> gnd pb vdd
    R124 (gnd zero) resistor r=0
    I151 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<1\> \
        B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> B\<1\> gnd pb vdd net16 \
        net5 net7 net76 net3 net4 net8 net1) prjage_8andnb_schematic
    I163 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<7\> \
        B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> B\<7\> gnd pb vdd net292 \
        net291 net290 net289 net288 net287 net286 net285) \
        prjage_8andnb_schematic
    I161 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<6\> \
        B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> B\<6\> gnd pb vdd net318 \
        net317 net316 net315 net314 net313 net312 net311) \
        prjage_8andnb_schematic
    I159 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<5\> \
        B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> B\<5\> gnd pb vdd net344 \
        net343 net342 net341 net340 net339 net338 net337) \
        prjage_8andnb_schematic
    I157 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<4\> \
        B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> B\<4\> gnd pb vdd net454 \
        net453 net452 net451 net450 net449 net448 net447) \
        prjage_8andnb_schematic
    I155 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<3\> \
        B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> B\<3\> gnd pb vdd net428 \
        net427 net426 net425 net424 net423 net422 net421) \
        prjage_8andnb_schematic
    I154 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<2\> \
        B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> B\<2\> gnd pb vdd net420 \
        net419 net418 net417 net416 net415 net414 net413) \
        prjage_8andnb_schematic
    I150 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\> \
        B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> B\<0\> gnd pb vdd M\<0\> \
        net10 net12 net14 net15 net2 net11 net13) prjage_8andnb_schematic
    I183 (net10 net12 net14 net15 net2 net11 net13 zero net16 net5 net7 \
        net76 net3 net4 net8 net1 M\<1\> net24 net23 net22 net21 net20 \
        net19 net18 net17 zero gnd pb vdd) ADD8
    I186 (net62 net61 net60 net59 net58 net57 net56 net55 net454 net453 \
        net452 net451 net450 net449 net448 net447 M\<4\> net352 net351 \
        net350 net349 net348 net347 net346 net345 zero gnd pb vdd) ADD8
    I185 (net52 net51 net50 net49 net48 net47 net46 net45 net428 net427 \
        net426 net425 net424 net423 net422 net421 M\<3\> net62 net61 net60 \
        net59 net58 net57 net56 net55 zero gnd pb vdd) ADD8
    I184 (net24 net23 net22 net21 net20 net19 net18 net17 net420 net419 \
        net418 net417 net416 net415 net414 net413 M\<2\> net52 net51 net50 \
        net49 net48 net47 net46 net45 zero gnd pb vdd) ADD8
    I187 (net352 net351 net350 net349 net348 net347 net346 net345 net344 \
        net343 net342 net341 net340 net339 net338 net337 M\<5\> net326 \
        net325 net324 net323 net322 net321 net320 net319 zero gnd pb vdd) \
        ADD8
    I188 (net326 net325 net324 net323 net322 net321 net320 net319 net318 \
        net317 net316 net315 net314 net313 net312 net311 M\<6\> net300 \
        net299 net298 net297 net296 net295 net294 net293 zero gnd pb vdd) \
        ADD8
    I189 (net300 net299 net298 net297 net296 net295 net294 net293 net292 \
        net291 net290 net289 net288 net287 net286 net285 M\<7\> M\<8\> \
        M\<9\> M\<10\> M\<11\> M\<12\> M\<13\> M\<14\> M\<15\> zero gnd pb \
        vdd) ADD8
ends MP8
// End of subcircuit definition.

// Library name: prjage
// Cell name: array_multiplier_8bit
// View name: schematic
R20 (gnd clk) resistor r=100
R75 (pbody net1) resistor r=0
R17 (vdd net5) resistor r=0
R16 (gnd net4) resistor r=0

V9 (net1 0) vsource dc={pbody} type=dc
V1 (net4 0) vsource dc=0 type=dc
V0 (net5 0) vsource dc={vdd} type=dc

V13 (A\<7\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputA[7] else "0"} period={period}n delay=100p rise=1p fall=1p
V4 (A\<6\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputA[6] else "0"} period={period}n delay=100p rise=1p fall=1p
V12 (A\<5\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputA[5] else "0"} period={period}n delay=100p rise=1p fall=1p
V3 (A\<4\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputA[4] else "0"} period={period}n delay=100p rise=1p fall=1p
V11 (A\<3\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputA[3] else "0"} period={period}n delay=100p rise=1p fall=1p
V2 (A\<2\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputA[2] else "0"} period={period}n delay=100p rise=1p fall=1p
V10 (A\<1\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputA[1] else "0"} period={period}n delay=100p rise=1p fall=1p
V100 (A\<0\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputA[0] else "0"} period={period}n delay=100p rise=1p fall=1p
V17 (B\<7\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputB[7] else "0"} period={period}n delay=100p rise=1p fall=1p
V8 (B\<6\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputB[6] else "0"} period={period}n delay=100p rise=1p fall=1p
V16 (B\<5\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputB[5] else "0"} period={period}n delay=100p rise=1p fall=1p
V7 (B\<4\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputB[4] else "0"} period={period}n delay=100p rise=1p fall=1p
V15 (B\<3\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputB[3] else "0"} period={period}n delay=100p rise=1p fall=1p
V6 (B\<2\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputB[2] else "0"} period={period}n delay=100p rise=1p fall=1p
V14 (B\<1\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputB[1] else "0"} period={period}n delay=100p rise=1p fall=1p
V5 (B\<0\> gnd) vsource type=pulse val0=0 val1={str(vdd) if inputB[0] else "0"} period={period}n delay=100p rise=1p fall=1p

V18 (clk gnd) vsource type=pulse val0=0 val1={vdd} period={period}n delay=100p rise=1f fall=1f
C18 (M15 gnd) capacitor c=1f
C17 (M14 gnd) capacitor c=1f
C16 (net8 gnd) capacitor c=1f
C15 (net9 gnd) capacitor c=1f
C14 (net10 gnd) capacitor c=1f
C13 (net11 gnd) capacitor c=1f
C12 (net12 gnd) capacitor c=1f
C11 (net13 gnd) capacitor c=1f
C10 (net14 gnd) capacitor c=1f
C9 (net15 gnd) capacitor c=1f
C8 (net16 gnd) capacitor c=1f
C7 (net17 gnd) capacitor c=1f
C6 (net18 gnd) capacitor c=1f
C5 (net19 gnd) capacitor c=1f
C4 (net20 gnd) capacitor c=1f
C3 (net21 gnd) capacitor c=1f
I5 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\> B\<1\> \
        B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> B\<7\> net21 net11 net10 net9 \
        net8 M14 M15 net20 net19 net18 net17 net16 net15 net14 net13 net12 \
        gnd pbody vdd) MP8
"""
    
    with open(netlist_name, "w") as f:
        f.write(netlist)
        f.flush()
        f.close()
    
    return True


def update_ocean(ocean_name, log_file, period):
    script = \
    f"""
simulator( 'spectre )
design(	 "{NETLIST_DIR}")
resultsDir( "{RESULT_DIR}" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "{period}n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
save( 'i "/V0/PLUS" )
temp( 27 ) 
run()
selectResult( 'tran )
plot(getData("/V0/PLUS") getData("/vdd") getData("/M14") getData("/M15") getData("/clk") )

M15_delay = delay(?wf1 VT("/clk"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M15"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
M14_delay = delay(?wf1 VT("/clk"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M14"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
vddpower = average( abs( VT("/vdd") * IT("/V0/PLUS") ) )


reportFile = outfile("{log_file}")
ocnPrint(?output reportFile "15" M15_delay)
ocnPrint(?output reportFile "14" M14_delay)
ocnPrint(?output reportFile "power" vddpower)

close(reportFile)
exit()
"""

    with open(ocean_name, "w") as f:
        f.write(script)
        f.flush()
        f.close()
    
    return True


def run_ocean_script(ocean_name):
    subprocess.call(["ocean", "-replay", ocean_name])


def bin(num: int, bit_len: int):
    num_cpy = num
    bit_num = list(map(int, reversed(format(num_cpy, f'0{bit_len}b'))))
    return bit_num


def read_log_file(ocean_logname, log):
    text = open(ocean_logname, 'r').read()

    delay_pattern = r"(\w+)\s+([\d\.]+[a-z])"
    delays = {match[0]: match[1] for match in re.findall(delay_pattern, text) if match[0] != "power"}

    power_pattern = r"power\s+([\d\.]+[a-z])"
    power_values = re.findall(power_pattern, text)

    ret = {
        "delays": delays,
        "power": power_values
    }
    if log:
        log.println(f"{ret}")
    return ret

def get_delay_power(ocean_logname, log, force=False):
    try:
        r = read_log_file(ocean_logname, log)
    except:
        if not force:
            return
        raise RuntimeError(f"ocean_log not exist - {ocean_logname}")
    
    delay = r['delays'].get('14', False)
    if not delay:
        return False    # circuit is off or too LONG propagation delay
    
    power = r['power']
    return (delay, power[0])


def log_filename(vdd, pbody, A, B):
    return f"./log/MP8_vdd{vdd}_pbody{pbody}_A{A}_B{B}.txt"


if __name__ == "__main__":
    
    # standard input case to find delay
    A = 83
    B = 199
    Abin = bin(A, 8)
    Bbin = bin(B, 8)
    period = 20 #ns -> matters in simulating power usage

    log = Log(f"{__file__}.log", terminal=True)
    vdd_range = range(60, 90+1, 5) # /100 (vdd voltage 0.6v -> 0.65 -> ... -> 0.9v)

    if CADENCE_SERVER:
        log.println(f"# RUNNING IN CADENCE SERVER MODE")
        
        for r_vdd in vdd_range:
            pbody_range = range(r_vdd, 1000)   # /100 (body voltage vdd -> 0.81v -> ... -> 9.9v)
            for r_pbody in pbody_range:
                vdd = r_vdd / 100
                pbody = r_pbody / 100
                ocean_log = log_filename(vdd, pbody, A, B)

                update_netlist(NETLIST_DIR, vdd, pbody, Abin, Bbin, period)
                update_ocean(OCEAN_DIR, ocean_log, period)
                run_ocean_script(OCEAN_DIR)

                log.println(f"vdd [{vdd:.3f}] pbody [{pbody:.3f}] A[{A:3}] B[{B:3}] DONE")

                # early exit if circuit is off after this point
                if (not get_delay_power(ocean_log, log=False)):
                    break

    else:
        log.println(f"# RUNNING IN CLIENT MODE")

        for r_vdd in vdd_range:
            pbody_range = range(r_vdd, 1000)   # /100 (body voltage vdd -> 0.81v -> ... -> 9.9v)
            for r_pbody in pbody_range:
                vdd = r_vdd / 100
                pbody = r_pbody / 100
                ocean_log = log_filename(vdd, pbody, A, B)

                r = get_delay_power(ocean_log, log=False)
                if not r:
                    continue    # multiplier is not ON, and power & delay dont exist

                delay, power = r
                log.println(f"vdd [{vdd:.3f}] pbody [{pbody:.3f}] \t=>\t delay [{delay:9}] power [{power:9}]")
        