import subprocess


def update_netlist(file_name: str, pand, pfa):
    _netlist = \
    f"""

// Library name: prjage
// Cell name: FA_for_trojan
// View name: schematic
subckt FA_for_trojan A B C carry gnd pb sum vdd
    N31 (C nxx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N32 (nC xx sum gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
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
    N2 (A nB xx gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N3 (nA B xx gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N1 (nA A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
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
    P31 (C xx sum pb) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P32 (nC nxx sum pb) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P39 (C nxx carry pb) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P38 (A xx carry pb) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P1 (A B xx pb) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n nf=1 \
        par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P4 (nA nB xx pb) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (nA A vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n pd=488n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P26 (nB B vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P27 (nC C vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P28 (nxx xx vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
ends FA_for_trojan
// End of subcircuit definition.

// Library name: prjage
// Cell name: and_for_trojan
// View name: schematic
subckt and_for_trojan gnd pb r vdd x y
    N2 (net10 x net16 gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N0 (net16 y gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N1 (r net10 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P1 (net10 x vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P0 (net10 y vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (r net10 vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
ends and_for_trojan
// End of subcircuit definition.

// Library name: prjage
// Cell name: main_MP4
// View name: schematic
I30 (net54 net36 net27 net113 gnd pfa\<6\> net52 vdd) FA_for_trojan
I40 (net15 net58 net100 M\<7\> gnd pfa\<11\> M\<6\> vdd) FA_for_trojan
I38 (net13 net57 net10 net100 gnd pfa\<10\> M\<5\> vdd) FA_for_trojan
I36 (net8 net52 net2 net10 gnd pfa\<9\> M\<4\> vdd) FA_for_trojan
I34 (net3 net47 gnd net2 gnd pfa\<8\> M\<3\> vdd) FA_for_trojan
I32 (net40 net37 net113 net58 gnd pfa\<7\> net57 vdd) FA_for_trojan
I28 (net49 net31 net43 net27 gnd pfa\<5\> net47 vdd) FA_for_trojan
I26 (net44 net26 gnd net43 gnd pfa\<4\> M\<2\> vdd) FA_for_trojan
I24 (net18 gnd net112 net37 gnd pfa\<3\> net36 vdd) FA_for_trojan
I22 (net33 net61 net111 net112 gnd pfa\<2\> net31 vdd) FA_for_trojan
I20 (net28 net79 net1 net111 gnd pfa\<1\> net26 vdd) FA_for_trojan
I18 (net23 net69 gnd net1 gnd pfa\<0\> M\<1\> vdd) FA_for_trojan
V15 (pfa\<11\> gnd) vsource dc={pfa[11]} type=dc
V14 (pfa\<10\> gnd) vsource dc={pfa[10]} type=dc
V13 (pfa\<9\> gnd) vsource dc={pfa[9]} type=dc
V12 (pfa\<8\> gnd) vsource dc={pfa[8]} type=dc
V11 (pfa\<7\> gnd) vsource dc={pfa[7]} type=dc
V10 (pfa\<6\> gnd) vsource dc={pfa[6]} type=dc
V9 (pfa\<5\> gnd) vsource dc={pfa[5]} type=dc
V8 (pfa\<4\> gnd) vsource dc={pfa[4]} type=dc
V7 (pfa\<3\> gnd) vsource dc={pfa[3]} type=dc
V6 (pfa\<2\> gnd) vsource dc={pfa[2]} type=dc
V4 (pfa\<1\> gnd) vsource dc={pfa[1]} type=dc
V3 (pfa\<0\> gnd) vsource dc={pfa[0]} type=dc
V2 (pand gnd) vsource dc={pand} type=dc
V0 (vdd 0) vsource dc=800.0m type=dc
V1 (gnd 0) vsource dc=0 type=dc
V5 (step gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
I25 (gnd pand net18 vdd B\<3\> A\<1\>) and_for_trojan
I45 (gnd pand net61 vdd B\<3\> A\<0\>) and_for_trojan
I44 (gnd pand M\<0\> vdd B\<0\> A\<0\>) and_for_trojan
I43 (gnd pand net79 vdd B\<2\> A\<0\>) and_for_trojan
I42 (gnd pand net69 vdd B\<1\> A\<0\>) and_for_trojan
I41 (gnd pand net15 vdd B\<3\> A\<3\>) and_for_trojan
I39 (gnd pand net13 vdd B\<2\> A\<3\>) and_for_trojan
I37 (gnd pand net8 vdd B\<1\> A\<3\>) and_for_trojan
I35 (gnd pand net3 vdd B\<0\> A\<3\>) and_for_trojan
I33 (gnd pand net40 vdd B\<3\> A\<2\>) and_for_trojan
I31 (gnd pand net54 vdd B\<2\> A\<2\>) and_for_trojan
I29 (gnd pand net49 vdd B\<1\> A\<2\>) and_for_trojan
I27 (gnd pand net44 vdd B\<0\> A\<2\>) and_for_trojan
I23 (gnd pand net33 vdd B\<2\> A\<1\>) and_for_trojan
I21 (gnd pand net28 vdd B\<1\> A\<1\>) and_for_trojan
I19 (gnd pand net23 vdd B\<0\> A\<1\>) and_for_trojan
R22 (step B\<3\>) resistor r=0
R21 (step B\<2\>) resistor r=0
R20 (step B\<1\>) resistor r=0
R19 (step B\<0\>) resistor r=0
R18 (step A\<3\>) resistor r=0
R17 (step A\<2\>) resistor r=0
R16 (step A\<1\>) resistor r=0
R15 (step A\<0\>) resistor r=0
C7 (M\<7\> gnd) capacitor c=1f
C6 (M\<6\> gnd) capacitor c=1f
C5 (M\<5\> gnd) capacitor c=1f
C4 (M\<4\> gnd) capacitor c=1f
C3 (M\<3\> gnd) capacitor c=1f
C2 (M\<2\> gnd) capacitor c=1f
C1 (M\<1\> gnd) capacitor c=1f
C0 (M\<0\> gnd) capacitor c=1f
"""
    
    with open(file_name, "w") as f:
        f.write(_netlist)
        f.flush()
        f.close()

    return None



def update_ocean(file_name: str, log_file):
    script = \
    f"""
simulator( 'spectre )
design(	 "/home/mheidary/simulation/main_MP4/spectre/schematic/netlist/netlist")
resultsDir( "/home/mheidary/simulation/main_MP4/spectre/schematic" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "5n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
save( 'v "/step" )
temp( 27 ) 
run()
selectResult( 'tran )
/*
plot(getData("/step") getData("/M<0>") getData("/M<1>") getData("/M<2>") getData("/M<3>") getData("/M<4>") getData("/M<5>") getData("/M<6>") getData("/M<7>") )
*/

MP7_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<7>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
MP6_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<6>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
MP5_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<5>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
MP4_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<4>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
MP3_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<3>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
MP2_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<2>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
MP1_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<1>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
MP0_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<0>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)


reportFile = outfile("{log_file}")
/*fprintf(reportFile "%f\\n%f\\n%f\\n%f\\n" MP7_delay*1e12 MP6_delay*1e12 MP5_delay*1e12 MP4_delay*1e12)*/
ocnPrint(?output reportFile "7" MP7_delay)
ocnPrint(?output reportFile "6" MP6_delay)
ocnPrint(?output reportFile "5" MP5_delay)
ocnPrint(?output reportFile "4" MP4_delay)
ocnPrint(?output reportFile "3" MP3_delay)
ocnPrint(?output reportFile "2" MP2_delay)
ocnPrint(?output reportFile "1" MP1_delay)
ocnPrint(?output reportFile "0" MP0_delay)
close(reportFile)

exit()

    """

    with open(file_name, "w") as f:
        f.write(script)
        f.flush()
        f.close()

    return None


def run_ocean_script(script_address):
    subprocess.call(["ocean", "-replay", script_address])


def __ocean_plot(file_name: str):
    script = \
    f"""
simulator( 'spectre )
design(	 "/home/mheidary/simulation/main_MP4/spectre/schematic/netlist/netlist")
resultsDir( "/home/mheidary/simulation/main_MP4/spectre/schematic" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "5n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
save( 'v "/step" )
temp( 27 ) 
run()
selectResult( 'tran )
plot(getData("/step") getData("/M<0>") getData("/M<1>") getData("/M<2>") getData("/M<3>") getData("/M<4>") getData("/M<5>") getData("/M<6>") getData("/M<7>") )
    """
        
    with open(file_name, "w") as f:
        f.write(script)
        f.flush()
        f.close()

    subprocess.call(["ocean", "-replay", file_name])



import lib.NBTI_formula as NBTI 
import lib.vth_body_map as VTH

for t_week in range(1, 100, 1):
    t_sec = t_week * 30 * 24 * 60 * 60
    

    # normal aging
    pand = 0.8

    alpha = [160, 160, 160, 192, 144, 160, 164, 192, 136, 148, 164, 192]
    vth = [(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, a/256, NBTI.Tclk, t_sec)) for a in alpha]
    pfa = [VTH.get_body_voltage(v) for v in vth]
    
    
    update_netlist("/home/mheidary/simulation/main_MP4/spectre/schematic/netlist/netlist", pand, pfa)
    update_ocean("./tmp_main_MP4.ocn", f"./log/N-{t_week}.txt")
    run_ocean_script("./tmp_main_MP4.ocn")
    
    
    # rewired aging
    alpha = [192, 208, 160, 192, 192, 232, 164, 192, 192, 196, 198, 202]
    vth = [(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, a/256, NBTI.Tclk, t_sec)) for a in alpha]
    pfa = [VTH.get_body_voltage(v) for v in vth]


    update_netlist("/home/mheidary/simulation/main_MP4/spectre/schematic/netlist/netlist", pand, pfa)
    update_ocean("./tmp_main_MP4.ocn", f"./log/M-{t_week}.txt")
    run_ocean_script("./tmp_main_MP4.ocn")
    


