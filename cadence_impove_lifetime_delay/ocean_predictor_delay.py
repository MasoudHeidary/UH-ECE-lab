import subprocess
from tool.log import Log

"""
NOTE:
normal equation: "(A6 | B0) & (B0 | B1) & (~A6 | ~A7) & (~A6 | ~B0 | ~B1)"
purpose: finding the delay
"""

log = Log("tlog.txt")

NETLIST = "/home/mheidary/simulation/logic_predictor/spectre/schematic/netlist/netlist"
SCHEMATIC = "/home/mheidary/simulation/logic_predictor/spectre/schematic"
OCEAN = "tmp.ocn"


def update_netlist(filename, a, b, c, d):
    netlist = \
    f"""

// Library name: prjage
// Cell name: logic_predictor
// View name: schematic
V1 (net2 0) vsource dc=0 type=dc
V0 (net3 0) vsource dc=800.0m type=dc
R20 (gnd clk) resistor r=100
R17 (vdd net3) resistor r=0
R16 (gnd net2) resistor r=0
V18 (clk gnd) vsource type=pulse val0=0 val1=800m period=5n delay=100p \\
        rise=1p fall=1p
V100 (A gnd) vsource type=pulse val0=0 val1={"800m" if a else "0"} period=5n delay=100p \\
        rise=1p fall=1p
V10 (B gnd) vsource type=pulse val0=0 val1={"800m" if b else "0"} period=5n delay=100p \\
        rise=1p fall=1p
V2 (C gnd) vsource type=pulse val0=0 val1={"800m" if c else "0"} period=5n delay=100p \\
        rise=1p fall=1p
V11 (D gnd) vsource type=pulse val0=0 val1={"800m" if d else "0"} period=5n delay=100p \\
        rise=1p fall=1p
N22 (nA A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n nf=1 \\
        par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N21 (nB B gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n nf=1 \\
        par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N20 (nC C gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n nf=1 \\
        par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N19 (nD D gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n nf=1 \\
        par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N17 (net7 m3 gnd gnd) nfet w=320n l=20n as=26.88f ad=26.88f ps=808n \\
        pd=808n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N16 (net6 m2 net7 gnd) nfet w=320n l=20n as=26.88f ad=26.88f ps=808n \\
        pd=808n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N15 (net129 nA gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N14 (net129 nC gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N13 (m3 net129 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N12 (net129 nD gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N11 (net115 nA gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N10 (m2 net115 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N9 (net115 nB gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N8 (net135 C gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N7 (m1 net135 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N6 (net135 D gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N5 (net139 A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N4 (m0 net139 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N3 (net139 C gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N2 (net44 m0 net46 gnd) nfet w=320n l=20n as=26.88f ad=26.88f ps=808n \\
        pd=808n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N1 (out net44 gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
N0 (net46 m1 net6 gnd) nfet w=320n l=20n as=26.88f ad=26.88f ps=808n \\
        pd=808n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P22 (nA A vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n pd=488n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P21 (nB B vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n pd=488n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P20 (nC C vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n pd=488n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P19 (nD D vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n pd=488n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P17 (net44 m3 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P16 (net44 m2 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P15 (net130 nC net131 vdd) pfet w=480n l=20n as=40.32f ad=40.32f ps=1.128u \\
        pd=1.128u nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P14 (m3 net129 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P13 (net131 nA vdd vdd) pfet w=480n l=20n as=40.32f ad=40.32f ps=1.128u \\
        pd=1.128u nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P12 (net129 nD net130 vdd) pfet w=480n l=20n as=40.32f ad=40.32f ps=1.128u \\
        pd=1.128u nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P11 (m2 net115 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P10 (net93 nA vdd vdd) pfet w=320n l=20n as=26.88f ad=26.88f ps=808n \\
        pd=808n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P9 (net115 nB net93 vdd) pfet w=320n l=20n as=26.88f ad=26.88f ps=808n \\
        pd=808n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P8 (m1 net135 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P7 (net71 C vdd vdd) pfet w=320n l=20n as=26.88f ad=26.88f ps=808n pd=808n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P6 (net135 D net71 vdd) pfet w=320n l=20n as=26.88f ad=26.88f ps=808n \\
        pd=808n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P5 (m0 net139 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P4 (net48 A vdd vdd) pfet w=320n l=20n as=26.88f ad=26.88f ps=808n pd=808n \\
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \\
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P2 (net139 C net48 vdd) pfet w=320n l=20n as=26.88f ad=26.88f ps=808n \\
        pd=808n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P3 (out net44 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P1 (net44 m1 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
P0 (net44 m0 vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \\
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \\
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \\
        tie_orient=0 swrfmhc_local=0 analog=0
C3 (out gnd) capacitor c=1f
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
design(	 "{NETLIST}")
resultsDir( "{SCHEMATIC}" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "0.4n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
temp( 27 ) 
run()
selectResult( 'tran )
plot(getData("/clk") getData("/out") )
hardCopyOptions(?hcOutputFile "{log_file}.png" ?hcResolution 500 ?hcFontSize 18 ?hcOutputFormat "png" ?hcImageWidth 3000 ?hcImageHeight 2000)
hardCopy()

out_delay = delay(?wf1 VT("/clk"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/out"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)

reportFile = outfile("{log_file}")
ocnPrint(?output reportFile "out_delay" out_delay)
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

    for i in range(0, 2**4):
        bin_input = b(i, 4)

        log_file_name = f"log/logic_predictor_{i}.txt"

        update_netlist(NETLIST, bin_input[0], bin_input[1], bin_input[2], bin_input[3])

        update_ocean(OCEAN, log_file_name)
        run_ocean_script(OCEAN)