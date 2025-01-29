import subprocess
import random
import re
from tool.log import Log

"""
NOTE:
MAX: {'A': 18275, 'B': 14507, 'delay': 432.82, 'result': {'delays': {'out': '432.82p'}, 'power': ['6.97419u']}}
"""

log = Log(f"{__file__}.log")

NETLIST = "/home/mheidary/simulation/delay_ADD16/spectre/schematic/netlist/netlist"
SCHEMATIC = "/home/mheidary/simulation/delay_ADD16/spectre/schematic"
OCEAN = "tmp.ocn"


def update_netlist(filename, A_bin, B_bin):
    netlist = \
    f"""

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
// Cell name: delay_ADD16_simp
// View name: schematic
C18 (cout gnd) capacitor c=1f
C17 (net14 gnd) capacitor c=1f
C16 (net13 gnd) capacitor c=1f
C15 (net12 gnd) capacitor c=1f
C14 (net11 gnd) capacitor c=1f
C13 (net10 gnd) capacitor c=1f
C12 (net9 gnd) capacitor c=1f
C11 (net8 gnd) capacitor c=1f
C10 (out gnd) capacitor c=1f
C8 (net6 gnd) capacitor c=1f
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

V26 (B\<15\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[15] else "0"} period=10n \
        delay=100p rise=1p fall=1p
V34 (B\<14\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[14] else "0"} period=10n \
        delay=100p rise=1p fall=1p
V25 (B\<13\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[13] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V33 (B\<12\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[12] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V24 (B\<11\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[11] else "0"} period=10n \
        delay=100p rise=1p fall=1p
V32 (B\<10\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[10] else "0"} period=10n \
        delay=100p rise=1p fall=1p
V23 (B\<9\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[9] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V31 (B\<8\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[8] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V22 (B\<7\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[7] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V30 (B\<6\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[6] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V21 (B\<5\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[5] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V29 (B\<4\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[4] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V20 (B\<3\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[3] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V28 (B\<2\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[2] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V19 (B\<1\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[1] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V27 (B\<0\> gnd) vsource type=pulse val0=0 val1={"800m" if B_bin[0] else "0"} period=10n delay=100p \
        rise=1p fall=1p

V17 (A\<15\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[15] else "0"} period=10n \
        delay=100p rise=1p fall=1p
V8 (A\<14\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[14] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V16 (A\<13\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[13] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V7 (A\<12\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[12] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V15 (A\<11\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[11] else "0"} period=10n \
        delay=100p rise=1p fall=1p
V6 (A\<10\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[10] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V14 (A\<9\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[9] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V5 (A\<8\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[8] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V13 (A\<7\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[7] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V4 (A\<6\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[6] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V12 (A\<5\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[5] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V3 (A\<4\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[4] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V11 (A\<3\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[3] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V2 (A\<2\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[2] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V10 (A\<1\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[1] else "0"} period=10n delay=100p \
        rise=1p fall=1p
V100 (A\<0\> gnd) vsource type=pulse val0=0 val1={"800m" if A_bin[0] else "0"} period=10n delay=100p \
        rise=1p fall=1p

V18 (clk gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
I183 (A\<0\> A\<1\> A\<2\> A\<3\> A\<4\> A\<5\> A\<6\> A\<7\> B\<0\> \
        B\<1\> B\<2\> B\<3\> B\<4\> B\<5\> B\<6\> B\<7\> net1 net28 net29 \
        net30 net31 net32 net33 net6 net2 gnd gnd vdd pbody) ADD8
I1 (A\<8\> A\<9\> A\<10\> A\<11\> A\<12\> A\<13\> A\<14\> A\<15\> B\<8\> \
        B\<9\> B\<10\> B\<11\> B\<12\> B\<13\> B\<14\> B\<15\> net14 net13 \
        net12 net11 net10 net9 net8 out cout net2 gnd vdd pbody) ADD8
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
analysis('tran ?stop "2n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
save( 'i "/R17/PLUS" )
temp( 27 ) 
run()
selectResult( 'tran )
plot(getData("/clk") getData("/vdd") getData("/R17/PLUS") getData("/out") getData("/cout") )
hardCopyOptions(?hcOutputFile "{log_file}.png" ?hcResolution 500 ?hcFontSize 18 ?hcOutputFormat "png" ?hcImageWidth 3000 ?hcImageHeight 2000)
hardCopy()

out_delay = delay(?wf1 VT("/clk"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/out"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
cout_delay = delay(?wf1 VT("/clk"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/cout"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
vddpower = average( abs( VT("/vdd")*IT("/R17/PLUS") ) )

reportFile = outfile("{log_file}")
ocnPrint(?output reportFile "out" out_delay)
ocnPrint(?output reportFile "cout" cout_delay)
ocnPrint(?output reportFile "power" vddpower)
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


def extract_delays_and_power(text):
    delay_pattern = r"(\w+)\s+([\d\.]+[a-z])"
    delays = {match[0]: match[1] for match in re.findall(delay_pattern, text) if match[0] != "power"}

    power_pattern = r"power\s+([\d\.]+[a-z])"
    power_values = re.findall(power_pattern, text)

    # Return results as a dictionary
    return {
        "delays": delays,
        "power": power_values
    }


if __name__ == "__main__":

    bit_len = 16

    sample_count = 0
    MAX_SAMPLE_COUNT = 5000

    max_delay_combination = {
        'A': 0,
        'B': 0,
        'delay': 0,
        'result': {}
    }

    while True:
        A = random.randint(0, 2**bit_len-1)
        B = random.randint(0, 2**bit_len-1)
        # checking bit "out", not "carry out"
        if b(A+B, bit_len+1)[-2] == 1:
            

            A_bin = b(A, bit_len)
            B_bin = b(B, bit_len)
            update_netlist(NETLIST, A_bin, B_bin)

            log_name = f"log/log-ADD16-delay-{A}-{B}.txt"
            update_ocean(OCEAN, log_name)
            run_ocean_script(OCEAN)

            result = extract_delays_and_power(
                open(log_name, 'r').read()
            )
            log.println(f"{sample_count}\t: {A} {B} =>\t {result}")

            out_delay   = float( result['delays'].get('out', '0.0p')[0:-1] )
            cout_delay  = float( result['delays'].get('cout', '0.0p')[0:-1] )
            if max(out_delay, cout_delay) > max_delay_combination['delay']:
                max_delay_combination['A'] = A
                max_delay_combination['B'] = B
                max_delay_combination['delay'] = max(out_delay, cout_delay)
                max_delay_combination['result'] = result

                log.println(f"$$$ MAX: {max_delay_combination}")

            sample_count += 1
            if sample_count >= MAX_SAMPLE_COUNT:
                log.println("PROGRAM DONE")
                break

        


