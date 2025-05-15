
import re

import subprocess
from tool import vth_body_map
from tool.log import Log

CADENCE_SERVER = False
NETLIST_DIR = "/home/mheidary/simulation/single_tgate_delay_power/spectre/schematic/netlist/netlist" if CADENCE_SERVER else "./netlist"
RESULT_DIR = "/home/mheidary/simulation/single_tgate_delay_power/spectre/schematic" if CADENCE_SERVER else "./result"
OCEAN_DIR = "./tmp.ocn"

# pb: PMOS body voltage (V)
def update_netlist(netlist_name, pb):
    netlist = \
    f"""
// Library name: prjage
// Cell name: single_tgate_delay_power
// View name: schematic
P33 (net14 step vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
P0 (vdd step out pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
P32 (gnd net14 out pbody) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
V0 (vdd 0) vsource dc=800.0m type=dc
V1 (gnd 0) vsource dc=0 type=dc
V2 (pb gnd) vsource dc={pb} type=dc
V100 (step gnd) vsource type=pulse val0=800m val1=0 period=2n delay=1n \
        rise=1f fall=1f
N33 (net14 step gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
N32 (gnd step out gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
N0 (vdd net14 out gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
C0 (out gnd) capacitor c=6f
    """

    with open(netlist_name, "w") as f:
        f.write(netlist)
        f.flush()
        f.close()

    return True


def update_ocean(ocean_name: str, log_file: str):
    script = \
    f"""
simulator( 'spectre )
design(	 "{NETLIST_DIR}")
resultsDir( "{RESULT_DIR}" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "1000n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
save( 'i "/V0/PLUS" )
temp( 27 ) 
run()
selectResult( 'tran )

plot(getData("/out") getData("/step") getData("/V0/PLUS") )

ps_delay = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "falling", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/out"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
cir_power = average( abs(IT("/V0/PLUS") * VT("/vdd")) )

report_file = outfile("{log_file}")
ocnPrint(?output report_file "ps_delay" ps_delay)
ocnPrint(?output report_file "cir_power" cir_power)

exit()
    """

    with open(ocean_name, "w") as f:
        f.write(script)
        f.flush()
        f.close()

    return True


def run_ocean_script(ocean_name):
    subprocess.call(["ocean", "-replay", ocean_name])


def read_log_file(log_file):
    delay = None
    power = None
    with open(log_file, "r") as file:
        for line in file:
            if line.strip().startswith("ps_delay"):
                match = re.search(r"([\d.]+[pn]?)", line)
                if match:
                    delay = match.group(0)
            elif line.strip().startswith("cir_power"):
                match = re.search(r"([\d.]+u)", line)
                if match:
                    power = match.group(0)

        if (not delay) or (not power):
            msg = f"missing data in {log_file}"
            raise RuntimeError(msg)
        return {'delay': delay, 'power': power}



if __name__ == "__main__":
    log = Log(f"{__file__}.log", terminal=True)

    if CADENCE_SERVER:
        log.println(f"# RUNNING IN CADENCE SERVER MODE")
        for r_vth in range(450, 748, 2):    #maximum Vth in mapping
            vth = r_vth / 1000
            pb = vth_body_map.get_body_voltage(vth)
            ocean_log = f"./log/single_tgate_vth_{vth}.txt"

            update_netlist(NETLIST_DIR, pb)
            update_ocean(OCEAN_DIR, ocean_log)
            run_ocean_script(OCEAN_DIR)

            log.println(f"vth: {vth} DONE")

    else:
        log.println(f"# RUNNING IN CLIENT MODE")
        
        vth_lst = []
        delay_lst = []
        power_lst = []
        for r_vth in range(450, 748, 2):    #maximum Vth in mapping
            vth = r_vth / 1000

            ocean_log = f"./log/single_tgate_vth_{vth}.txt"
            delay, power = read_log_file(ocean_log).values()

            vth_lst.append(vth)
            delay_lst.append(delay)
            power_lst.append(power)

        log.println(f"extracted data as follow")
        log.println(f"vth: \n{vth_lst}")
        log.println(f"delay: \n{delay_lst}")
        log.println(f"power: \n{power_lst}")