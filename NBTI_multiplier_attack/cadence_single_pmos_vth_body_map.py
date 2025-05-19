
"""
purpose: mapping the Vth and Body Voltage values together
"""

import re

import subprocess
from tool.log import Log

CADENCE_SERVER = False
NETLIST_DIR = "/home/mheidary/simulation/vth/spectre/schematic/netlist/netlist" if CADENCE_SERVER else "./netlist"
RESULT_DIR = "/home/mheidary/simulation/vth/spectre/schematic" if CADENCE_SERVER else "./result"
OCEAN_DIR = "./tmp.ocn"

# pb: PMOS body voltage (V)
def update_netlist(netlist_name, pb):
    netlist = \
    f"""
// Library name: prjage
// Cell name: vth
// View name: schematic
P3 (gnd step net1 pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
V3 (pbody 0) vsource dc={pb} type=dc
V2 (step gnd) vsource type=dc
V1 (gnd 0) vsource dc=0 type=dc
V0 (vdd 0) vsource dc=800.0m type=dc
R167 (vdd net1) resistor r=0
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
analysis('dc ?dev "/V2"  ?param "dc"  ?start "-0.8"  
		?stop "0.8"  ?step "0.001"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("dc") 
)
save( 'i "/R167/PLUS" )
temp( 27 ) 
run()
selectResult( 'dc )

reportFile = outfile("{log_file}")
ocnPrint(?output reportFile getData("/R167/PLUS"))
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


def read_log_file(log_file):
    with open(log_file) as file:
        # skipping headers
        for _ in range(3):
            file.readline()

        pattern = r"(-?[0-9.]+)([a-zA-Z]+)"
        num = []
        for line in file.readlines():
            match = re.findall(pattern, line)
            if len(match) >= 2:
                num += [match]
    
    for rev in reversed(num):
        if(rev[1][1] == 'u'):
            if(float(rev[1][0]) > 10):
                vth = round(float(rev[0][0]) / 1000 - 0.8, 5)
                return vth
    
    msg = f"ERROR: provided log file {log_file} does not have Vth data"
    log.println(msg)
    raise RuntimeError(msg)




if __name__ == "__main__":
    log = Log(f"{__file__}.log", terminal=True)
    pbody_range = range(80, 1000)   # /100

    if CADENCE_SERVER:
        log.println(f"# RUNNING IN CADENCE SERVER MODE")
        
        for r_pb in pbody_range:
            pb = r_pb / 100

            ocean_log = f"./log/single_pmos_pb_{pb}.txt"

            update_netlist(NETLIST_DIR, pb)
            update_ocean(OCEAN_DIR, ocean_log)
            run_ocean_script(OCEAN_DIR)

            log.println(f"pb: {pb} DONE")

    else:
        log.println(f"# RUNNING IN CLIENT MODE")
        res = []

        for r_pb in pbody_range:
            pb = r_pb / 100

            ocean_log = f"./log/single_pmos_pb_{pb}.txt"
            log.println(f"{pb} \t=>\t {read_log_file(ocean_log)}")
            res.append(
                (pb, read_log_file(ocean_log))
            )
        
        # log.println(f"result: \n{res}")
