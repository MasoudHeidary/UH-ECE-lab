
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
def update_netlist(netlist_name, vdd, pbody):
    netlist = \
    f"""
// Library name: prjage
// Cell name: vth
// View name: schematic
P3 (gnd step net1 pbody) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
V3 (pbody 0) vsource dc={pbody} type=dc
V2 (step gnd) vsource type=dc
V1 (gnd 0) vsource dc=0 type=dc
V0 (vdd 0) vsource dc={vdd} type=dc
R167 (vdd net1) resistor r=0
    """

    with open(netlist_name, "w") as f:
        f.write(netlist)
        f.flush()
        f.close()

    return True


def update_ocean(ocean_name: str, vdd, log_file: str):
    script = \
    f"""
simulator( 'spectre )
design(	 "{NETLIST_DIR}")
resultsDir( "{RESULT_DIR}" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('dc ?dev "/V2"  ?param "dc"  ?start "-{vdd}"  
		?stop "{vdd}"  ?step "0.001"  )
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


def read_log_file(ocean_logname, log, force=False):
    with open(ocean_logname) as file:
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
    
    msg = f"ERROR: provided log file {ocean_logname} does not have Vth data"
    log.println(msg)
    if force:
        raise RuntimeError(msg)
    return False


def log_filename(vdd, pbody):
    return f"./log/single_pmos_vdd{vdd}_pbody{pbody}.txt"

if __name__ == "__main__":
    log = Log(f"{__file__}.log", terminal=True)
    vdd_range = range(60, 90+1, 5) # /100 (vdd voltage 0.6v -> 0.65 -> ... -> 0.9v)

    if CADENCE_SERVER:
        log.println(f"# RUNNING IN CADENCE SERVER MODE")
        
        for r_vdd in vdd_range:
            pbody_range = range(r_vdd, 1000)   # /100 (body voltage 0.8v -> 0.81v -> ... -> 9.9v)
            for r_pbody in pbody_range:
                vdd = r_vdd / 100
                pbody = r_pbody / 100
                ocean_log = log_filename(vdd, pbody)

                update_netlist(NETLIST_DIR, vdd, pbody)
                update_ocean(OCEAN_DIR, vdd, ocean_log)
                run_ocean_script(OCEAN_DIR)

                log.println(f"vdd [{vdd:.3f}] pbody [{pbody:.3f}] DONE")

    else:
        log.println(f"# RUNNING IN CLIENT MODE")

        for r_vdd in vdd_range:
            res = []
            pbody_range = range(r_vdd, 1000)   # /100 (body voltage 0.8v -> 0.81v -> ... -> 9.9v)
            vdd = r_vdd / 100
            
            for r_pbody in pbody_range:
                pbody = r_pbody / 100
                ocean_log = log_filename(vdd, pbody)

                vth = read_log_file(ocean_log, log)
                if not vth:
                    continue

                res.append((pbody, vth))
                log.println(f"vdd [{vdd:.3f}] pbody [{pbody:.3f}] \t=>\t {vth}")
        
            log.println(f"Vdd {vdd} result: \n{res}")
