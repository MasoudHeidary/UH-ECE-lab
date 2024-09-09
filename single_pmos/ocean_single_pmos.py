import subprocess
from tool.log import Log

CADENCE_SERVER = True
NETLIST_DIR = "/home/mheidary/simulation/single_pmos/spectre/schematic/netlist/netlist" if CADENCE_SERVER else "./netlist"
RESULT_DIR = "/home/mheidary/simulation/single_pmos/spectre/schematic" if CADENCE_SERVER else "./result"



def update_netlist(file_name: str, pb):
    _netlist =\
    f"""
// Library name: prjage
// Cell name: single_pmos
// View name: schematic
P0 (out gnd vdd pb) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n pd=488n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
R0 (out gnd) resistor r=10
V2 (pb gnd) vsource dc={pb} type=dc
V1 (gnd 0) vsource dc=0 type=dc
V0 (vdd 0) vsource dc=800.0m type=dc
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
design(	 "{NETLIST_DIR}")
resultsDir( "{RESULT_DIR}" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('tran ?stop "5n"  ?errpreset "conservative"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("tran") 
)
save( 'i "/R0/PLUS" )
temp( 27 ) 
run()
selectResult( 'tran )
plot(getData("/R0/PLUS") )


x = ymax(IT("/R0/PLUS"))

reportFile = outfile("{log_file}")
ocnPrint(?output reportFile x)
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



if __name__ == "__main__":
    log = Log("terminal-log.txt")

    if True:
        for body_voltage in range(800, 5000, 10):
            pb = body_voltage / 1000

            log_name = f"./log/pmos-pb-{pb:.2f}.txt"
            update_netlist(NETLIST_DIR, pb)
            update_ocean("./tmp_main.ocn", log_name)
            if CADENCE_SERVER:
                run_ocean_script("./tmp_main.ocn")
            log.println(f"pb: {pb} DONE")
            
