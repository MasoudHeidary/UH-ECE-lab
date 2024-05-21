import subprocess

def update_netlist(file_name: str, pb: float):
    _netlist = \
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
    
    with open(file_name, "w") as f:
        f.write(_netlist)
        f.flush()
        f.close()

    return None



def update_ocean(file_name: str, log_file):
    script = \
    f"""
simulator( 'spectre )
design(	 "/home/mheidary/simulation/vth/spectre/schematic/netlist/netlist")
resultsDir( "/home/mheidary/simulation/vth/spectre/schematic" )
modelFile( 
    '("$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")
)
analysis('dc ?dev "/V2"  ?param "dc"  ?start "0.01"  
		?stop "0.8"  ?lin "1000"  )
desVar(	  "wireopt" 19	)
envOption(
	'analysisOrder  list("dc") 
)
save( 'i "/R167/PLUS" )
temp( 27 ) 
run()
selectResult( 'dc )

reportFile = outfile("{log_file}.txt")
ocnPrint(?output reportFile getData("/R167/PLUS"))
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



# 0.8 - 2.5, step(0.01)
for _vb in range(80, 500, 2):
    vb = _vb / 100
    update_netlist("/home/mheidary/simulation/vth/spectre/schematic/netlist/netlist", vb)
    update_ocean("./tmp_pmos_vth.ocn", f"./log/{vb}.txt")
    run_ocean_script("./tmp_pmos_vth.ocn")