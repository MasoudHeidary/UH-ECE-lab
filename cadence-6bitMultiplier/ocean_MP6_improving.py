import subprocess
import tool.NBTI_formula as NBTI 
import tool.vth_body_map as VTH


CADENCE_SERVER = True
NETLIST_DIR = "/home/mheidary/simulation/main_MP6/spectre/schematic/netlist/netlist" if CADENCE_SERVER else "./netlist"
RESULT_DIR = "/home/mheidary/simulation/main_MP6/spectre/schematic" if CADENCE_SERVER else "./result"



# normal alpha
normal_alpha = \
    [
        [
            [0.75, 0.625, 0.625], [0.75, 0.625, 0.625], [0.75, 0.625, 0.625], 
            [0.75, 0.625, 0.625], [0.75, 0.625, 0.625], [0.5, 0.75, 0.75],  # [1,...] => [0.5, ...]
        ], 
        [
            [0.625, 0.5625, 0.5625], [0.625, 0.5625, 0.5625], [0.625, 0.5625, 0.5625],
            [0.625, 0.5625, 0.5625], [0.79296875, 0.64453125, 0.64453125], [0.91796875, 0.75, 0.75],
        ],
        [
            [0.5625, 0.53125, 0.53125], [0.5625, 0.53125, 0.53125], [0.5625, 0.53125, 0.53125], 
            [0.66796875, 0.58203125, 0.58203125], [0.751953125, 0.646484375, 0.646484375], [0.8828125, 0.75, 0.75],
        ],
        [
            [0.53125, 0.515625, 0.515625], [0.53125, 0.515625, 0.515625], [0.59375, 0.541015625, 0.541015625],
            [0.646484375, 0.5810546875, 0.5810546875], [0.736328125, 0.646484375, 0.646484375], [0.8662109375, 0.75, 0.75],
        ],
        [
            [0.515625, 0.5078125, 0.5078125], [0.552734375, 0.5224609375, 0.5224609375], [0.5849609375, 0.54248046875, 0.54248046875],
            [0.6416015625, 0.5830078125, 0.5830078125], [0.7275390625, 0.6474609375, 0.6474609375], [0.85791015625, 0.75, 0.75],
        ],
    ]


# modifing XOR FA[4][5] T0
# improved_alpha = \
# [
#     [
#         [0.7421875, 0.61572265625, 0.61572265625], [0.73779296875, 0.61474609375, 0.61474609375], [0.72998046875, 0.607421875, 0.607421875], 
#         [0.7158203125, 0.59521484375, 0.59521484375], [0.6748046875, 0.568359375, 0.568359375], [0.5, 0.6708984375, 0.6708984375],              # [1,...] => [0.5,...]
#     ],
#     [
#         [0.6123046875, 0.5546875, 0.5546875], [0.6064453125, 0.54931640625, 0.54931640625], [0.59033203125, 0.53564453125, 0.53564453125], 
#         [0.55810546875, 0.51708984375, 0.51708984375], [0.7509765625, 0.58837890625, 0.58837890625], [0.88330078125, 0.66796875, 0.66796875], 
#     ],
#     [
#         [0.5478515625, 0.51904296875, 0.51904296875], [0.5322265625, 0.5126953125, 0.5126953125], [0.51025390625, 0.5029296875, 0.5029296875], 
#         [0.62744140625, 0.5419921875, 0.5419921875], [0.69580078125, 0.57373046875, 0.57373046875], [0.830078125, 0.654296875, 0.654296875], 
#     ],
#     [
#         [0.509765625, 0.5048828125, 0.5048828125], [0.4970703125, 0.49951171875, 0.49951171875], [0.56103515625, 0.50341796875, 0.50341796875], 
#         [0.58056640625, 0.51025390625, 0.51025390625], [0.666015625, 0.57080078125, 0.57080078125], [0.79931640625, 0.6748046875, 0.6748046875], 
#     ],
#     [
#         [0.49267578125, 0.501220703125, 0.501220703125], [0.5244140625, 0.504638671875, 0.504638671875], [0.5302734375, 0.508056640625, 0.508056640625], 
#         [0.59228515625, 0.548583984375, 0.548583984375], [0.69873046875, 0.617431640625, 0.617431640625], [0.7158203125, 0.678955078125, 0.678955078125], 
#     ]
# ]


# modifing XOR FA[4][5] T1
improved_alpha = \
[
    [
        [0.75, 0.625, 0.625], [0.75, 0.625, 0.625], [0.75, 0.625, 0.625],
        [0.75, 0.625, 0.625], [0.625, 0.5625, 0.5625], [0.5, 0.625, 0.625],
    ],
    [
        [0.625, 0.5625, 0.5625], [0.625, 0.5625, 0.5625], [0.625, 0.5625, 0.5625], 
        [0.54296875, 0.5234375, 0.5234375], [0.728515625, 0.591796875, 0.591796875], [0.876953125, 0.625, 0.625], 
    ],
    [
        [0.5625, 0.53125, 0.53125], [0.5625, 0.53125, 0.53125], [0.51171875, 0.5078125, 0.5078125], 
        [0.625, 0.55078125, 0.55078125], [0.6845703125, 0.5947265625, 0.5947265625], [0.82421875, 0.625, 0.625]
    ],
    [
        [0.53125, 0.515625, 0.515625], [0.5, 0.505859375, 0.505859375], [0.56640625, 0.521484375, 0.521484375], 
        [0.6015625, 0.54931640625, 0.54931640625], [0.6689453125, 0.5947265625, 0.5947265625], [0.79931640625, 0.625, 0.625]
    ],
    [
        [0.5, 0.503662109375, 0.503662109375], [0.53515625, 0.516845703125, 0.516845703125], [0.556640625, 0.540283203125, 0.540283203125], 
        [0.5966796875, 0.562255859375, 0.562255859375], [0.66015625, 0.598388671875, 0.598388671875], [0.786865234375, 0.5, 0.5]
    ]
]



def generate_body_voltage(alpha_lst, t_sec):
    # equation starts from 1s
    if t_sec <= 0:
        t_sec = 1

    body_voltage = []
    
    for layer in range(5):
        lay_v = []
        for fa_index in range(6):
            if (layer==4) and (fa_index==5):
                v0 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
                v1 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
                tmp_v = abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec)
                v2 = VTH.get_body_voltage(tmp_v * 1.2)
                v3 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec))
                v4 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec))
                v5 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec))
            else:
                v0 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
                v1 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][0], NBTI.Tclk, t_sec))
                v2 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec))
                v3 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][1], NBTI.Tclk, t_sec))
                v4 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec))
                v5 = VTH.get_body_voltage(abs(NBTI.Vth) + NBTI.delta_vth(NBTI.Vdef, NBTI.T, 1-alpha_lst[layer][fa_index][2], NBTI.Tclk, t_sec))
            # v0= v1= v2= v3= v4= v5= 2.5

            lay_v.append([v0, v1, v2, v3, v4, v5])
        body_voltage.append(lay_v)
    return body_voltage
            

def update_netlist(file_name: str, pfa):
    _netlist =\
    f"""
// Library name: prjage
// Cell name: FA_6pb
// View name: schematic
subckt FA_6pb A B C carry gnd pb\<0\> pb\<1\> pb\<2\> pb\<3\> pb\<4\> \
        pb\<5\> sum vdd
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
    N28 (nxx xx gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N27 (nC C gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    N26 (nB B gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
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
    N1 (nA A gnd gnd) nfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P32 (nC nxx sum pb\<3\>) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P39 (C nxx carry pb\<4\>) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P38 (A xx carry pb\<5\>) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P31 (C xx sum pb\<2\>) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P28 (nxx xx vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P27 (nC C vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P26 (nB B vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P1 (A B xx pb\<0\>) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n pd=328n \
        nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 ptwell=0 \
        ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P4 (nA nB xx pb\<1\>) pfet w=80n l=20n as=6.72f ad=6.72f ps=328n \
        pd=328n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
    P3 (nA A vdd vdd) pfet w=160n l=20n as=13.44f ad=13.44f ps=488n \
        pd=488n nf=1 par=(1) par_nf=(1) * (1) m=1 plorient=0 acv_opt=-1 \
        ptwell=0 ngcon=1 nscon=1 ndcon=1 p_la=0 p_wa=0 ulp=0 lle_pcpc=84n \
        tie_orient=0 swrfmhc_local=0 analog=0
ends FA_6pb
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
// Cell name: main_MP6
// View name: schematic
I85 (net4 net78 net77 net42 gnd pbFA44\<0\> pbFA44\<1\> pbFA44\<2\> \
        pbFA44\<3\> pbFA44\<4\> pbFA44\<5\> M\<9\> vdd) FA_6pb
I84 (net55 net69 net2 net16 gnd pbFA41\<0\> pbFA41\<1\> pbFA41\<2\> \
        pbFA41\<3\> pbFA41\<4\> pbFA41\<5\> M\<6\> vdd) FA_6pb
I83 (net58 net57 gnd net2 gnd pbFA40\<0\> pbFA40\<1\> pbFA40\<2\> \
        pbFA40\<3\> pbFA40\<4\> pbFA40\<5\> M\<5\> vdd) FA_6pb
I80 (net5 net43 net42 M\<11\> gnd pbFA45\<0\> pbFA45\<1\> pbFA45\<2\> \
        pbFA45\<3\> pbFA45\<4\> pbFA45\<5\> M\<10\> vdd) FA_6pb
I79 (net10 net17 net16 net1 gnd pbFA42\<0\> pbFA42\<1\> pbFA42\<2\> \
        pbFA42\<3\> pbFA42\<4\> pbFA42\<5\> M\<7\> vdd) FA_6pb
I74 (net3 net380 net1 net77 gnd pbFA43\<0\> pbFA43\<1\> pbFA43\<2\> \
        pbFA43\<3\> pbFA43\<4\> pbFA43\<5\> M\<8\> vdd) FA_6pb
I73 (net94 net288 net86 net87 gnd pbFA34\<0\> pbFA34\<1\> pbFA34\<2\> \
        pbFA34\<3\> pbFA34\<4\> pbFA34\<5\> net380 vdd) FA_6pb
I72 (net89 net278 net92 net88 gnd pbFA31\<0\> pbFA31\<1\> pbFA31\<2\> \
        pbFA31\<3\> pbFA31\<4\> pbFA31\<5\> net57 vdd) FA_6pb
I71 (net106 net310 gnd net92 gnd pbFA30\<0\> pbFA30\<1\> pbFA30\<2\> \
        pbFA30\<3\> pbFA30\<4\> pbFA30\<5\> M\<4\> vdd) FA_6pb
I68 (net95 net279 net87 net43 gnd pbFA35\<0\> pbFA35\<1\> pbFA35\<2\> \
        pbFA35\<3\> pbFA35\<4\> pbFA35\<5\> net78 vdd) FA_6pb
I67 (net90 net259 net88 net91 gnd pbFA32\<0\> pbFA32\<1\> pbFA32\<2\> \
        pbFA32\<3\> pbFA32\<4\> pbFA32\<5\> net69 vdd) FA_6pb
I62 (net93 net318 net91 net86 gnd pbFA33\<0\> pbFA33\<1\> pbFA33\<2\> \
        pbFA33\<3\> pbFA33\<4\> pbFA33\<5\> net17 vdd) FA_6pb
I61 (net104 net218 net250 net280 gnd pbFA24\<0\> pbFA24\<1\> pbFA24\<2\> \
        pbFA24\<3\> pbFA24\<4\> pbFA24\<5\> net318 vdd) FA_6pb
I60 (net293 net158 net102 net270 gnd pbFA21\<0\> pbFA21\<1\> pbFA21\<2\> \
        pbFA21\<3\> pbFA21\<4\> pbFA21\<5\> net310 vdd) FA_6pb
I59 (net262 net241 gnd net102 gnd pbFA20\<0\> pbFA20\<1\> pbFA20\<2\> \
        pbFA20\<3\> pbFA20\<4\> pbFA20\<5\> M\<3\> vdd) FA_6pb
I56 (net105 net209 net280 net279 gnd pbFA25\<0\> pbFA25\<1\> pbFA25\<2\> \
        pbFA25\<3\> pbFA25\<4\> pbFA25\<5\> net288 vdd) FA_6pb
I55 (net267 net148 net270 net251 gnd pbFA22\<0\> pbFA22\<1\> pbFA22\<2\> \
        pbFA22\<3\> pbFA22\<4\> pbFA22\<5\> net278 vdd) FA_6pb
I50 (net103 net249 net251 net250 gnd pbFA23\<0\> pbFA23\<1\> pbFA23\<2\> \
        pbFA23\<3\> pbFA23\<4\> pbFA23\<5\> net259 vdd) FA_6pb
I49 (net100 net128 net139 net210 gnd pbFA14\<0\> pbFA14\<1\> pbFA14\<2\> \
        pbFA14\<3\> pbFA14\<4\> pbFA14\<5\> net249 vdd) FA_6pb
I48 (net223 net117 net224 net149 gnd pbFA11\<0\> pbFA11\<1\> pbFA11\<2\> \
        pbFA11\<3\> pbFA11\<4\> pbFA11\<5\> net241 vdd) FA_6pb
I47 (net101 net199 gnd net224 gnd pbFA10\<0\> pbFA10\<1\> pbFA10\<2\> \
        pbFA10\<3\> pbFA10\<4\> pbFA10\<5\> M\<2\> vdd) FA_6pb
I43 (net208 net118 net210 net209 gnd pbFA15\<0\> pbFA15\<1\> pbFA15\<2\> \
        pbFA15\<3\> pbFA15\<4\> pbFA15\<5\> net218 vdd) FA_6pb
I42 (net151 net150 net149 net140 gnd pbFA12\<0\> pbFA12\<1\> pbFA12\<2\> \
        pbFA12\<3\> pbFA12\<4\> pbFA12\<5\> net158 vdd) FA_6pb
I37 (net99 net138 net140 net139 gnd pbFA13\<0\> pbFA13\<1\> pbFA13\<2\> \
        pbFA13\<3\> pbFA13\<4\> pbFA13\<5\> net148 vdd) FA_6pb
I36 (net131 net130 net129 net119 gnd pbFA04\<0\> pbFA04\<1\> pbFA04\<2\> \
        pbFA04\<3\> pbFA04\<4\> pbFA04\<5\> net138 vdd) FA_6pb
I35 (net189 net66 net167 net108 gnd pbFA01\<0\> pbFA01\<1\> pbFA01\<2\> \
        pbFA01\<3\> pbFA01\<4\> pbFA01\<5\> net199 vdd) FA_6pb
I34 (net163 net29 gnd net167 gnd pbFA00\<0\> pbFA00\<1\> pbFA00\<2\> \
        pbFA00\<3\> pbFA00\<4\> pbFA00\<5\> M\<1\> vdd) FA_6pb
I31 (net121 gnd net119 net118 gnd pbFA05\<0\> pbFA05\<1\> pbFA05\<2\> \
        pbFA05\<3\> pbFA05\<4\> pbFA05\<5\> net128 vdd) FA_6pb
I30 (net110 net15 net108 net107 gnd pbFA02\<0\> pbFA02\<1\> pbFA02\<2\> \
        pbFA02\<3\> pbFA02\<4\> pbFA02\<5\> net117 vdd) FA_6pb
I25 (net166 net21 net107 net129 gnd pbFA03\<0\> pbFA03\<1\> pbFA03\<2\> \
        pbFA03\<3\> pbFA03\<4\> pbFA03\<5\> net150 vdd) FA_6pb
V233 (pbFA45\<5\> gnd) vsource dc={pfa[4][5][5]} type=dc
V232 (pbFA45\<4\> gnd) vsource dc={pfa[4][5][4]} type=dc
V231 (pbFA45\<3\> gnd) vsource dc={pfa[4][5][3]} type=dc
V230 (pbFA45\<2\> gnd) vsource dc={pfa[4][5][2]} type=dc
V229 (pbFA45\<1\> gnd) vsource dc={pfa[4][5][1]} type=dc
V228 (pbFA45\<0\> gnd) vsource dc={pfa[4][5][0]} type=dc
V227 (pbFA44\<5\> gnd) vsource dc={pfa[4][4][5]} type=dc
V226 (pbFA44\<4\> gnd) vsource dc={pfa[4][4][4]} type=dc
V225 (pbFA44\<3\> gnd) vsource dc={pfa[4][4][3]} type=dc
V224 (pbFA44\<2\> gnd) vsource dc={pfa[4][4][2]} type=dc
V223 (pbFA44\<1\> gnd) vsource dc={pfa[4][4][1]} type=dc
V222 (pbFA44\<0\> gnd) vsource dc={pfa[4][4][0]} type=dc
V221 (pbFA43\<5\> gnd) vsource dc={pfa[4][3][5]} type=dc
V220 (pbFA43\<4\> gnd) vsource dc={pfa[4][3][4]} type=dc
V219 (pbFA43\<3\> gnd) vsource dc={pfa[4][3][3]} type=dc
V218 (pbFA43\<2\> gnd) vsource dc={pfa[4][3][2]} type=dc
V217 (pbFA43\<1\> gnd) vsource dc={pfa[4][3][1]} type=dc
V216 (pbFA43\<0\> gnd) vsource dc={pfa[4][3][0]} type=dc
V215 (pbFA42\<5\> gnd) vsource dc={pfa[4][2][5]} type=dc
V214 (pbFA42\<4\> gnd) vsource dc={pfa[4][2][4]} type=dc
V213 (pbFA42\<3\> gnd) vsource dc={pfa[4][2][3]} type=dc
V212 (pbFA42\<2\> gnd) vsource dc={pfa[4][2][2]} type=dc
V211 (pbFA42\<1\> gnd) vsource dc={pfa[4][2][1]} type=dc
V210 (pbFA42\<0\> gnd) vsource dc={pfa[4][2][0]} type=dc
V209 (pbFA41\<5\> gnd) vsource dc={pfa[4][1][5]} type=dc
V208 (pbFA41\<4\> gnd) vsource dc={pfa[4][1][4]} type=dc
V207 (pbFA41\<3\> gnd) vsource dc={pfa[4][1][3]} type=dc
V206 (pbFA41\<2\> gnd) vsource dc={pfa[4][1][2]} type=dc
V205 (pbFA41\<1\> gnd) vsource dc={pfa[4][1][1]} type=dc
V204 (pbFA41\<0\> gnd) vsource dc={pfa[4][1][0]} type=dc
V203 (pbFA40\<5\> gnd) vsource dc={pfa[4][0][5]} type=dc
V202 (pbFA40\<4\> gnd) vsource dc={pfa[4][0][4]} type=dc
V201 (pbFA40\<3\> gnd) vsource dc={pfa[4][0][3]} type=dc
V199 (pbFA40\<2\> gnd) vsource dc={pfa[4][0][2]} type=dc
V198 (pbFA40\<1\> gnd) vsource dc={pfa[4][0][1]} type=dc
V197 (pbFA40\<0\> gnd) vsource dc={pfa[4][0][0]} type=dc
V196 (pbFA35\<5\> gnd) vsource dc={pfa[3][5][5]} type=dc
V195 (pbFA35\<4\> gnd) vsource dc={pfa[3][5][4]} type=dc
V194 (pbFA35\<3\> gnd) vsource dc={pfa[3][5][3]} type=dc
V193 (pbFA35\<2\> gnd) vsource dc={pfa[3][5][2]} type=dc
V192 (pbFA35\<1\> gnd) vsource dc={pfa[3][5][1]} type=dc
V191 (pbFA35\<0\> gnd) vsource dc={pfa[3][5][0]} type=dc
V190 (pbFA34\<5\> gnd) vsource dc={pfa[3][4][5]} type=dc
V189 (pbFA34\<4\> gnd) vsource dc={pfa[3][4][4]} type=dc
V188 (pbFA34\<3\> gnd) vsource dc={pfa[3][4][3]} type=dc
V187 (pbFA34\<2\> gnd) vsource dc={pfa[3][4][2]} type=dc
V186 (pbFA34\<1\> gnd) vsource dc={pfa[3][4][1]} type=dc
V185 (pbFA34\<0\> gnd) vsource dc={pfa[3][4][0]} type=dc
V184 (pbFA33\<5\> gnd) vsource dc={pfa[3][3][5]} type=dc
V183 (pbFA33\<4\> gnd) vsource dc={pfa[3][3][4]} type=dc
V182 (pbFA33\<3\> gnd) vsource dc={pfa[3][3][3]} type=dc
V181 (pbFA33\<2\> gnd) vsource dc={pfa[3][3][2]} type=dc
V180 (pbFA33\<1\> gnd) vsource dc={pfa[3][3][1]} type=dc
V179 (pbFA33\<0\> gnd) vsource dc={pfa[3][3][0]} type=dc
V178 (pbFA32\<5\> gnd) vsource dc={pfa[3][2][5]} type=dc
V177 (pbFA32\<4\> gnd) vsource dc={pfa[3][2][4]} type=dc
V176 (pbFA32\<3\> gnd) vsource dc={pfa[3][2][3]} type=dc
V175 (pbFA32\<2\> gnd) vsource dc={pfa[3][2][2]} type=dc
V174 (pbFA32\<1\> gnd) vsource dc={pfa[3][2][1]} type=dc
V173 (pbFA32\<0\> gnd) vsource dc={pfa[3][2][0]} type=dc
V172 (pbFA31\<5\> gnd) vsource dc={pfa[3][1][5]} type=dc
V171 (pbFA31\<4\> gnd) vsource dc={pfa[3][1][4]} type=dc
V170 (pbFA31\<3\> gnd) vsource dc={pfa[3][1][3]} type=dc
V169 (pbFA31\<2\> gnd) vsource dc={pfa[3][1][2]} type=dc
V168 (pbFA31\<1\> gnd) vsource dc={pfa[3][1][1]} type=dc
V167 (pbFA31\<0\> gnd) vsource dc={pfa[3][1][0]} type=dc
V166 (pbFA30\<5\> gnd) vsource dc={pfa[3][0][5]} type=dc
V165 (pbFA30\<4\> gnd) vsource dc={pfa[3][0][4]} type=dc
V164 (pbFA30\<3\> gnd) vsource dc={pfa[3][0][3]} type=dc
V163 (pbFA30\<2\> gnd) vsource dc={pfa[3][0][2]} type=dc
V162 (pbFA30\<1\> gnd) vsource dc={pfa[3][0][1]} type=dc
V161 (pbFA30\<0\> gnd) vsource dc={pfa[3][0][0]} type=dc
V160 (pbFA25\<5\> gnd) vsource dc={pfa[2][5][5]} type=dc
V159 (pbFA25\<4\> gnd) vsource dc={pfa[2][5][4]} type=dc
V158 (pbFA25\<3\> gnd) vsource dc={pfa[2][5][3]} type=dc
V157 (pbFA25\<2\> gnd) vsource dc={pfa[2][5][2]} type=dc
V156 (pbFA25\<1\> gnd) vsource dc={pfa[2][5][1]} type=dc
V155 (pbFA25\<0\> gnd) vsource dc={pfa[2][5][0]} type=dc
V154 (pbFA24\<5\> gnd) vsource dc={pfa[2][4][5]} type=dc
V153 (pbFA24\<4\> gnd) vsource dc={pfa[2][4][4]} type=dc
V152 (pbFA24\<3\> gnd) vsource dc={pfa[2][4][3]} type=dc
V151 (pbFA24\<2\> gnd) vsource dc={pfa[2][4][2]} type=dc
V150 (pbFA24\<1\> gnd) vsource dc={pfa[2][4][1]} type=dc
V149 (pbFA24\<0\> gnd) vsource dc={pfa[2][4][0]} type=dc
V148 (pbFA23\<5\> gnd) vsource dc={pfa[2][3][5]} type=dc
V147 (pbFA23\<4\> gnd) vsource dc={pfa[2][3][4]} type=dc
V146 (pbFA23\<3\> gnd) vsource dc={pfa[2][3][3]} type=dc
V145 (pbFA23\<2\> gnd) vsource dc={pfa[2][3][2]} type=dc
V144 (pbFA23\<1\> gnd) vsource dc={pfa[2][3][1]} type=dc
V143 (pbFA23\<0\> gnd) vsource dc={pfa[2][3][0]} type=dc
V142 (pbFA22\<5\> gnd) vsource dc={pfa[2][2][5]} type=dc
V141 (pbFA22\<4\> gnd) vsource dc={pfa[2][2][4]} type=dc
V140 (pbFA22\<3\> gnd) vsource dc={pfa[2][2][3]} type=dc
V139 (pbFA22\<2\> gnd) vsource dc={pfa[2][2][2]} type=dc
V138 (pbFA22\<1\> gnd) vsource dc={pfa[2][2][1]} type=dc
V137 (pbFA22\<0\> gnd) vsource dc={pfa[2][2][0]} type=dc
V136 (pbFA21\<5\> gnd) vsource dc={pfa[2][1][5]} type=dc
V135 (pbFA21\<4\> gnd) vsource dc={pfa[2][1][4]} type=dc
V134 (pbFA21\<3\> gnd) vsource dc={pfa[2][1][3]} type=dc
V133 (pbFA21\<2\> gnd) vsource dc={pfa[2][1][2]} type=dc
V132 (pbFA21\<1\> gnd) vsource dc={pfa[2][1][1]} type=dc
V131 (pbFA21\<0\> gnd) vsource dc={pfa[2][1][0]} type=dc
V130 (pbFA20\<5\> gnd) vsource dc={pfa[2][0][5]} type=dc
V129 (pbFA20\<4\> gnd) vsource dc={pfa[2][0][4]} type=dc
V128 (pbFA20\<3\> gnd) vsource dc={pfa[2][0][3]} type=dc
V127 (pbFA20\<2\> gnd) vsource dc={pfa[2][0][2]} type=dc
V126 (pbFA20\<1\> gnd) vsource dc={pfa[2][0][1]} type=dc
V125 (pbFA20\<0\> gnd) vsource dc={pfa[2][0][0]} type=dc
V124 (pbFA15\<5\> gnd) vsource dc={pfa[1][5][5]} type=dc
V123 (pbFA15\<4\> gnd) vsource dc={pfa[1][5][4]} type=dc
V122 (pbFA15\<3\> gnd) vsource dc={pfa[1][5][3]} type=dc
V121 (pbFA15\<2\> gnd) vsource dc={pfa[1][5][2]} type=dc
V120 (pbFA15\<1\> gnd) vsource dc={pfa[1][5][1]} type=dc
V119 (pbFA15\<0\> gnd) vsource dc={pfa[1][5][0]} type=dc
V118 (pbFA14\<5\> gnd) vsource dc={pfa[1][4][5]} type=dc
V117 (pbFA14\<4\> gnd) vsource dc={pfa[1][4][4]} type=dc
V116 (pbFA14\<3\> gnd) vsource dc={pfa[1][4][3]} type=dc
V115 (pbFA14\<2\> gnd) vsource dc={pfa[1][4][2]} type=dc
V114 (pbFA14\<1\> gnd) vsource dc={pfa[1][4][1]} type=dc
V113 (pbFA14\<0\> gnd) vsource dc={pfa[1][4][0]} type=dc
V112 (pbFA13\<5\> gnd) vsource dc={pfa[1][3][5]} type=dc
V111 (pbFA13\<4\> gnd) vsource dc={pfa[1][3][4]} type=dc
V110 (pbFA13\<3\> gnd) vsource dc={pfa[1][3][3]} type=dc
V109 (pbFA13\<2\> gnd) vsource dc={pfa[1][3][2]} type=dc
V108 (pbFA13\<1\> gnd) vsource dc={pfa[1][3][1]} type=dc
V99 (pbFA13\<0\> gnd) vsource dc={pfa[1][3][0]} type=dc
V98 (pbFA12\<5\> gnd) vsource dc={pfa[1][2][5]} type=dc
V97 (pbFA12\<4\> gnd) vsource dc={pfa[1][2][4]} type=dc
V96 (pbFA12\<3\> gnd) vsource dc={pfa[1][2][3]} type=dc
V95 (pbFA12\<2\> gnd) vsource dc={pfa[1][2][2]} type=dc
V94 (pbFA12\<1\> gnd) vsource dc={pfa[1][2][1]} type=dc
V93 (pbFA12\<0\> gnd) vsource dc={pfa[1][2][0]} type=dc
V92 (pbFA11\<5\> gnd) vsource dc={pfa[1][1][5]} type=dc
V91 (pbFA11\<4\> gnd) vsource dc={pfa[1][1][4]} type=dc
V90 (pbFA11\<3\> gnd) vsource dc={pfa[1][1][3]} type=dc
V89 (pbFA11\<2\> gnd) vsource dc={pfa[1][1][2]} type=dc
V88 (pbFA11\<1\> gnd) vsource dc={pfa[1][1][1]} type=dc
V87 (pbFA11\<0\> gnd) vsource dc={pfa[1][1][0]} type=dc
V86 (pbFA10\<5\> gnd) vsource dc={pfa[1][0][5]} type=dc
V85 (pbFA10\<4\> gnd) vsource dc={pfa[1][0][4]} type=dc
V84 (pbFA10\<3\> gnd) vsource dc={pfa[1][0][3]} type=dc
V83 (pbFA10\<2\> gnd) vsource dc={pfa[1][0][2]} type=dc
V82 (pbFA10\<1\> gnd) vsource dc={pfa[1][0][1]} type=dc
V81 (pbFA10\<0\> gnd) vsource dc={pfa[1][0][0]} type=dc
V80 (pbFA05\<5\> gnd) vsource dc={pfa[0][5][5]} type=dc
V79 (pbFA05\<4\> gnd) vsource dc={pfa[0][5][4]} type=dc
V78 (pbFA05\<3\> gnd) vsource dc={pfa[0][5][3]} type=dc
V77 (pbFA05\<2\> gnd) vsource dc={pfa[0][5][2]} type=dc
V76 (pbFA05\<1\> gnd) vsource dc={pfa[0][5][1]} type=dc
V75 (pbFA05\<0\> gnd) vsource dc={pfa[0][5][0]} type=dc
V74 (pbFA04\<5\> gnd) vsource dc={pfa[0][4][5]} type=dc
V73 (pbFA04\<4\> gnd) vsource dc={pfa[0][4][4]} type=dc
V72 (pbFA04\<3\> gnd) vsource dc={pfa[0][4][3]} type=dc
V71 (pbFA04\<2\> gnd) vsource dc={pfa[0][4][2]} type=dc
V70 (pbFA04\<1\> gnd) vsource dc={pfa[0][4][1]} type=dc
V69 (pbFA04\<0\> gnd) vsource dc={pfa[0][4][0]} type=dc
V68 (pbFA03\<5\> gnd) vsource dc={pfa[0][3][5]} type=dc
V67 (pbFA03\<4\> gnd) vsource dc={pfa[0][3][4]} type=dc
V66 (pbFA03\<3\> gnd) vsource dc={pfa[0][3][3]} type=dc
V65 (pbFA03\<2\> gnd) vsource dc={pfa[0][3][2]} type=dc
V64 (pbFA03\<1\> gnd) vsource dc={pfa[0][3][1]} type=dc
V63 (pbFA03\<0\> gnd) vsource dc={pfa[0][3][0]} type=dc
V62 (pbFA02\<5\> gnd) vsource dc={pfa[0][2][5]} type=dc
V61 (pbFA02\<4\> gnd) vsource dc={pfa[0][2][4]} type=dc
V60 (pbFA02\<3\> gnd) vsource dc={pfa[0][2][3]} type=dc
V59 (pbFA02\<2\> gnd) vsource dc={pfa[0][2][2]} type=dc
V58 (pbFA02\<1\> gnd) vsource dc={pfa[0][2][1]} type=dc
V57 (pbFA02\<0\> gnd) vsource dc={pfa[0][2][0]} type=dc
V56 (pbFA01\<5\> gnd) vsource dc={pfa[0][1][5]} type=dc
V55 (pbFA01\<4\> gnd) vsource dc={pfa[0][1][4]} type=dc
V54 (pbFA01\<3\> gnd) vsource dc={pfa[0][1][3]} type=dc
V53 (pbFA01\<2\> gnd) vsource dc={pfa[0][1][2]} type=dc
V52 (pbFA01\<1\> gnd) vsource dc={pfa[0][1][1]} type=dc
V51 (pbFA01\<0\> gnd) vsource dc={pfa[0][1][0]} type=dc
V50 (pbFA00\<5\> gnd) vsource dc={pfa[0][0][5]} type=dc
V49 (pbFA00\<4\> gnd) vsource dc={pfa[0][0][4]} type=dc
V48 (pbFA00\<3\> gnd) vsource dc={pfa[0][0][3]} type=dc
V47 (pbFA00\<2\> gnd) vsource dc={pfa[0][0][2]} type=dc
V46 (pbFA00\<1\> gnd) vsource dc={pfa[0][0][1]} type=dc
V45 (pbFA00\<0\> gnd) vsource dc={pfa[0][0][0]} type=dc
V1 (gnd 0) vsource dc=0 type=dc
V0 (vdd 0) vsource dc=800.0m type=dc
V5 (net9 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V4 (net8 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V3 (net7 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V2 (net6 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V100 (net37 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V200 (step gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V105 (net36 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V107 (net35 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V103 (net34 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V101 (net33 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V106 (net32 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V104 (net31 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
V102 (net30 gnd) vsource type=pulse val0=0 val1=800m period=10n delay=100p \
        rise=1p fall=1p
I82 (gnd vdd net55 vdd B\<1\> A\<5\>) and_for_trojan
I81 (gnd vdd net4 vdd B\<4\> A\<5\>) and_for_trojan
I78 (gnd vdd net5 vdd B\<5\> A\<5\>) and_for_trojan
I77 (gnd vdd net10 vdd B\<2\> A\<5\>) and_for_trojan
I76 (gnd vdd net3 vdd B\<3\> A\<5\>) and_for_trojan
I75 (gnd vdd net58 vdd B\<0\> A\<5\>) and_for_trojan
I70 (gnd vdd net89 vdd B\<1\> A\<4\>) and_for_trojan
I69 (gnd vdd net94 vdd B\<4\> A\<4\>) and_for_trojan
I66 (gnd vdd net95 vdd B\<5\> A\<4\>) and_for_trojan
I65 (gnd vdd net90 vdd B\<2\> A\<4\>) and_for_trojan
I64 (gnd vdd net93 vdd B\<3\> A\<4\>) and_for_trojan
I63 (gnd vdd net106 vdd B\<0\> A\<4\>) and_for_trojan
I58 (gnd vdd net293 vdd B\<1\> A\<3\>) and_for_trojan
I57 (gnd vdd net104 vdd B\<4\> A\<3\>) and_for_trojan
I54 (gnd vdd net105 vdd B\<5\> A\<3\>) and_for_trojan
I53 (gnd vdd net267 vdd B\<2\> A\<3\>) and_for_trojan
I52 (gnd vdd net103 vdd B\<3\> A\<3\>) and_for_trojan
I51 (gnd vdd net262 vdd B\<0\> A\<3\>) and_for_trojan
I46 (gnd vdd net223 vdd B\<1\> A\<2\>) and_for_trojan
I45 (gnd vdd net100 vdd B\<4\> A\<2\>) and_for_trojan
I41 (gnd vdd net208 vdd B\<5\> A\<2\>) and_for_trojan
I40 (gnd vdd net151 vdd B\<2\> A\<2\>) and_for_trojan
I39 (gnd vdd net99 vdd B\<3\> A\<2\>) and_for_trojan
I38 (gnd vdd net101 vdd B\<0\> A\<2\>) and_for_trojan
I33 (gnd vdd net189 vdd B\<1\> A\<1\>) and_for_trojan
I32 (gnd vdd net131 vdd B\<4\> A\<1\>) and_for_trojan
I29 (gnd vdd net121 vdd B\<5\> A\<1\>) and_for_trojan
I28 (gnd vdd net110 vdd B\<2\> A\<1\>) and_for_trojan
I27 (gnd vdd net166 vdd B\<3\> A\<1\>) and_for_trojan
I26 (gnd vdd net163 vdd B\<0\> A\<1\>) and_for_trojan
I12 (gnd vdd net130 vdd B\<5\> A\<0\>) and_for_trojan
I11 (gnd vdd net21 vdd B\<4\> A\<0\>) and_for_trojan
I10 (gnd vdd net15 vdd B\<3\> A\<0\>) and_for_trojan
I9 (gnd vdd net66 vdd B\<2\> A\<0\>) and_for_trojan
I8 (gnd vdd net29 vdd B\<1\> A\<0\>) and_for_trojan
I44 (gnd vdd M\<0\> vdd B\<0\> A\<0\>) and_for_trojan
R3 (net8 B\<5\>) resistor r=0
R2 (net9 B\<4\>) resistor r=0
R1 (net6 B\<2\>) resistor r=0
R0 (net7 B\<3\>) resistor r=0
R23 (step gnd) resistor r=10kOhms
R15 (net37 A\<0\>) resistor r=0
R16 (net33 A\<1\>) resistor r=0
R17 (net30 A\<2\>) resistor r=0
R18 (net34 A\<3\>) resistor r=0
R19 (net31 A\<4\>) resistor r=0
R20 (net36 A\<5\>) resistor r=0
R21 (net32 B\<0\>) resistor r=0
R22 (net35 B\<1\>) resistor r=0
C12 (M\<11\> gnd) capacitor c=1f
C11 (M\<10\> gnd) capacitor c=1f
C10 (M\<9\> gnd) capacitor c=1f
C9 (M\<8\> gnd) capacitor c=1f
C8 (M\<7\> gnd) capacitor c=1f
C7 (M\<6\> gnd) capacitor c=1f
C6 (M\<5\> gnd) capacitor c=1f
C5 (M\<4\> gnd) capacitor c=1f
C4 (M\<3\> gnd) capacitor c=1f
C3 (M\<2\> gnd) capacitor c=1f
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
temp( 27 ) 
run()
selectResult( 'tran )


/* getData("/M<10>") getData("/M<11>") */
/*
plot(getData("/step"))
hardCopyOptions(?hcOutputFile "{log_file}.png" ?hcResolution 500 ?hcFontSize 18 ?hcOutputFormat "png" ?hcImageWidth 1920 ?hcImageHeight 1080)
hardCopy()
*/


/*calculate the output delays*/
out_0_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<0>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_1_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<1>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_2_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<2>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_3_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<3>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_4_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<4>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_5_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<5>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_6_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<6>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_7_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<7>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_8_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<8>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_9_delay     = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<9>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_10_delay    = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<10>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)
out_11_delay    = delay(?wf1 VT("/step"), ?value1 0.4, ?edge1 "rising", ?nth1 1, ?td1 0.0, ?tol1 nil, ?wf2 VT("/M<11>"), ?value2 0.4, ?edge2 "rising", ?nth2 1, ?tol2 nil,  ?td2 nil , ?stop nil, ?multiple nil)

reportFile = outfile("{log_file}")
ocnPrint(?output reportFile "0" out_0_delay)
ocnPrint(?output reportFile "1" out_1_delay)
ocnPrint(?output reportFile "2" out_2_delay)
ocnPrint(?output reportFile "3" out_3_delay)
ocnPrint(?output reportFile "4" out_4_delay)
ocnPrint(?output reportFile "5" out_5_delay)
ocnPrint(?output reportFile "6" out_6_delay)
ocnPrint(?output reportFile "7" out_7_delay)
ocnPrint(?output reportFile "8" out_8_delay)
ocnPrint(?output reportFile "9" out_9_delay)
ocnPrint(?output reportFile "10" out_10_delay)
ocnPrint(?output reportFile "11" out_11_delay)
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
    import tool.log as log
    log = log.Log("terminal_log.txt")
    log.println("### log file created")

    if True:
        for t_week in range(0, 200, 20):
            t_sec = t_week * (30/2) * 24 * 60 * 60


            # normal aging
            log_name = f"./log/Normal-{t_week}.txt"
            body_voltage = generate_body_voltage(normal_alpha, t_sec)
            # log.println(str(body_voltage))
            update_netlist(NETLIST_DIR, body_voltage)
            update_ocean("./tmp_main.ocn", log_name)
            if CADENCE_SERVER:
                run_ocean_script("./tmp_main.ocn")
            log.println(log_name)


            # improved aging
            log_name = f"./log/improved-FA[4][5]-T0-{t_week}.txt"
            update_netlist(NETLIST_DIR, generate_body_voltage(improved_alpha, t_sec))
            update_ocean("./tmp_main.ocn", log_name)
            if CADENCE_SERVER:
                run_ocean_script("./tmp_main.ocn")
            log.println(log_name)



    import matplotlib.pyplot as plt
    # plot Vth
    if False:
        vth = []
        time = []
        for i in range(6):
            for t_week in range(0, 200, 199):
                t_sec = t_week * (30/2) * 24 * 60 * 60

                time += [t_week]
                vth += [generate_body_voltage(normal_alpha, t_sec)[4][5][i]]
        
            plt.plot(time, vth)
        plt.show()

