

"""
"""

from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.bin_func import signed_b, reverse_signed_b
from msimulator.Multiplier import Wallace_rew
from alpha_shrinked import AlphaSampled


from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

from propagation_delay import wallace_multiplier_error_rate_sample

import matplotlib.pyplot as plt
from datetime import datetime


BIT_LEN = 16
TEMP = 273.15 + 80
ALPHA_VERIFICATION = False
ALPHA_SAMPLE_COUNT = 500_000
RND_SEED = 7
PROCESS_COUNT = 10
log = Log(f"{__file__}.{BIT_LEN}.log", terminal=True)


no_tamper = \
[]

tamper_critical_path = \
[(5, 0, 'A', 'C', 'B', 0.559487579332143), (6, 0, 'A', 'C', 'B', 0.559487579332143), (7, 0, 'A', 'C', 'B', 0.559487579332143), (8, 0, 'A', 'C', 'B', 0.559487579332143), (9, 0, 'A', 'C', 'B', 0.559487579332143), (10, 0, 'A', 'C', 'B', 0.559487579332143), (11, 0, 'A', 'C', 'B', 0.559487579332143), (12, 0, 'A', 'C', 'B', 0.559487579332143), (13, 0, 'A', 'C', 'B', 0.559487579332143), (14, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.552478684071531), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (14, 1, 'A', 'C', 'B', 0.27974378966607144), (14, 2, 'A', 'C', 'B', 0.16901833547121398), (14, 3, 'A', 'C', 'B', 0.13654350796884568), (14, 4, 'A', 'C', 'B', 0.12157415092599572), (14, 5, 'A', 'C', 'B', 0.1113835861062904), (14, 6, 'A', 'C', 'B', 0.10663832926716804), (14, 7, 'A', 'C', 'B', 0.10257216395449248), (14, 8, 'A', 'C', 'B', 0.10257216395449248), (14, 9, 'C', 'A', 'B', 0.0979107455754254), (14, 10, 'C', 'A', 'B', 0.09673700713465283), (14, 11, 'C', 'A', 'B', 0.09438114640710249), (14, 12, 'C', 'A', 'B', 0.08961912187596943), (14, 13, 'C', 'A', 'B', 0.07844764707361834), (0, 0, 'C', 'A', 'B', 0.06218895051694939), (14, 14, 'C', 'A', 'B', 0.05194212175545043), (14, 15, 'A', 'B', 'C', 0)]

tamper_full_circuit =\
[(5, 0, 'A', 'C', 'B', 0.559487579332143), (6, 0, 'A', 'C', 'B', 0.559487579332143), (7, 0, 'A', 'C', 'B', 0.559487579332143), (8, 0, 'A', 'C', 'B', 0.559487579332143), (9, 0, 'A', 'C', 'B', 0.559487579332143), (10, 0, 'A', 'C', 'B', 0.559487579332143), (11, 0, 'A', 'C', 'B', 0.559487579332143), (12, 0, 'A', 'C', 'B', 0.559487579332143), (13, 0, 'A', 'C', 'B', 0.559487579332143), (14, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.552478684071531), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (14, 1, 'A', 'C', 'B', 0.27974378966607144), (1, 1, 'A', 'C', 'B', 0.23806349924964598), (1, 2, 'A', 'C', 'B', 0.23806349924964598), (1, 3, 'A', 'C', 'B', 0.23806349924964598), (1, 4, 'A', 'C', 'B', 0.23806349924964598), (1, 5, 'A', 'C', 'B', 0.23806349924964598), (1, 6, 'A', 'C', 'B', 0.23806349924964598), (1, 7, 'A', 'C', 'B', 0.23806349924964598), (1, 8, 'A', 'C', 'B', 0.23806349924964598), (1, 9, 'A', 'C', 'B', 0.23806349924964598), (1, 10, 'A', 'C', 'B', 0.23806349924964598), (1, 11, 'A', 'C', 'B', 0.23806349924964598), (1, 12, 'A', 'C', 'B', 0.23806349924964598), (1, 13, 'A', 'C', 'B', 0.23806349924964598), (2, 2, 'A', 'C', 'B', 0.22623808445886484), (2, 3, 'A', 'C', 'B', 0.22623808445886484), (2, 4, 'A', 'C', 'B', 0.22623808445886484), (2, 6, 'A', 'C', 'B', 0.22623808445886484), (2, 9, 'A', 'C', 'B', 0.22623808445886484), (2, 12, 'A', 'C', 'B', 0.22623808445886484), (2, 1, 'A', 'C', 'B', 0.22468707294784407), (2, 5, 'A', 'C', 'B', 0.22468707294784407), (2, 11, 'A', 'C', 'B', 0.22468707294784407), (2, 7, 'A', 'C', 'B', 0.22441040602966233), (2, 8, 'A', 'C', 'B', 0.22285939451864156), (2, 10, 'A', 'C', 'B', 0.22285939451864156), (3, 1, 'A', 'C', 'B', 0.21645413617042697), (3, 2, 'A', 'C', 'B', 0.21645413617042697), (3, 3, 'A', 'C', 'B', 0.21645413617042697), (3, 4, 'A', 'C', 'B', 0.21645413617042697), (3, 5, 'A', 'C', 'B', 0.21645413617042697), (3, 6, 'A', 'C', 'B', 0.21645413617042697), (3, 7, 'A', 'C', 'B', 0.21645413617042697), (3, 8, 'A', 'C', 'B', 0.21645413617042697), (3, 9, 'A', 'C', 'B', 0.21645413617042697), (3, 10, 'A', 'C', 'B', 0.21645413617042697), (3, 11, 'A', 'C', 'B', 0.21645413617042697), (4, 1, 'A', 'C', 'B', 0.2103842316624328), (4, 5, 'A', 'C', 'B', 0.2103842316624328), (4, 10, 'A', 'C', 'B', 0.2103842316624328), (4, 2, 'A', 'C', 'B', 0.20976382705802482), (4, 4, 'A', 'C', 'B', 0.20976382705802482), (4, 9, 'A', 'C', 'B', 0.20976382705802482), (5, 2, 'A', 'C', 'B', 0.20916438206863014), (4, 8, 'A', 'C', 'B', 0.20859008861725226), (4, 3, 'A', 'C', 'B', 0.20805352247289927), (4, 6, 'A', 'C', 'B', 0.20805352247289927), (4, 7, 'A', 'C', 'B', 0.20805352247289927), (5, 9, 'A', 'C', 'B', 0.20741635017648002), (6, 3, 'A', 'C', 'B', 0.20739539056146644), (5, 3, 'A', 'C', 'B', 0.2068336728790966), (5, 4, 'A', 'C', 'B', 0.2068336728790966), (5, 1, 'A', 'C', 'B', 0.20508564098694648), (5, 5, 'A', 'C', 'B', 0.20508564098694648), (5, 6, 'A', 'C', 'B', 0.20508564098694648), (5, 7, 'A', 'C', 'B', 0.20508564098694648), (5, 8, 'A', 'C', 'B', 0.20508564098694648), (6, 1, 'A', 'C', 'B', 0.20220998180705446), (6, 2, 'A', 'C', 'B', 0.20220998180705446), (6, 4, 'A', 'C', 'B', 0.20220998180705446), (6, 5, 'A', 'C', 'B', 0.20220998180705446), (6, 6, 'A', 'C', 'B', 0.20220998180705446), (6, 7, 'A', 'C', 'B', 0.20220998180705446), (6, 8, 'A', 'C', 'B', 0.20220998180705446), (7, 2, 'A', 'C', 'B', 0.20220159796104858), (7, 4, 'A', 'C', 'B', 0.20220159796104858), (7, 6, 'A', 'C', 'B', 0.20220159796104858), (7, 8, 'A', 'C', 'B', 0.20220159796104858), (8, 3, 'A', 'C', 'B', 0.20220159796104858), (8, 9, 'A', 'C', 'B', 0.20220159796104858), (5, 10, 'A', 'C', 'B', 0.20101528375126826), (7, 1, 'A', 'C', 'B', 0.20048290952991782), (7, 3, 'A', 'C', 'B', 0.20048290952991782), (7, 5, 'A', 'C', 'B', 0.20048290952991782), (7, 7, 'A', 'C', 'B', 0.20048290952991782), (8, 1, 'A', 'C', 'B', 0.20048290952991782), (8, 2, 'A', 'C', 'B', 0.20048290952991782), (8, 4, 'A', 'C', 'B', 0.20048290952991782), (8, 5, 'A', 'C', 'B', 0.20048290952991782), (8, 6, 'A', 'C', 'B', 0.20048290952991782), (8, 8, 'A', 'C', 'B', 0.20048290952991782), (9, 1, 'A', 'C', 'B', 0.20048290952991782), (9, 2, 'A', 'C', 'B', 0.20048290952991782), (9, 3, 'A', 'C', 'B', 0.20048290952991782), (9, 5, 'A', 'C', 'B', 0.20048290952991782), (9, 6, 'A', 'C', 'B', 0.20048290952991782), (9, 8, 'A', 'C', 'B', 0.20048290952991782), (10, 1, 'A', 'C', 'B', 0.20048290952991782), (10, 2, 'A', 'C', 'B', 0.20048290952991782), (10, 3, 'A', 'C', 'B', 0.20048290952991782), (10, 4, 'A', 'C', 'B', 0.20048290952991782), (10, 5, 'A', 'C', 'B', 0.20048290952991782), (10, 6, 'A', 'C', 'B', 0.20048290952991782), (10, 7, 'A', 'C', 'B', 0.20048290952991782), (10, 8, 'A', 'C', 'B', 0.20048290952991782), (11, 1, 'A', 'C', 'B', 0.20048290952991782), (11, 2, 'A', 'C', 'B', 0.20048290952991782), (11, 3, 'A', 'C', 'B', 0.20048290952991782), (11, 4, 'A', 'C', 'B', 0.20048290952991782), (11, 5, 'A', 'C', 'B', 0.20048290952991782), (11, 6, 'A', 'C', 'B', 0.20048290952991782), (11, 7, 'A', 'C', 'B', 0.20048290952991782), (11, 8, 'A', 'C', 'B', 0.20048290952991782), (12, 2, 'A', 'C', 'B', 0.20048290952991782), (12, 3, 'A', 'C', 'B', 0.20048290952991782), (12, 4, 'A', 'C', 'B', 0.20048290952991782), (12, 5, 'A', 'C', 'B', 0.20048290952991782), (12, 6, 'A', 'C', 'B', 0.20048290952991782), (12, 7, 'A', 'C', 'B', 0.20048290952991782), (12, 8, 'A', 'C', 'B', 0.20048290952991782), (13, 1, 'A', 'C', 'B', 0.20048290952991782), (13, 2, 'A', 'C', 'B', 0.20048290952991782), (13, 3, 'A', 'C', 'B', 0.20048290952991782), (13, 4, 'A', 'C', 'B', 0.20048290952991782), (13, 5, 'A', 'C', 'B', 0.20048290952991782), (13, 6, 'A', 'C', 'B', 0.20048290952991782), (13, 7, 'A', 'C', 'B', 0.20048290952991782), (13, 8, 'A', 'C', 'B', 0.20048290952991782), (4, 11, 'A', 'C', 'B', 0.20044518222289298), (6, 9, 'A', 'C', 'B', 0.19987927261752092), (9, 9, 'A', 'C', 'B', 0.19987088877151504), (13, 9, 'A', 'C', 'B', 0.19987088877151504), (8, 7, 'A', 'C', 'B', 0.19876422109878705), (9, 4, 'A', 'C', 'B', 0.19876422109878705), (9, 7, 'A', 'C', 'B', 0.19876422109878705), (12, 1, 'A', 'C', 'B', 0.19876422109878705), (5, 11, 'A', 'C', 'B', 0.19866780686972374), (7, 11, 'A', 'C', 'B', 0.19866780686972374), (7, 9, 'A', 'C', 'B', 0.19815220034038428), (10, 9, 'A', 'C', 'B', 0.19815220034038428), (11, 9, 'A', 'C', 'B', 0.19815220034038428), (12, 9, 'A', 'C', 'B', 0.19815220034038428), (6, 10, 'A', 'C', 'B', 0.19754856342798738), (7, 10, 'A', 'C', 'B', 0.1975401795819815), (8, 10, 'A', 'C', 'B', 0.1975401795819815), (9, 10, 'A', 'C', 'B', 0.1975401795819815), (10, 10, 'A', 'C', 'B', 0.1975401795819815), (11, 10, 'A', 'C', 'B', 0.1975401795819815), (12, 10, 'A', 'C', 'B', 0.1975401795819815), (13, 10, 'A', 'C', 'B', 0.1975401795819815), (6, 11, 'A', 'C', 'B', 0.19749406842895117), (9, 11, 'A', 'C', 'B', 0.19691977497757363), (10, 11, 'A', 'C', 'B', 0.19691977497757363), (11, 11, 'A', 'C', 'B', 0.19691977497757363), (13, 11, 'A', 'C', 'B', 0.19691977497757363), (3, 12, 'A', 'C', 'B', 0.1969030072855622), (4, 12, 'A', 'C', 'B', 0.19681078497950177), (8, 12, 'A', 'C', 'B', 0.1967814415184821), (10, 12, 'A', 'C', 'B', 0.1967814415184821), (11, 12, 'A', 'C', 'B', 0.1967814415184821), (13, 12, 'A', 'C', 'B', 0.1967814415184821), (8, 13, 'A', 'C', 'B', 0.19592209730291676), (9, 13, 'A', 'C', 'B', 0.19592209730291676), (10, 13, 'A', 'C', 'B', 0.19592209730291676), (11, 13, 'A', 'C', 'B', 0.19592209730291676), (12, 13, 'A', 'C', 'B', 0.19592209730291676), (13, 13, 'A', 'C', 'B', 0.19592209730291676), (8, 11, 'A', 'C', 'B', 0.19520108654644286), (5, 12, 'A', 'C', 'B', 0.19506275308735133), (6, 12, 'A', 'C', 'B', 0.19506275308735133), (7, 12, 'A', 'C', 'B', 0.19506275308735133), (9, 12, 'A', 'C', 'B', 0.19506275308735133), (12, 12, 'A', 'C', 'B', 0.19506275308735133), (5, 13, 'A', 'C', 'B', 0.194203408871786), (6, 13, 'A', 'C', 'B', 0.194203408871786), (7, 13, 'A', 'C', 'B', 0.194203408871786), (12, 11, 'A', 'C', 'B', 0.1940273481056703), (11, 14, 'A', 'C', 'B', 0.1927410939903031), (12, 14, 'A', 'C', 'B', 0.1927410939903031), (13, 14, 'A', 'C', 'B', 0.1927410939903031), (4, 13, 'A', 'C', 'B', 0.19249310428666044), (8, 14, 'A', 'C', 'B', 0.1909140053052623), (9, 14, 'A', 'C', 'B', 0.1909140053052623), (10, 14, 'A', 'C', 'B', 0.1909140053052623), (3, 13, 'A', 'C', 'B', 0.1907827997015351), (7, 14, 'A', 'C', 'B', 0.19060390309725056), (6, 14, 'A', 'C', 'B', 0.1873645921946439), (2, 13, 'A', 'C', 'B', 0.18584052248128324), (5, 14, 'A', 'C', 'B', 0.17836743759193058), (14, 2, 'A', 'C', 'B', 0.16901833547121398), (4, 14, 'A', 'C', 'B', 0.16769824540818262), (3, 14, 'A', 'C', 'B', 0.1437575168356166), (14, 3, 'A', 'C', 'B', 0.13654350796884568), (14, 4, 'A', 'C', 'B', 0.12157415092599572), (2, 14, 'C', 'A', 'B', 0.11587346153684991), (14, 5, 'A', 'C', 'B', 0.1113835861062904), (14, 6, 'A', 'C', 'B', 0.10663832926716804), (14, 7, 'A', 'C', 'B', 0.10257216395449248), (14, 8, 'A', 'C', 'B', 0.10257216395449248), (14, 9, 'C', 'A', 'B', 0.0979107455754254), (14, 10, 'C', 'A', 'B', 0.09673700713465283), (1, 14, 'C', 'A', 'B', 0.09518042500764795), (14, 11, 'C', 'A', 'B', 0.09438114640710249), (14, 12, 'C', 'A', 'B', 0.08961912187596943), (14, 13, 'C', 'A', 'B', 0.07844764707361834), (1, 15, 'B', 'C', 'A', 0.07412403438510667), (0, 0, 'C', 'A', 'B', 0.06218895051694939), (14, 14, 'C', 'A', 'B', 0.05194212175545043), (2, 15, 'C', 'B', 'A', 0.012848333528830136), (3, 15, 'C', 'B', 'A', 0.005668012802252154), (5, 15, 'C', 'B', 'A', 0.0042311108132110364), (0, 15, 'A', 'C', 'B', 0.0034351592740922143), (4, 15, 'C', 'B', 'A', 0.0028235333545586494), (0, 4, 'B', 'A', 'C', 0.0017175796370459406), (0, 6, 'A', 'C', 'B', 0.0017175796370459406), (0, 7, 'B', 'A', 'C', 0.0017175796370459406), (0, 10, 'A', 'C', 'B', 0.0017175796370459406), (0, 11, 'A', 'C', 'B', 0.0017175796370459406), (0, 14, 'B', 'A', 'C', 0.0017175796370459406), (8, 15, 'B', 'A', 'C', 0.0017175796370459406), (0, 1, 'A', 'B', 'C', 0), (0, 2, 'A', 'B', 'C', 0), (0, 3, 'A', 'B', 'C', 0), (0, 5, 'A', 'B', 'C', 0), (0, 8, 'A', 'B', 'C', 0), (0, 9, 'A', 'B', 'C', 0), (0, 12, 'A', 'B', 'C', 0), (0, 13, 'A', 'B', 'C', 0), (6, 15, 'A', 'B', 'C', 0), (7, 15, 'A', 'B', 'C', 0), (9, 15, 'A', 'B', 'C', 0), (10, 15, 'A', 'B', 'C', 0), (11, 15, 'A', 'B', 'C', 0), (12, 15, 'A', 'B', 'C', 0), (13, 15, 'A', 'B', 'C', 0), (14, 15, 'A', 'B', 'C', 0)]


########################################################################################
################## Critical Path
########################################################################################

"""range(bit_len - 1)"""
def create_crit(i_th):
    crit = []
    for lay in range(0, BIT_LEN-2):
        crit += [
            (lay, max(i_th - lay, 0))
        ]
    crit += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
    return crit

"""for multiplier propagation delay and optimization"""
CRITICAL_FA_lst = create_crit(0)
log.println(f"Critical eFA list: {CRITICAL_FA_lst}")

########################################################################################
##################### Functions
########################################################################################


def get_alpha(raw_mp, bit_len, log=log, rew_lst=[], verify=False):
    # return AlphaMultiprocess(raw_mp, bit_len, log=log, rew_lst=rew_lst).run()
    return AlphaSampled(raw_mp, bit_len, rew_lst).run_multi(ALPHA_SAMPLE_COUNT, RND_SEED, log=log, proc_count=PROCESS_COUNT)


def get_FA_delay(fa_alpha, temp, sec):
    tg1_alpha = max(fa_alpha[0], fa_alpha[1])
    tg1_vth = abs(BTI.Vth) + BTI.delta_vth(BTI.Vdef, temp, tg1_alpha, BTI.Tclk, sec)
    tg1_pb = pmos_vth_to_body(tg1_vth)

    tg2_alpha = max(fa_alpha[2], fa_alpha[3])
    tg2_vth = abs(BTI.Vth) + BTI.delta_vth(BTI.Vdef, temp, tg2_alpha, BTI.Tclk, sec)
    tg2_pb = pmos_vth_to_body(tg2_vth)

    return tgate_pb_to_delay(tg1_pb) + tgate_pb_to_delay(tg2_pb)


def get_MP_delay(critical_fa_lst, alpha, temp, sec):
    ps = 0
    for fa_lay, fa_i in critical_fa_lst:
        ps += get_FA_delay(alpha[fa_lay][fa_i], temp, sec)
    return ps


def get_best_worst_wire_comb(
        log = log,
        bitlen = BIT_LEN,
        temp = TEMP,
        mp = Wallace_rew,
        critical_fa_lst = CRITICAL_FA_lst,
):
    # default wiring in multiplier
    best_wiring = [fa + ('A', 'B', 'C', 0) for fa in critical_fa_lst]
    worst_wiring = [fa + ('A', 'B', 'C', 0) for fa in critical_fa_lst]

    default_alpha = get_alpha(mp, bitlen, log=False, rew_lst=[])
    # log.println(f"{default_alpha}")
    if log:
        log.println(f"default alpha done")


    # iterate in Critical FA list, and log the worst wiring for each
    for fa_index, fa in enumerate(critical_fa_lst):
        wire_combination = [
            ('A', 'B', 'C'),
            ('A', 'C', 'B'),
            ('B', 'A', 'C'),
            ('B', 'C', 'A'),
            ('C', 'A', 'B'),
            ('C', 'B', 'A'),
        ]
        lay, i = fa
        FA_zero_delay = get_FA_delay(default_alpha[lay][i], temp, 0)

        aging_period = 12*30 *24*60*60
        fa_default_delay = get_FA_delay(default_alpha[lay][i], temp, aging_period)
        fa_default_aging_rate = (fa_default_delay - FA_zero_delay) / FA_zero_delay
        if log:
            log.println(f"default wiring, delay rate: {fa_default_aging_rate * 100 :.2f}% [t:{aging_period}s]")
        
        _worst_rate = fa_default_aging_rate
        _best_rate = fa_default_aging_rate
        for comb in wire_combination[1:]:
            rewire = fa + comb
            rewire_alpha = get_alpha(mp, bitlen, log=False, rew_lst=[rewire])

            fa_delay = get_FA_delay(rewire_alpha[lay][i], temp, aging_period)
            fa_aging_rate = (fa_delay - FA_zero_delay) / FA_zero_delay
            if log:
                log.println(f"{rewire}, delay rate: {fa_aging_rate * 100 :.2f}% [t:{aging_period}s]")

            if fa_aging_rate > _worst_rate:
                _worst_rate = fa_aging_rate
                worst_wiring[fa_index] = fa + comb + (fa_aging_rate - fa_default_aging_rate, )
                if log:
                    log.println(f"-new worst")
            if fa_aging_rate < _best_rate:
                _best_rate = fa_aging_rate
                best_wiring[fa_index] = fa + comb + (fa_aging_rate - fa_default_aging_rate, )
                if log:
                    log.println(f"-new best")
            
        if log:
            log.println()
    
    if log:
        log.println(f"best wiring combination: \n{best_wiring}")
        log.println(f"worst wiring combination: \n{worst_wiring}")

    return (best_wiring, worst_wiring)



"""
wire combination notation:

(0, 0, 'C', 'B', 'A', 0.50)
- FA index
- FA wiring combination
- difference of aging_rate and default_aging_rate
"""
def examine_wire_comb(
        wire_comb,
        bit_len = BIT_LEN, 
        temp = TEMP, 
        log = log, 
        plot = "DELAY" or "RATE" or True, 
        plot_save_clear = True,
        plot_label = "",
        alpha_verification = ALPHA_VERIFICATION, 
        critical_fa_lst = CRITICAL_FA_lst, 
        mp = Wallace_rew,
    ): 
    if log:   
        log.println(f"aging log for following wire comb \n{wire_comb}")
    
    alpha = get_alpha(mp, bit_len, log=False, rew_lst=wire_comb, verify=alpha_verification)
    _mp_zero_delay = get_MP_delay(critical_fa_lst, alpha, temp, 0)
    
    res_week = []
    res_delay = []

    for week in range(0, 200):
        delay = get_MP_delay(critical_fa_lst, alpha, temp, week * 7 *24*60*60)
        aging_rate = (delay - _mp_zero_delay) / _mp_zero_delay
        if log:
            log.println(f"week {week:03}: {delay: 8.3f} [{aging_rate * 100 :4.2f}%]")

        if plot:
            res_week.append(week)

            if plot == "DELAY":
                res_delay.append(delay)
            elif plot == "RATE" or True:
                res_delay.append(aging_rate)

    
    if plot:
        plt.plot(res_week, res_delay, label=plot_label)
        plt.title(f"WallceTree-BIT-{bit_len}-TEMP-{temp}")

        if plot_save_clear:
            timestamp = datetime.now().strftime('%m,%d-%H:%M:%S.%f')
            fig_name = f"fig-{timestamp}.jpg"
            plt.legend()
            plt.savefig(fig_name)
            plt.clf()
            if log:
                log.println(f"plot saved in {fig_name}")
    

def examine_multi_wire_comb(
        multi_wire_comb,
        plot_labels,
        log = log, 
        plot = True,
    ): 

    for i_sub_wc, sub_wc in enumerate(multi_wire_comb):
        examine_wire_comb(
            sub_wc,
            log = log, 
            plot = plot, 
            plot_save_clear = True if i_sub_wc==len(multi_wire_comb)-1 else False,
            plot_label = plot_labels[i_sub_wc],
        )

def sort_rewiring(wiring):
    return sorted(wiring, key=lambda x: x[-1], reverse=True)

########################################################################################
################## MAIN
########################################################################################


if False and (__name__ == "__main__"):
    """each path gets full rewiring -> aging(1 year) -> route with higher aging higher priority"""
    log.println(f"RUNNING: critical path priorities, bit len [{BIT_LEN}]")
    for i in range(BIT_LEN - 1):
        critical_path = create_crit(i)
        _, worst_wiring = get_best_worst_wire_comb(
            log=False,
            bitlen=BIT_LEN,
            temp=TEMP,
            mp=Wallace_rew,
            critical_fa_lst=critical_path
        )

        ts_1year = 50 *7 *24*60*60

        alpha_nomitigation = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=[])
        alpha_rewired = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=worst_wiring)
        delay_nomitigation = get_MP_delay(critical_path, alpha_nomitigation, TEMP, ts_1year)
        delay_rewired = get_MP_delay(critical_path, alpha_rewired, TEMP, ts_1year)

        log.println(f"path [{i}] -> delay({ts_1year}s): {delay_nomitigation:.4f} -> {delay_rewired:.4f}")
        log.println(f"{sort_rewiring(worst_wiring)}")


"""specific wire combination aging"""
if True and (__name__ == "__main__"):
    # normal aging without mitigation

    # REW_LST = []
    REW_LST = tamper_critical_path

    #rewire top-10% of circuit
    circuit_size = BIT_LEN * (BIT_LEN - 1)
    REW_LST = sort_rewiring(REW_LST)
    REW_LST = REW_LST[:(circuit_size//10)]
    
    examine_wire_comb(
        wire_comb=REW_LST, 
        bit_len=BIT_LEN, 
        temp=TEMP, 
        log=log, 
        plot=True, 
        alpha_verification=ALPHA_VERIFICATION,
        critical_fa_lst=CRITICAL_FA_lst
        )

"""
extracting best and worst wiring combination for the provided multiplier
"""
if False and (__name__ == "__main__"):
    log.println(f"RUNNING: extracting best and worst wiring in critical-path")
    best_wiring, worst_wiring = get_best_worst_wire_comb(log=False)
    log.println(f"worst wiring:\n{worst_wiring}")
    
    examine_multi_wire_comb(
        [worst_wiring, [], best_wiring],
        ["attack", "no-mitigation", "optimization"],
        log=log,
        plot="DELAY",
    )


"""
partial rewiring
"""
if False and (__name__ == "__main__"):
    _, worst_wiring = get_best_worst_wire_comb(log=False)
    worst_wiring = sorted(worst_wiring, key=lambda x: x[-1], reverse=True)

    full_combo = []
    full_combo_label = []
    for combo in [0, len(worst_wiring)//4, len(worst_wiring)//2, len(worst_wiring)*3//4, len(worst_wiring)]:
        full_combo.append(worst_wiring[0:combo])
        full_combo_label.append(f"{combo} / {len(worst_wiring)}")
    log.println(f"wire combo:\n{full_combo}")

    
    examine_multi_wire_comb(
        multi_wire_comb = full_combo,
        plot_labels = full_combo_label,
        log = log,
        plot = "DELAY"
    )


"""
rewiring list for all the FAs (sorted)
"""
if False and (__name__ == "__main__"):
    log.println(f"RUNNING: extracting best and worst wiring for full-circuit")
    lst = []
    for fa_i in range(BIT_LEN - 1):
        for fa_j in range(BIT_LEN):
            lst += [(fa_i, fa_j)]

    best_wiring, worst_wiring = get_best_worst_wire_comb(critical_fa_lst=lst, log=False)
    log.println(f"worst complete wiring:\n{worst_wiring}")
    worst_wiring = sort_rewiring(worst_wiring)
    log.println(f"worst complete wiring (sorted):\n{worst_wiring}")



"""
error rate of wire combination
"""
if False and __name__ == "__main__":
    REW_LST = tamper_full_circuit

    log.println(f"RUNNING: error rate bitlen [{BIT_LEN}], REW_LST [{len(REW_LST)}]: \n{REW_LST}")

    alpha_notamper = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=[], verify=False)
    log.println(f"alpha_notamper: \n{alpha_notamper}")
    alpha = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=REW_LST, verify=False)
    log.println(f"alpha: \n{alpha}")

    margin_t_sec = 199 *7 *24*60*60
    max_ps_delay = get_MP_delay(CRITICAL_FA_lst, alpha_notamper, TEMP, margin_t_sec)   # fixed margin
    log.println(f"max_ps_delay: {max_ps_delay}")
    
    res = []
    # for t_week in range(200):
    for t_week in [180, 190, 199]:
        t_sec = t_week *7 *24*60*60
        
        err_rate, max_seen_delay = wallace_multiplier_error_rate_sample(BIT_LEN, alpha, TEMP, t_sec, max_ps_delay, ALPHA_SAMPLE_COUNT, RND_SEED)
        res.append(err_rate)
        
        log.println(f"REW [{len(REW_LST)}] week [{t_week:03}], error rate: {err_rate:.4f}, max seen delay: {max_seen_delay:.3f}, max_allowed_delay: {max_ps_delay:.3f}")
    log.println(f"REW [{len(REW_LST)}], error rate: \n{res}")

