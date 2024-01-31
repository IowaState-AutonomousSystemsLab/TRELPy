import sys
sys.path.append("..")
import numpy as np
import os
from pathlib import Path
import pdb
try: 
    from system_evaluation.markov_chain import construct_mc as cmp
    from system_evaluation.controllers import construct_controllers as K_des
    from system_evaluation.markov_chain.MC import construct_controllers as K_des
except:
    from markov_chain import construct_mc as cmp
    from controllers import construct_controllers as K_des
    from markov_chain.markov_chain_setup import call_MC, call_MC_param

import matplotlib as plt
# from figure_plot import probability_plot
import time
import json
import sys
sys.setrecursionlimit(10000)

def initialize(vmax, MAX_V):
    Ncar = int(MAX_V*(MAX_V+1)/2 + 10)
    Vlow=  0
    Vhigh = vmax
    x_vmax_stop = MAX_V*(MAX_V+1)/2 + 1
    xcross_start = 2
    Nped = Ncar - xcross_start + 1
    if x_vmax_stop >= xcross_start:
        min_xped = int(x_vmax_stop + 1 - (xcross_start - 1))
    else:
        min_xped = 3
    assert(min_xped > 0)
    assert(min_xped<= Nped)
    if min_xped < Nped:
        xped = np.random.randint(min_xped, Nped)
    else:
        xped = int(min_xped)
    xped = int(min_xped)
    xcar_stop = xped + xcross_start - 2
    assert(xcar_stop > 0)
    state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1)+v
    bad_states = set()
    good_state = set()
    def get_formula_states(xcar_stop):
        bst = set()
        for vi in range(0,Vhigh+1):
            state = state_f(xcar_stop, vi)
            bst |= {"S"+str(state)}
        gst = {"S" + str(state_f(xcar_stop,0))}
        bad = "" # Expression for bad states
        good = "" # Expression for good states
        for st in list(gst):
            if good == "":
                good = good + "\"" + st+"\""
            else:
                good = good + "|\""+st+"\""
        for st in list(bst):
            if bad == "":
                bad = bad + "\"" + st+"\""
            else:
                bad = bad + "|\""+st+"\""
        return good, bad, gst, bst
    good, bad, gst, bst = get_formula_states(xcar_stop)
    good_state |= gst
    bad_states |= bst
    formula = "P=?[!("+str(bad)+") U "+str(good)+"]"

    phi1 = "!("+good+")"
    phi2 = "("+good+") | !("+bad
    for xcar_ii in range(xcar_stop+1, Ncar+1):
        good, bad, gst, bst = get_formula_states(xcar_ii) # We only want the bad states; ignore the good states output here
        bad_states |= bst
        phi2 = phi2 + "|" + bad
    phi2 = phi2 + ")"
    formula = "P=?[G("+str(phi1)+") && G("+str(phi2)+")]"
    return Ncar, Vlow, Vhigh, xcross_start, xped, bad_states, good_state, formula

cm_fn = "/home/apurvabadithela/software/run_nuscenes_evaluations/saved_cms/lidar/mini/cm.pkl"
control_dir = "/home/apurvabadithela/software/run_nuscenes_evaluations/system_evaluation/controllers/"
C, param_C = cmp.confusion_matrix(cm_fn)

VMAX = []
INIT_V = dict()
P = dict()
P_param = dict()
# For some reason, unable to synthesize for max_v = 6
MAX_V = 6
for vmax in range(1,MAX_V-1):
    INIT_V[vmax] = []
    P[vmax] = []
    P_param[vmax] = []
    print("===========================================================")
    print("Max Velocity: ", vmax)
    # Initial conditions set for all velocities
    Ncar, Vlow, Vhigh, xcross_start, xped, bad_states, good_state, formula = initialize(vmax, MAX_V)
    print("Specification: ")
    print(formula)
    for vcar in range(1, vmax+1):  # Initial speed at starting point
        state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1) + v
        start_state = "S"+str(state_f(1,vcar))
        print(start_state)
        S, state_to_S, K_backup = cmp.system_states_example_ped(Ncar, Vlow, Vhigh)
        K = K_des.construct_controllers(Ncar, Vlow, Vhigh, xped, vcar,control_dir=control_dir)
        true_env = str(1) #Sidewalk 3
        true_env_type = "ped"
        O = {"ped", "obj", "empty"}
        state_info = dict()
        state_info["start"] = start_state
        state_info["bad"] = bad_states
        state_info["good"] = good_state
        for st in list(good_state):
            formula2 = 'P=?[F(\"'+st+'\")]'
        M = call_MC(S, O, state_to_S, K, K_backup, C, true_env, true_env_type, state_info)
        param_M = call_MC_param(S, O, state_to_S, K, K_backup, param_C, true_env, true_env_type, xped, state_info)

        # result = M.prob_TL(formula)
        result2 = M.prob_TL(formula2)
        result_param = param_M.prob_TL(formula2)
        pdb.set_trace()
        print('Probability of eventually reaching good state for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, vmax, result2[start_state]))
        # Store results:
        VMAX.append(vmax)
        INIT_V[vmax].append(vcar)
        # p = result[start_state]
        # print('Probability of satisfaction for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, vmax, p))
        P[vmax].append(result2[start_state])
        P_param[vmax].append(result_param[start_state])

# Write to json file:
results_folder = "/home/apurvabadithela/software/run_nuscenes_evaluations/saved_cms/lidar/mini/probability_results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
result_type="class"
dataset_label="mini"
fname_v = Path(f"{results_folder}/{dataset_label}_{result_type}_cm_ped_vmax_"+str(MAX_V)+"_initv.json")
fname_p = Path(f"{results_folder}/{dataset_label}_{result_type}_cm_ped_vmax_"+str(MAX_V)+"_prob.json")
fname_param_p = Path(f"{results_folder}/{dataset_label}_{result_type}_param_cm_ped_vmax_"+str(MAX_V)+"_prob.json")

#pdb.set_trace()
with open(fname_v, 'w') as f:
    json.dump(INIT_V, f)
with open(fname_p, 'w') as f:
    json.dump(P, f)
with open(fname_param_p, 'w') as f:
    json.dump(P_param, f)
