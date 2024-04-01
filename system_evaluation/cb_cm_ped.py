# Copy of cb_cm_single_ped.py for comparison.
import sys
sys.path.append("..")
import numpy as np
import os
import pdb
from pathlib import Path
from experiment_file import *
from print_utils import print_cm, print_param_cm
# from ..custom_env import cm_dir, is_set_to_mini
try: 
    from system_evaluation.markov_chain import construct_mc as cmp
    from system_evaluation.controllers import construct_controllers as K_des
    from system_evaluation.markov_chain.markov_chain_setup import call_MC, call_MC_param
except:
    from markov_chain import construct_mc as cmp
    from controllers import construct_controllers as K_des
    from markov_chain.markov_chain_setup import call_MC, call_MC_param

import matplotlib as plt
# from figure_plot import probability_plot
import time
import json
import sys
from pdb import set_trace as st
sys.setrecursionlimit(10000)
from formula import *

def initialize(MAX_V, Ncar, maxv_init=None):
    '''
    Inputs::
    MAX_V: Maximum speed that the car can travel at
    Ncar: Maximum discrete states for the car
    vmax_init: Max initial speed of the car (specified if different from MAX_V)

    Outputs::
    Vlow: Minimum car speed (0)
    Vhigh: Maximum car speed (MAX_V)
    xped: Pedestrian position
    '''

    Vlow = 0
    Vhigh = MAX_V
    
    if maxv_init:
        xmax_stop = int(maxv_init*(maxv_init+1)/2 + 1) # earliest stopping point for car 
    else:
        xmax_stop = int(Vhigh*(Vhigh+1)/2 + 1) # earliest stopping point for car 
    
    xped, xcar_stop = set_crosswalk_cell(Ncar, xmax_stop)
    formula = formula_ev_good(xcar_stop, Vhigh, Vlow)
    return Vlow, Vhigh, xped, formula

def get_confusion_matrix():
    C, param_C = cmp.confusion_matrix(cm_fn)
    return C, param_C

def init(MAX_V=6):
    Ncar = int(MAX_V*(MAX_V+1)/2 + 4)
    return Ncar

def simulate(MAX_V=6):
    Ncar = init(MAX_V=MAX_V)
    INIT_V, P, P_param = compute_probabilities(Ncar, MAX_V)
    save_results(INIT_V, P, P_param)

def compute_probabilities(Ncar, MAX_V):
    C, param_C = cmp.confusion_matrix(cm_fn)
    print(" =============== Full confusion matrix ===============")
    print_cm(C)
    print(" =============== Parametrized confusion matrix ===============")
    print_param_cm(param_C)
    print("===========================================================")
    st()
    VMAX = []
    INIT_V = dict()
    P = dict()
    P_param = dict()
    for vmax in range(1,MAX_V+1):
        INIT_V[vmax] = []
        P[vmax] = []
        P_param[vmax] = []
        print("===========================================================")
        print("Max Velocity: ", vmax)
        # Initial conditions set for all velocities
        Vlow, Vhigh, xped, formula = initialize(vmax, Ncar)
        print("Specification: ")
        print(formula)
        for vcar in range(1, vmax+1):  # Initial speed at starting point
            state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1) + v
            start_state = "S"+str(state_f(1,vcar))
            print(start_state)

            S, state_to_S, K_backup = cmp.system_states_example_ped(Ncar, Vlow, Vhigh)
            K = K_des.construct_controllers(Ncar, Vlow, Vhigh, xped, vcar,control_dir=control_dir)
            st()
            true_env = str(1) # Sidewalk 3
            true_env_type = "ped"
            O = {"ped", "obj", "empty"}
            state_info = dict()
            state_info["start"] = start_state
            M = call_MC(S, O, state_to_S, K, K_backup, C, true_env, true_env_type, state_info)
            param_M = call_MC_param(S, O, state_to_S, K, K_backup, param_C, true_env, true_env_type, xped, state_info)

            # result = M.prob_TL(formula)
            result = M.prob_TL(formula)
            result_param = param_M.prob_TL(formula)
            print('Probability of eventually reaching good state for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, vmax, result[start_state]))
            # Store results:
            VMAX.append(vmax)
            INIT_V[vmax].append(vcar)
            # p = result[start_state]
            # print('Probability of satisfaction for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, vmax, p))
            P[vmax].append(result[start_state])
            P_param[vmax].append(result_param[start_state])
    return INIT_V, P, P_param

def save_results(INIT_V, P, P_param):
    results_folder = f"{cm_dir}/probability_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    result_type="class"
    fname_v = Path(f"{results_folder}/{result_type}_cm_ped_vmax_"+str(MAX_V)+"_initv.json")
    fname_p = Path(f"{results_folder}/{result_type}_cm_ped_vmax_"+str(MAX_V)+"_prob.json")
    fname_param_p = Path(f"{results_folder}/{result_type}_param_cm_ped_vmax_"+str(MAX_V)+"_prob.json")

    #pdb.set_trace()
    with open(fname_v, 'w') as f:
        json.dump(INIT_V, f)
    with open(fname_p, 'w') as f:
        json.dump(P, f)
    with open(fname_param_p, 'w') as f:
        json.dump(P_param, f)

if __name__=="__main__":
    MAX_V = 6
    simulate(MAX_V)