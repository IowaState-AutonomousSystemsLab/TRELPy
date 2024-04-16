# Copy of cb_cm_single_ped.py with simple controller instead of tulip controller.
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
    from system_evaluation.simple_markov_chain import prop_construct_mc as cmp
    from system_evaluation.simple_markov_chain.setup_prop_mc import call_MC, call_MC_param
except:
    from simple_markov_chain import prop_construct_mc as cmp
    from simple_markov_chain.setup_prop_mc import call_MC, call_MC_param

import matplotlib as plt
# from figure_plot import probability_plot
import time
import json
import sys
from formula import *
sys.setrecursionlimit(10000)

def get_confusion_matrix():
    C, param_C = cmp.confusion_matrix(cm_fn)
    return C, param_C

def init(MAX_V=6):
    Ncar = int(MAX_V*(MAX_V+1)/2 + 4)
    return Ncar

def save_results(INIT_V, P, P_param, result_type, true_env):
    results_folder = f"{cm_dir}/probability_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    fname_v = Path(f"{results_folder}/{result_type}_cm_{true_env}_vmax_"+str(MAX_V)+"_initv.json")
    fname_p = Path(f"{results_folder}/{result_type}_cm_{true_env}_vmax_"+str(MAX_V)+"_prob.json")
    fname_param_p = Path(f"{results_folder}/{result_type}_param_cm_{true_env}_vmax_"+str(MAX_V)+"_prob.json")

    #pdb.set_trace()
    with open(fname_v, 'w') as f:
        json.dump(INIT_V, f)
    with open(fname_p, 'w') as f:
        json.dump(P, f)
    with open(fname_param_p, 'w') as f:
        json.dump(P_param, f)
        
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
        xmax_stop = maxv_init*(maxv_init+1)/2 + 1 # earliest stopping point for car 
    else:
        xmax_stop = Vhigh*(Vhigh+1)/2 + 1 # earliest stopping point for car 
    
    xped, xcar_stop = set_crosswalk_cell(Ncar, xmax_stop)
    formula = formula_ev_good(xcar_stop, Vhigh, Vlow)
    return Vlow, Vhigh, xped, formula

def simulate_prop_sensitivity(MAX_V=6):
    Ncar = init(MAX_V=MAX_V)
    prop_dict = {0: {'empty'}, 1: {'ped'}, 2:{'obs'}, 3:{'ped','obs'}}
    INIT_V, P, P_param = compute_probabilities(Ncar, MAX_V, prop_dict)
    save_results(INIT_V, P, P_param, "prop_sensitivity", "ped")


def compute_probabilities(Ncar, MAX_V, label_dict, true_env_type="ped"):
    Vlow, Vhigh, xped, formula = initialize(MAX_V, Ncar)
    tp_range = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1-1e-5]
    runs = 20
    INIT_V = {tp: dict() for tp in tp_range}
    P = {tp: dict() for tp in tp_range}
    std_P = dict()
    P_runs = {tp: dict() for tp in tp_range}
    print("===========================================================")
    # Initial conditions set for all velocities
    print("Specification: ")
    print(formula)

    for tp in tp_range:
        INIT_V[tp] = []
        P[tp] = []
        std_P[tp] = []
        P_runs[tp] = np.zeros((MAX_V,runs)) 

        for vcar in range(1, MAX_V+1):  # Initial speed at starting point
            state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1) + v
            start_state = "S"+str(state_f(1,vcar))
            print(start_state)
            S, state_to_S = cmp.system_states_example_ped(Ncar, Vlow, Vhigh)
            
            true_env = str(1) # Sidewalk 3
            O = ["ped", "obs", ("ped","obs"), "empty"]
            state_info = dict()
            state_info["start"] = start_state
            
            for n in range(runs):
                C = cmp.construct_CM(tp, 0.9, 0.9)
                M = call_MC(S, O, state_to_S, C, label_dict, true_env, true_env_type, state_info, Ncar, xped, Vhigh)
                result = M.prob_TL(formula)
                print('Probability of eventually reaching good state for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, MAX_V, result[start_state]))
                P_runs[tp][vcar-1,n] = result2[start_state]
            
            mean = np.mean(P_runs[tp][vcar-1,:])
            stdev = np.std(P_runs[tp][vcar-1,:])
            P[tp].append(mean)
            std_P[tp].append(stdev)

            st()
            print('Probability of eventually reaching good state for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, MAX_V, result[start_state]))
            # Store results:
            INIT_V[tp].append(vcar)
        return INIT_V, P, P_param

if __name__=="__main__":
    MAX_V = 4
    simulate_prop_sensitivity(MAX_V=MAX_V)
