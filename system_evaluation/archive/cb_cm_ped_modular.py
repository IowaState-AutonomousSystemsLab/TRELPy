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
    from system_evaluation.markov_chain.markov_chain_setup import call_MC, call_MC_param
except:
    from markov_chain import construct_mc as cmp
    from markov_chain.markov_chain_setup import call_MC, call_MC_param

import importlib
import matplotlib as plt
# from figure_plot import probability_plot
import time
import json
import sys
sys.setrecursionlimit(10000)

def get_state(x, v, Vhigh, Vlow=0):
    state_num = (Vhigh-Vlow+1)*(x-1)+v
    state_str = "S"+str(state_num)
    return state_num, state_str

def get_formula_states(xcar_stop, Vhigh, Vlow=0):
        bst = set()
        _, good_state_str = get_state(xcar_stop, 0, Vhigh)
        gst = {good_state_str}
        for vi in range(0,Vhigh+1):
            _, state_str = get_state(xcar_stop, vi, Vhigh)
            if state_str not in gst:
                bst |= {state_str}
        
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

def set_crosswalk_cell(Ncar, xmax_stop):
    '''
    Inputs: 
    Vhigh: Highest possible speed for a car
    Ncar: Number of discrete cells for a car
    xmax_stop: The maximum cell that a car can stop by at its highest initial speed. 
    Vhigh need not be the car's highest initial speed.
    '''
    
    Nped = Ncar - 1 # Penultimate cell
    xped = np.random.randint(xmax_stop+1, Ncar) # Pick a pedestrian cell from [xmax_stop+1, Ncar]
    
    xcar_stop = xped - 1 # Cell state of the car by which v = 0
    assert(xcar_stop >= xmax_stop)
    return xped, xcar_stop

def formula_deprec(xcar_stop, Vhigh, Vlow=0):
    '''
    Formula to compute the probability of not reaching a bad state until a good state has been visited.
    This formula is not used in the results but was kept in the code.
    '''
    bad_states = set()
    good_state = set()
    
    good, bad, gst, bst = get_formula_states(xcar_stop, Vhigh, Vlow=0)
    good_state |= gst
    bad_states |= bst
    formula = "P=?[!("+str(bad)+") U "+str(good)+"]"
    return formula

def formula_visit_bad_states(Ncar, xcar_stop, Vhigh, Vlow=0):
    '''
    Probability of never visiting a good state, and never ...
    '''
    bad_states = set()
    good_state = set()
    
    good, bad, gst, bst = get_formula_states(xcar_stop, Vhigh, Vlow=0)

    phi1 = "!("+good+")"
    phi2 = "("+good+") | !("+bad

    for xcar_ii in range(xcar_stop+1, Ncar+1):
        good, bad, gst, bst = get_formula_states(xcar_ii) # We only want the bad states; ignore the good states output here
        bad_states |= bst
        phi2 = phi2 + "|" + bad
    phi2 = phi2 + ")"
    formula = "P=?[G("+str(phi1)+") && G("+str(phi2)+")]"
    return formula

def formula_ev_good(xcar_stop, Vhigh, Vlow=0):
    '''
    Probability of eventually reaching a good state
    '''
    good_state = set()
    
    good, bad, gst, bst = get_formula_states(xcar_stop, Vhigh, Vlow=0)
    good_state |= gst
    for st in list(good_state):
        formula = 'P=?[F(\"'+st+'\")]'
    return formula

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

def get_controllers(vcar):
    K = dict()
    module_name = f"controllers.ped_controller_init_speed_{vcar}"
    Kped = importlib.import_module(module_name, package=None)
    st()
    module_name = f"controllers.not_ped_controller_init_speed_{vcar}"
    Kobs = importlib.import_module(module_name, package=None)

    module_name = f"controllers.empty_controller_init_speed_{vcar}"
    Kempty = importlib.import_module(module_name, package=None)

    K["ped"] = Kped
    K["obs"] = Kobs
    K["empty"] = Kempty
    return K

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
            K = get_controllers(vcar)
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