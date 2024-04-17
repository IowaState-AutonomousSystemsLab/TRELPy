# Common imports and functions:
from system_evaluation.print_utils import print_cm, print_param_cm
from system_evaluation.formula import *
import json
from system_evaluation.simple_markov_chain import construct_mc as cmp
from system_evaluation.simple_markov_chain.setup_mc import call_MC, call_MC_param
from custom_env import *

cm_fn = f"{cm_dir}/low_thresh_cm.pkl"
prop_cm_fn = f"{cm_dir}/low_thresh_prop_cm.pkl"
prop_cm_seg_fn = f"{cm_dir}/low_thresh_prop_cm_cluster.pkl"
prop_dict_file = f"{cm_dir}/prop_dict.pkl"
control_dir = f"{repo_dir}/system_evaluation/controllers/"

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

def simulate(MAX_V=6):
    Ncar = init(MAX_V=MAX_V)
    C, param_C = cmp.confusion_matrix(cm_fn)
    print(" =============== Full confusion matrix ===============")
    print_cm(C)
    print(" =============== Parametrized confusion matrix ===============")
    print_param_cm(param_C)
    print("===========================================================")
    INIT_V, P, P_param = compute_probabilities(Ncar, MAX_V, C, param_C,true_env_type="ped")
    save_results(INIT_V, P, P_param, "class", "ped")

def compute_probabilities(Ncar, MAX_V,C, param_C,true_env_type="ped"):
    INIT_V = []
    P = []
    P_param = []
    
    Vlow, Vhigh, xped, formula = initialize(MAX_V, Ncar)
    print("===========================================================")
    # Initial conditions set for all velocities
    print("Specification: ")
    print(formula)
    for vcar in range(1, MAX_V+1):  # Initial speed at starting point
        state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1) + v
        start_state = "S"+str(state_f(1,vcar))
        print(start_state)
        S, state_to_S = cmp.system_states_example_ped(Ncar, Vlow, Vhigh)
        
        true_env = str(1) # Sidewalk 3
        O = {"ped", "obs", "empty"}
        class_dict = {0: {'ped'}, 1: {'obs'}, 2: {'empty'}}
        state_info = dict()
        state_info["start"] = start_state
    
        M = call_MC(S, O, state_to_S, C, class_dict, true_env, true_env_type, state_info, Ncar, xped, Vhigh)
        result = M.prob_TL(formula)
        P.append(result[start_state])

        param_M = call_MC_param(S, O, state_to_S, param_C, class_dict, true_env, true_env_type, state_info, Ncar, xped, Vhigh)
        result_param = param_M.prob_TL(formula)
        P_param.append(result_param[start_state])
        
        print('Probability of eventually reaching good state for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, MAX_V, result[start_state]))
        # Store results:
        INIT_V.append(vcar)
            
    return INIT_V, P, P_param


if __name__ == "__main__":
    MAX_V = 6 # Max speed 
    simulate(MAX_V=MAX_V)
