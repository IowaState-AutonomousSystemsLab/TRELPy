import sys
sys.path.append("../..")
import simple_markov_chain.prop_construct_mc as cmp
from controllers.simple_controller import prop_control_dict
import importlib
from pdb import set_trace as st

def call_MC(S, O, state_to_S, C, prop_dict, true_env, true_env_type, state_info, Ncar, xped, Vhigh):
    K_strat = prop_control_dict(Ncar, Vhigh, O, xped)
    M = cmp.synth_markov_chain(S, O, state_to_S)
    M.set_confusion_matrix(C, prop_dict)
    M.set_true_env_state(true_env, true_env_type)
    M.set_controller(K_strat)

    # Construct Markov chain:
    M.construct_markov_chain()
    start_state = state_info["start"]
    MC = M.to_MC(start_state) # For setting initial conditions and assigning bad/good labels
    return M

def call_MC_param(S, O, state_to_S, param_C, prop_dict, true_env, true_env_type, state_info, Ncar, xped, Vhigh):
    K_strat = prop_control_dict(Ncar, Vhigh, O, xped)
    M = cmp.synth_markov_chain(S, O, state_to_S)
    M.set_param_confusion_matrix(param_C, prop_dict)
    M.set_true_env_state(true_env, true_env_type)
    M.set_controller(K_strat)

    # Construct Markov chain:
    ped_st = (xped, 0)
    M.construct_param_markov_chain(ped_st)
    start_state = state_info["start"]
    MC = M.to_MC(start_state) # For setting initial conditions and assigning bad/good labels
    for k,v in M.M.items():
        s1 = M.reverse_state_dict[k[0]]
        s2 = M.reverse_state_dict[k[1]]
        print("(start: %s"% (s1,) + ", end: %s"% (s2,) +"): "+str(v))
    return M