import sys
sys.path.append("../..")
import simple_markov_chain.construct_mc as cmp
import simple_markov_chain.construct_param_mc as param_cmp
from controllers.simple_controller import control_dict
import importlib
from pdb import set_trace as st

def construction(S, O, state_to_S, C, true_env, true_env_type, Ncar, xped, Vhigh):
    K_strat = control_dict(Ncar, Vhigh, O, xped)
    M = cmp.synth_markov_chain(S, O, state_to_S)
    M.set_confusion_matrix(C)
    M.set_true_env_state(true_env, true_env_type)
    M.set_controller(K_strat)
    M.construct_markov_chain()
    for k,v in M.M.items():
        s1 = M.reverse_state_dict[k[0]]
        s2 = M.reverse_state_dict[k[1]]
        print("(start: %s"% (s1,) + ", end: %s"% (s2,) +"): "+str(v))
    return M

def set_init_state(M, state_info):
    start_state = state_info["start"]
    MC = M.to_MC(start_state) # For setting initial conditions and assigning bad/good labels

def call_MC(S, O, state_to_S, C, true_env, true_env_type, state_info, Ncar, xped, Vhigh):
    K_strat = control_dict(Ncar, Vhigh, O, xped)
    M = cmp.synth_markov_chain(S, O, state_to_S)
    M.set_confusion_matrix(C)
    M.set_true_env_state(true_env, true_env_type)
    M.set_controller(K_strat)

    # Construct Markov chain:
    M.construct_markov_chain()
    start_state = state_info["start"]
    MC = M.to_MC(start_state) # For setting initial conditions and assigning bad/good labels
    
    return M

def call_MC_param(S, O, state_to_S, param_C, true_env, true_env_type, state_info, Ncar, xped, Vhigh):
    K_strat = control_dict(Ncar, Vhigh, O, xped)
    M = param_cmp.synth_markov_chain(S, O, state_to_S)
    M.set_confusion_matrix(param_C)
    M.set_true_env_state(true_env, true_env_type)
    M.set_controller(K_strat)

    # Construct Markov chain:
    ped_st = [xped, 0]
    M.construct_markov_chain(ped_st)
    start_state = state_info["start"]
    MC = M.to_MC(start_state) # For setting initial conditions and assigning bad/good labels
    for k,v in M.M.items():
        s1 = M.reverse_state_dict[k[0]]
        s2 = M.reverse_state_dict[k[1]]
        print("(start: %s"% (s1,) + ", end: %s"% (s2,) +"): "+str(v))
    return M
