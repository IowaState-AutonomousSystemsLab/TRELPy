import sys
sys.path.append("../..")
import controllers.ped_controller_init_speed_1 as Kped
import controllers.not_ped_controller_init_speed_1 as Kobj
import controllers.empty_controller_init_speed_1 as Kempty
import markov_chain.construct_mc as cmp
import markov_chain.distance_param_construct_mc as param_cmp
import importlib
from pdb import set_trace as st

def call_MC(S, O, state_to_S, K, K_backup, C, true_env, true_env_type, state_info):
    importlib.reload(Kped)
    importlib.reload(Kobj)
    importlib.reload(Kempty)

    K_strat = dict()
    K_strat["ped"] = Kped
    K_strat["obs"] = Kobj
    K_strat["empty"] = Kempty

    # K_strat = K

    obs_keys = dict()
    obs_keys["ped"] = ["xcar", "vcar"]
    obs_keys["obs"] = ["xobs"]
    obs_keys["empty"] = ["xempty"]

    M = cmp.synth_markov_chain(S, O, state_to_S)
    M.set_confusion_matrix(C)
    M.set_true_env_state(true_env, true_env_type)
    M.set_controller(K, K_strat, K_backup)

    M.construct_internal_state_maps()

    # Construct Markov chain:
    M.construct_markov_chain()
    start_state = state_info["start"]
    MC = M.to_MC(start_state) # For setting initial conditions and assigning bad/good labels
    for k,v in M.M.items():
        s1 = M.reverse_state_dict[k[0]]
        s2 = M.reverse_state_dict[k[1]]
        print("(start: %s"% (s1,) + ", end: %s"% (s2,) +"): "+str(v))
    return M

def call_MC_param(S, O, state_to_S, K, K_backup, param_C, true_env, true_env_type, xped, state_info):
    # importlib.reload(Kped)
    # importlib.reload(Kobj)
    # importlib.reload(Kempty)

    # K_strat = dict()
    # K_strat["ped"] = Kped
    # K_strat["obs"] = Kobj
    # K_strat["empty"] = Kempty

    K_strat = K

    obs_keys = dict()
    obs_keys["ped"] = ["xcar", "vcar"]
    obs_keys["obs"] = ["xobs"]
    obs_keys["empty"] = ["xempty"]

    M = param_cmp.synth_markov_chain(S, O, state_to_S)
    M.set_confusion_matrix(param_C)
    M.set_true_env_state(true_env, true_env_type)
    M.set_controller(K, K_strat, K_backup)

    M.construct_internal_state_maps()

    # Construct Markov chain:
    ped_st = [xped, 0]
#    pdb.set_trace()
    M.construct_markov_chain(ped_st)
    start_state = state_info["start"]
    MC = M.to_MC(start_state) # For setting initial conditions and assigning bad/good labels
    for k,v in M.M.items():
        s1 = M.reverse_state_dict[k[0]]
        s2 = M.reverse_state_dict[k[1]]
        print("(start: %s"% (s1,) + ", end: %s"% (s2,) +"): "+str(v))
    return M
