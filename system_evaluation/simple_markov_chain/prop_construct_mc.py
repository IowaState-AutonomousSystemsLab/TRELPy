#!/uIsr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:12:22 2021
@author: apurvabadithela
"""
# The latest: construct_MP3.py (3/9 at 11:10 am)
import numpy as np
from tulip.transys import MarkovChain as MC
from tulip.transys import MarkovDecisionProcess as MDP
from itertools import compress, product
from tulip.interfaces import stormpy as stormpy_int
import os
from tulip.transys.compositions import synchronous_parallel
import pickle as pkl
from pdb import set_trace as st
import random 

prop_model_MC = "prop_model_MC.nm"

# Read confusion matrix from nuscnes file:
def read_confusion_matrix(cm_fn, prop_dict_file):
    conf_matrix = pkl.load( open(cm_fn, "rb" ))
    prop_dict = pkl.load( open(prop_dict_file, "rb" ))
    for k,v in conf_matrix.items():
        n = len(conf_matrix[k][0])
        break
    cm = np.zeros((n,n))
    for k, v in conf_matrix.items():
        cm += v # Total class based conf matrix w/o distance
    return cm, conf_matrix, prop_dict


# Script for confusion matrix of pedestrian
# Make this cleaner; more versatile
# C is a dicitionary: C(["ped", "nped"]) = N(observation|= "ped" | true_obs |= "nped") (cardinality of observations given as pedestrians while the true state is not a pedestrian)
# Confusion matrix for second confusion matrix
def confusion_matrix(conf_matrix, prop_dict_file):
    C = dict()
    param_C = dict()
    cm, param_cm, prop_dict = read_confusion_matrix(conf_matrix, prop_dict_file)
    st()
    C = construct_confusion_matrix_dict(cm, prop_dict)
    for k, cm in param_cm.items():
        param_C[k] = construct_confusion_matrix_dict(cm, prop_dict)
    return C, param_C, prop_dict # Parametrized cm

def construct_confusion_matrix_dict(cm, prop_dict):
    C = dict()    
    total_gt = dict()
    for k, v in prop_dict.items():
        total_gt = np.sum(cm[:,k])
        for j in prop_dict.keys():
            if total_gt !=0.0:
                C[j,k] = cm[j,k]/total_gt
            else:
                C[j,k] = 0.0
    return C

# Sensitivity:
def construct_CM(tp_ped, true_env_type, prop_dict):
    C = dict()
    for k, v in prop_dict.items():
        v == set({true_env_type})
    C["ped", "ped"] = tp_ped
    coeff = random.random()
    C["obj", "ped"] = coeff*(1-tp_ped)
    C["empty", "ped"] = (1-coeff)*(1 - tp_ped)

    C["ped", "obj"] = 0.1*(1-tp_obj)
    C["obj", "obj"] = tp_obj
    C["empty", "obj"] = 0.9*(1-tp_obj)

    C["ped", "empty"] = 0.5*(1-tp_emp)
    C["obj", "empty"] = 0.5*(1-tp_emp)
    C["empty", "empty"] = tp_emp
    return C
    
# Script for confusion matrix of pedestrian
# Varying the precision/recall confusion matrix values
def confusion_matrix_ped2(prec, recall):
    C = dict()
    tp = recall*100
    fn = tp/prec - tp
    tn = 200 - fn
    C["ped", "ped"] = (recall*100)/100.0
    C["ped", "obs"] = (fn/2.0)/100.0
    C["ped", "empty"] = (fn/2.0)/100.0

    C["obs", "ped"] = ((1-recall)*100.0/2)/100.0
    C["obs", "obs"] = (tn/2*4.0/5)/100.0
    C["obs", "empty"] = (tn/2*1/5)/100.0

    C["empty", "ped"] = ((1-recall)*100/2)/100.0
    C["empty", "obs"] = (tn/2*1.0/5)/100.0
    C["empty", "empty"] = (tn/2*4.0/5.0)/100.0
    tol = 1e-4
    assert(abs(C["ped", "ped"] + C["obs", "ped"] + C["empty", "ped"] - 1.0) < tol)
    assert(abs(C["ped", "obs"] + C["obs", "obs"] + C["empty", "obs"]- 1.0)< tol)
    assert(abs(C["ped", "empty"] + C["obs", "empty"] + C["empty", "empty"]- 1.0) < tol)
    return C

# Function that converts to a Markov chain from states and actions:
def _construct_mdpmc(states, transitions, init, actions=None):
    if actions is not None:
        ts = MDP()
        ts.actions.add_from(actions)
    else:
        ts = MC()
    ts.states.add_from(states)
    ts.states.initial.add(init)

    for transition in transitions:
        attr = {"probability": transition[2]}
        if len(transition) > 3:
            attr["action"] = transition[3]
        ts.transitions.add(
            transition[0],
            transition[1],
            attr,
        )

    for s in states:
        ts.atomic_propositions.add(s)
        ts.states[s]["ap"] = {s}

    return ts

# Creating the states of the markov chain for the system:
# Returns product states S and (pos,vel) to state dictionary
def system_states_example_ped(Ncar, Vlow, Vhigh):
    nS = Ncar*(Vhigh-Vlow+1)
    state = lambda x,v: (Vhigh-Vlow+1)*(x-1) + v
    state_to_S = dict()
    S = set()
    for xcar in range(1,Ncar+1):
        for vcar in range(Vlow, Vhigh+1):
            st = "S"+str(state(xcar, vcar))
            state_to_S[xcar,vcar] = st
            S|={st}
    return S, state_to_S

# This script automatically generates a Markov process for modeling the probability
# of satisfaction of a temporal formula.
class synth_markov_chain:
    def __init__(self, S, O, state_to_S):
        self.states = S    # Product states for car.
        self.state_dict = state_to_S
        self.reverse_state_dict = {v: k for k, v in state_to_S.items()}
        self.obs = O
        self.true_env = None # This state is defined in terms of the observation
        self.true_env_type = None # Type of the env obsect; is in one of obs
        self.C = dict() # Confusion matrix dictionary giving: C[obs, true] =  P(obs |- phi | true |- phi)
        self.M = dict() # Two-by-two dictionary. a(i,j) = Prob of transitioning from state i to state j
        self.K_strategy = None # Dictionary containing the scripts to the controller after it has been written to file
        self.formula = []
        self.MC = None # A Tulip Markov chain obsect that is consistent with TuLiP transition system markov chain
        self.param_MC = None # TuLiP Markov chain object 
        self.true_env_MC = None # A Markov chain representing the true evolution of the environment
        self.backup = dict() # This is a backup controller.

 # Convert this Markov chain obsect into a tulip transition system:
    def to_MC(self, init):
        states = set(self.states) # Set of product states of the car
        transitions = set()
        for k in self.M.keys():
            p_approx = min(1, abs(self.M[k]))
            t = (k[0], k[1], p_approx)
            transitions |= {t}
        assert init in self.states
        self.MC = _construct_mdpmc(states, transitions, init)
        markov_chain = _construct_mdpmc(states, transitions, init)
        for state in self.MC.states:
            self.MC.states[state]["ap"] = {state}
        self.check_MC() # Checking if Markov chain is valid
        return markov_chain

# Writing/Printing Markov chains to file:
    def print_MC(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        path_MC = os.path.join(model_path, prop_model_MC)
        env_MC = os.path.join(model_path, "env_MC.nm")
        path_MC_model = stormpy_int.build_stormpy_model(path_MC)
        env_MC_model = stormpy_int.build_stormpy_model(env_MC)
        stormpy_int.print_stormpy_model(path_MC_model)
        stormpy_int.print_stormpy_model(env_MC_model)

   # Function to check if all outgoing transition probabilities for any state in the Markov chain sum to 1.
    def check_MC(self):
        T = self.MC.transitions(data=True)
        for st in self.MC.states:
            end_states = [t for t in T if t[0]==st]
            prob = [(t[2])['probability'] for t in end_states]
       #    assert abs(sum(prob)-1)<1e-4 # Checking that probabilities add up to 1 for every state

   # Sets the state of the true environment
    def set_true_env_state(self, st, true_env_type):
        self.true_env = st
        new_st = "p"+st
        states = {new_st}
        transitions ={(new_st, new_st, 1)}
        init = new_st
        self.true_env_type = true_env_type
        self.true_env_MC = _construct_mdpmc(states, transitions, init)

    def set_confusion_matrix(self, C, prop_dict):
        self.C = C
        self.label_dict = prop_dict

    def set_param_confusion_matrix(self, param_C, prop_dict):
        self.param_C = param_C
        self.label_dict = prop_dict
        
    def set_controller(self, K):
        self.K_strategy = K

    def compute_next_state(self,obs, init_st): # The next state computed using the observation from the current state
        new_state = self.K_strategy[obs][init_st]
        next_st = {'xcar': new_state[0], 'vcar': new_state[1]}
        return next_st

    # Function to return the state of the environment given the observation:
    def get_env_state(self, obs):
        env_st =[1] # For static environment, env state is the same. Should modify this function for reactive environments
        if obs == self.true_env_type:
            env_st = [int(self.true_env)]
        return env_st

    # Function that returns the distance bin of a particular discrete state:
    def get_distbin(self, ped_st, init_st):
        # 1/30/24: TODO: Fix this logic. The distance bins need to be changed / fixed.
        ped_cell = ped_st[0]
        init_cell = init_st[0]
        distance_z = (abs(ped_cell-init_cell)//10)
        ld = int(distance_z*10)
        ud = int(ld + 10)
        ld += 1
        distbin = (ld, ud)
        return distbin

    # Constructing the Markov chain
    def construct_param_markov_chain(self, ped_st): # Construct probabilities and transitions in the markov chain given the controller and confusion matrix
        for Si in list(self.states):
            # print("Finding initial states in the Markov chain: ")
            
            init_st = self.reverse_state_dict[Si]
            distbin = self.get_distbin(ped_st, init_st)
            # The output state can be different depending on the observation as defined by the confusion matrix
            for obs in self.obs:
                next_st = self.compute_next_state(obs, init_st)
                Sj = self.state_dict[tuple(next_st.values())]

                if distbin not in self.param_C.keys():
                    pdb.set_trace()

                try:
                    for k, v in self.label_dict.items():
                        if v == set({obs}):
                            pred_j = k
                        if v == set({self.true_env_type}):
                            true_j = k
                    prob_t = self.param_C[distbin][pred_j, true_j] # Probability of transitions
                    if np.isnan(prob_t):
                        prob_T = 0.0
                except:
                    st()
                
                if (Si, Sj) in self.M.keys():
                    self.M[Si, Sj] = self.M[Si, Sj] + prob_t
                else:
                    self.M[Si, Sj] = prob_t
        return self.M

    # Constructing the Markov chain
    def construct_markov_chain(self): # Construct probabilities and transitions in the markov chain given the controller and confusion matrix
        for Si in list(self.states):
            init_st = self.reverse_state_dict[Si]
            for obs in self.obs:
                next_st = self.compute_next_state(obs, init_st)
                Sj = self.state_dict[tuple(next_st.values())]
                try:
                    for k, v in self.label_dict.items():
                        if v == set({obs}):
                            pred_j = k
                        if v == set({self.true_env_type}):
                            true_j = k
                    prob_t = self.C[pred_j, true_j] # Probability of transitions
                except:
                    st()
                if (Si, Sj) in self.M.keys():
                    self.M[Si, Sj] = self.M[Si, Sj] + prob_t
                else:
                    self.M[Si, Sj] = prob_t
        return self.M
    
    # Adding formulae to list of temporal logic formulas:
    def add_TL(self, phi):
        self.formula.append(phi)

    # Probabilistic satisfaction of a temporal logic with respect to a model:
    def prob_TL(self, phi):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        prism_file_path = os.path.join(model_path, "pedestrian.nm")
        path_MC = os.path.join(model_path, prop_model_MC)
        env_MC = os.path.join(model_path, "env_MC.nm")
        # Print self markov chain:
        # print(self.MC)
        # Writing prism files:
        stormpy_int.to_prism_file(self.MC, path_MC)
        stormpy_int.to_prism_file(self.true_env_MC, env_MC)
        composed = synchronous_parallel([self.MC, self.true_env_MC])
        # print(composed.transitions)
        result = stormpy_int.model_checking(composed, phi, prism_file_path)
        # Returns a tulip transys:
        # MC_ts = stormpy_int.to_tulip_transys(path_MC)
        result = stormpy_int.model_checking(self.MC, phi, prism_file_path) # Since there is no moving obstacle, try checking only the pedestrian obstacle
        #for state in self.MC.states:
        #    print("  State {}, with labels {}, Pr = {}".format(state, self.MC.states[state]["ap"], result[str(state)]))
        return result

    # Function to append labels to a .nm file:
    def add_labels(self, MAX_V):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        path_MC = os.path.join(model_path, prop_model_MC)
        for vi in range(0, MAX_V):
            var = "label \"v_"+str(vi)+"\"="
            flg_var = 0 # Set after the first state is appended
            for k, val in self.state_dict.items():
                if k[1] == vi:
                    if flg_var == 0:
                        flg_var = 1
                        var = var + "(s = "
                    var = var + ""
            with open(path_MC, 'a') as out:
                out.write(var + '\n')
