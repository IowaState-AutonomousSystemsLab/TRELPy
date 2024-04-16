import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pdb import set_trace as st
from pathlib import Path
from experiment_file import *
from plotting_utils import update_max

def plot_probability(INIT_V, P, max_p, name, ax):
    max_p = update_max(P, max_p)
    ax.plot(INIT_V, P, 'o--', label=name)
    st()
    
def load_result(results_folder, res_type, true_env_type, MAX_V):
    try:
        fname_v = Path(f"{results_folder}/{res_type}_cm_{true_env_type}_vmax_"+str(MAX_V)+"_initv.json")
        fname_p = Path(f"{results_folder}/{res_type}_cm_{true_env_type}_vmax_"+str(MAX_V)+"_prob.json")
        fname_p_param = Path(f"{results_folder}/{res_type}_param_cm_{true_env_type}_vmax_"+str(MAX_V)+"_prob.json")
    except:
        st()
    
    with open(fname_v) as fv:
        INIT_V = json.load(fv)
    with open(fname_p) as fp:
        P = json.load(fp)
    with open(fname_p_param) as fp_param:
        P_param = json.load(fp_param)
    return INIT_V, P, P_param
    

def plot_results(results_folder, MAX_V, true_env_type):
    figure_folder = Path(f"{results_folder}/figures")
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    fig_name = Path(f"{figure_folder}/guarantees_cm_{true_env_type}_vmax_"+str(MAX_V)+".png")

    fig, ax= plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=15)
    max_p = update_max()
    title = "System-level Guarantees"
    
    for res_type in ["class", "prop", "prop_seg"]:
        INIT_V, P, P_param = load_result(results_folder, res_type, true_env_type, MAX_V) 
        ax.plot(INIT_V, P, 'o--', label=res_type)
        ax.plot(INIT_V, P_param, 'o--', label=res_type+"_param")
        # plot_probability(INIT_V, P, max_p, res_type, ax)
        # plot_probability(INIT_V, P_param, max_p, res_type+"_param", ax)    
    
    leg = ax.legend(loc="best", fontsize=15)
    ax.set_xlabel("Initial speed",fontsize=15)
    ax.set_ylabel("Probability of satisfaction", fontsize=15)
    ax.set_xticks(np.arange(1,MAX_V+1,1))
    if title:
        ax.set_title(title,fontsize=20)
    y_upper_lim = min(1, max_p+0.1)
    ax.set_ylim(0,1)
    ax.get_figure().savefig(fig_name, format='png', dpi=400, bbox_inches = "tight")

if __name__=="__main__":
    MAX_V = 6
    results_folder = f"{cm_dir}/probability_results"
    true_env_type = "ped"
    plot_results(results_folder, MAX_V, true_env_type)