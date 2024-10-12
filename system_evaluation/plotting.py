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

def precision_recall_plots(INIT_V, P, prec_recall, fig_name,title=None):
    fig, ax = plt.subplots()
    width = 8
    height = 6
    fig.set_size_inches(width, height)  
    ax.tick_params(axis='both', which='major', labelsize=15)
    for k,v in INIT_V.items():
        probabilities = P[k]
        init_speed = INIT_V[k]
        # st()
        prec, rec = prec_recall[int(k)]
        ax.plot(init_speed, probabilities, 'o--', label=f"$p={prec}$, $r={rec}$")
    leg = ax.legend(loc="best", fontsize=15)
    ax.set_xlabel("Initial speed",fontsize=20)
    ax.set_ylabel("Probability of satisfaction", fontsize=20)
    ax.set_xticks(np.arange(1,max(init_speed)+1,1))
    # if title:
    #     plt.title(title,fontsize=20)
    ax.set_ylim(0,1.1)
    plt.savefig(fig_name, format='png', dpi=400, bbox_inches = "tight")

def plot_precision_recall(results_folder, result_type, true_env_type, MAX_V):
    fname_v = f"{results_folder}/{result_type}_cm_{true_env_type}_vmax_"+str(MAX_V)+"_initv.json"
    fname_p = f"{results_folder}/{result_type}_cm_{true_env_type}_vmax_"+str(MAX_V)+"_prob.json"
    fname_pr = f"{results_folder}/{result_type}_cm_{true_env_type}_vmax_"+str(MAX_V)+"_pr_pairs.json"

    figure_folder = Path(f"{results_folder}/figures")
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    fig_name = f"{figure_folder}/{result_type}_cm_{true_env_type}_vmax_"+str(MAX_V)+".png"

    title=f"Guarantees for varied precision/recall with {true_env_type} true environment"
    with open(fname_v) as fv:
        INIT_V = json.load(fv)
    with open(fname_p) as fp:
        P = json.load(fp)
    with open(fname_pr) as fp:
        prec_recall = json.load(fp)
    
    precision_recall_plots(INIT_V, P, prec_recall, fig_name,title=title)

def sensitivity_probability_plot_w_errorbars(INIT_V, P, std_P, fig_name,title=None):
    fig, ax = plt.subplots()
    width = 8
    height = 6
    fig.set_size_inches(width, height)  
    ax.tick_params(axis='both', which='major', labelsize=15)
    for k,v in INIT_V.items():
        probabilities = P[k]
        init_speed = INIT_V[k]
        error_bar = std_P[k]
        TP = "TP"
        plt.errorbar(init_speed, probabilities, yerr=error_bar,  fmt='-.o', capsize=6,label=f"$\mathtt{{TP}}={k}$")
    leg = plt.legend(loc="best")
    plt.xlabel("Initial speed",fontsize=20)
    plt.ylabel("Probability of satisfaction", fontsize=20)
    plt.xticks(np.arange(1,max(init_speed)+1,1))
    # if title:
    #     plt.title(title,fontsize=20)
    ax.set_ylim(0,1.1)
    plt.savefig(fig_name, format='png', dpi=400, bbox_inches = "tight")

def plot_sensitivity_results_w_errorbars(results_folder, result_type, MAX_V):
    runs = 20
    fname_v = f"{results_folder}/{result_type}_cm_ped_vmax_"+str(MAX_V)+"_initv.json"
    fname_p = f"{results_folder}/{result_type}_cm_ped_vmax_"+str(MAX_V)+"_mean_prob_runs_"+str(runs)+".json"
    fname_p = f"{results_folder}/{result_type}_cm_ped_vmax_"+str(MAX_V)+"_mean_prob_runs_"+str(runs)+".json"
    fname_stdp = f"{results_folder}/{result_type}_cm_ped_vmax_"+str(MAX_V)+"_std_prob"+str(runs)+".json"
    fname_tp = f"{results_folder}/{result_type}_tp.json"

    figure_folder = Path(f"{results_folder}/figures")
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    fig_name = f"{figure_folder}/{result_type}_cm_ped_vmax_"+str(MAX_V)+".png"

    title="Sensitivity for Max. Speed = "+str(MAX_V)
    with open(fname_v) as fv:
        INIT_V = json.load(fv)
    with open(fname_p) as fp:
        P = json.load(fp)
    with open(fname_stdp) as fp:
        std_P = json.load(fp)
    with open(fname_tp) as fp_tp:
        tp = json.load(fp_tp)

    sensitivity_probability_plot_w_errorbars(INIT_V, P, std_P, fig_name, title=title)

def plot_results(results_folder, MAX_V, true_env_type):
    figure_folder = Path(f"{results_folder}/figures")
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    fig_name = Path(f"{figure_folder}/guarantees_cm_{true_env_type}_vmax_"+str(MAX_V)+".png")
    # fig_name = Path(f"{figure_folder}/class_guarantees_cm_{true_env_type}_vmax_"+str(MAX_V)+".png")
    # fig_name = Path(f"{figure_folder}/prop_guarantees_cm_{true_env_type}_vmax_"+str(MAX_V)+".png")

    fig, ax= plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=15)
    max_p = update_max()
    title = "System-level Guarantees"
    CM = "CM"
    cm_dict = {"class": "$\mathtt{{CM}}_{{class}}$", "prop": "$\mathtt{{CM}}_{{logic}}$", "prop_seg": "$\mathtt{{CM}}_{{logic, seg}}$",
    "class_param": "$\{\mathtt{{CM}^k}_{{class}}\}$", "prop_param": "$\{\mathtt{{CM}^k}_{{logic}}\}$", "prop_seg_param": "$\{\mathtt{{CM}^k}_{{logic, seg}}\}$"}
    
    # for res_type in ["prop"]:
    for res_type in ["class", "prop", "prop_seg"]:
        INIT_V, P, P_param = load_result(results_folder, res_type, true_env_type, MAX_V) 
        if res_type != "prop_seg":
            ax.plot(INIT_V, P, 'o--', label=cm_dict[res_type])
        ax.plot(INIT_V, P_param, 'o--', label=cm_dict[res_type+"_param"])
        # plot_probability(INIT_V, P, max_p, res_type, ax)
        # plot_probability(INIT_V, P_param, max_p, res_type+"_param", ax)    
    
    leg = ax.legend(loc="best", fontsize=15)
    ax.set_xlabel("Initial speed",fontsize=20)
    ax.set_ylabel("Probability of satisfaction", fontsize=20)
    ax.set_xticks(np.arange(1,MAX_V+1,1))
    # if title:
    #     ax.set_title(title,fontsize=20)
    y_upper_lim = min(1, max_p+0.1)
    ax.set_ylim(0,1.1)
    ax.get_figure().savefig(fig_name, format='png', dpi=400, bbox_inches = "tight")

if __name__=="__main__":
    MAX_V = 6
    results_folder = f"{cm_dir}/probability_results"
    results_folder = f"{cm_dir}/simulated_probability_results_v1"

    true_env_type = "ped"
    plot_results(results_folder, MAX_V, true_env_type)

    # true_env_type = "obs"
    # plot_results(results_folder, MAX_V, true_env_type)
    # plot_sensitivity_results_w_errorbars(results_folder, "prop_sensitivity", MAX_V)

    # true_env_type = "obs"
    # plot_precision_recall(results_folder, "prec_recall", true_env_type, MAX_V)

    # true_env_type = "ped"
    # plot_precision_recall(results_folder, "prec_recall", true_env_type, MAX_V)