import matplotlib.pyplot as plt
import numpy as np
import json
import sys
sys.path.append("..")
import os
import pdb
from pathlib import Path
from experiment_file import *
from system_evaluation.utils.plotting_utils import update_max

def probability_plot(INIT_V, P, fig_name,title=None):
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=15)
    max_p = update_max()
    for k,v in INIT_V.items():
        probabilities = P[k]
        max_p = update_max(probabilities, max_p)
        init_speed = INIT_V[k]
        plt.plot(init_speed, probabilities, 'o--', label=f"V={k}")
    leg = plt.legend(loc="best", fontsize=20)
    # plt.legend(fontsize=20)
    plt.xlabel("Initial speed",fontsize=15)
    plt.ylabel("Probability of satisfaction", fontsize=15)
    plt.xticks(np.arange(1,10,1))
    if title:
        plt.title(title,fontsize=20)
    y_upper_lim = min(1, max_p+0.1)
    ax.set_ylim(0,max_p + 0.1)
    plt.savefig(fig_name, format='png', dpi=400, bbox_inches = "tight")
    # plt.show()

def sensitivity_probability_plot(INIT_V, P, fig_name,title=None):
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=15)
    for k,v in INIT_V.items():
        probabilities = P[k]
        init_speed = INIT_V[k]
        plt.plot(init_speed, probabilities, 'o--', label=f"TP={k}")
    leg = plt.legend(loc="best")
    plt.xlabel("Initial speed",fontsize=15)
    plt.ylabel("Probability of satisfaction", fontsize=15)
    plt.xticks(np.arange(1,10,1))
    if title:
        plt.title(title,fontsize=20)
    ax.set_ylim(0,1)
    plt.savefig(fig_name, format='png', dpi=1200, bbox_inches = "tight")

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

        plt.errorbar(init_speed, probabilities, yerr=error_bar,  fmt='-.o', capsize=6,label=f"TP={k}")
    leg = plt.legend(loc="best")
    plt.xlabel("Initial speed",fontsize=15)
    plt.ylabel("Probability of satisfaction", fontsize=15)
    plt.xticks(np.arange(1,11,1))
    if title:
        plt.title(title,fontsize=20)
    ax.set_ylim(0,1.1)
    plt.savefig(fig_name, format='png', dpi=400, bbox_inches = "tight")


def probability_split_plot(INIT_V, P, fig_name):
    fig, ax = plt.subplots(2)
    for k,v in INIT_V.items():
        probabilities = P[k]
        init_speed = INIT_V[k]
        if int(k)<=5:
            ax[0].plot(init_speed, probabilities, 'o--', label=f"V={k}")
        else:
            ax[1].plot(init_speed, probabilities, 'o--', label=f"V={k}")
    leg = ax[0].legend(loc="best")
    leg2 = ax[1].legend(loc="best")
    for axi in ax.flat:
        axi.set(xlabel='Initial speed', ylabel='Probability')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axi in ax.flat:
        axi.label_outer()
    plt.xlabel("Initial speed")
    plt.ylabel("Probability of satisfaction")
    plt.xticks(np.arange(1,11,1))
    plt.savefig(fig_name, format='png', dpi=1200)
    plt.show()

def probability_individual_plot(INIT_V, P, fig_name):
    fig, ax = plt.subplots(len(INIT_V)//2, 2)

    for k,v in INIT_V.items():
        probabilities = P[k]
        init_speed = INIT_V[k]
        row = (int(k)-1)%5
        col = (int(k)-1)//5
        ax[row, col].plot(init_speed, probabilities, 'o--', label=f"V={k}")
        leg = ax[row, col].legend(loc="best")
    for axi in ax.flat:
        axi.set(xlabel='Initial speed', ylabel='Probability')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axi in ax.flat:
        axi.label_outer()

    plt.xticks(np.arange(1,11,1))
    ax.set_ylim(0,1)
    plt.savefig(fig_name, format='png', dpi=1200)
    plt.show()

def plot_results(results_folder, MAX_V, res_type):
    fname_v = Path(f"{results_folder}/{res_type}_cm_ped_vmax_"+str(MAX_V)+"_initv.json")
    fname_p = Path(f"{results_folder}/{res_type}_cm_ped_vmax_"+str(MAX_V)+"_prob.json")
    fname_p_param = Path(f"{results_folder}/{res_type}_param_cm_ped_vmax_"+str(MAX_V)+"_prob.json")
    figure_folder = Path(f"{results_folder}/figures")
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    fig_name = Path(f"{figure_folder}/{res_type}_cm_ped_vmax_"+str(MAX_V)+".png")
    fig_name_param = Path(f"{figure_folder}/{res_type}_param_cm_ped_vmax_"+str(MAX_V)+".png")
    
    with open(fname_v) as fv:
        INIT_V = json.load(fv)
    with open(fname_p) as fp:
        P = json.load(fp)
    with open(fname_p_param) as fp_param:
        P_param = json.load(fp_param)
    
    if res_type == "prop":
        probability_plot(INIT_V, P, fig_name, title="Proposition-based")
        probability_plot(INIT_V, P_param, fig_name_param, title="Proposition-based, distance-parametrized")
    elif res_type == "prop_seg":
        probability_plot(INIT_V, P, fig_name, title="Proposition-based Segmented ")
        probability_plot(INIT_V, P_param, fig_name_param, title="Proposition-based, segmeneted, distance-parametrized")
    else:
        probability_plot(INIT_V, P, fig_name, title="Class-based")
        probability_plot(INIT_V, P_param, fig_name_param, title="Class-based, distance-parametrized")

def plot_sensitivity_results(MAX_V):
    fname_v = "results/sensitivity_cm_ped_vmax_"+str(MAX_V)+"_initv.json"
    fname_p = "results/sensitivity_cm_ped_vmax_"+str(MAX_V)+"_prob.json"
    fname_tp = "results/sensitivity_tp.json"
    fig_name = "figures/sensitivity_cm_ped_vmax_"+str(MAX_V)+".png"
    title="Sensitivity for Max. Speed = "+str(MAX_V)
    with open(fname_v) as fv:
        INIT_V = json.load(fv)
    with open(fname_p) as fp:
        P = json.load(fp)
    with open(fname_tp) as fp_tp:
        tp = json.load(fp_tp)

    sensitivity_probability_plot(INIT_V, P, fig_name,title=title)

def plot_sensitivity_results_w_errorbars(MAX_V):
    # fname_v = "results/sensitivity_cm_ped_vmax_"+str(MAX_V)+"_initv.json"
    # fname_p = "results/sensitivity_cm_ped_vmax_"+str(MAX_V)+"_mean_prob_runs_50.json"
    # fname_stdp = "results/sensitivity_cm_ped_vmax_"+str(MAX_V)+"_std_prob50.json"
    # fname_tp = "results/sensitivity_tp.json"
    runs = 20
    Ncar = 48
    fname_v = "results/sensitivity_cm_ped_vmax_"+str(MAX_V)+"_initv.json"
    fname_p = "results/sensitivity_cm_ped_vmax_"+str(MAX_V)+"_mean_prob_runs_"+str(runs)+"_ncells_"+str(Ncar)+".json"
    fname_stdp = "results/sensitivity_cm_ped_vmax_"+str(MAX_V)+"_std_prob"+str(runs)+"_ncells_"+str(Ncar)+".json"
    fname_tp = "results/sensitivity_tp.json"

    fig_name = "figures/sensitivity_cm_ped_vmax_"+str(MAX_V)+".png"
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

if __name__=="__main__":
    MAX_V = 6
    # plot_results(MAX_V, "prop_based")
    results_folder = Path(f"{cm_dir}/probability_results")
    result_type = "prop"
    plot_results(results_folder, MAX_V, result_type)
    #plot_sensitivity_results(MAX_V)
    # plot_sensitivity_results_w_errorbars(MAX_V)

