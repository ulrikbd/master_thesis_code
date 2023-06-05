import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
#Options
params = {'text.usetex' : False,
          'font.size' : 9,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params)

import seaborn

from gtda.homology import FlagserPersistence
from helper_functions import (
        compute_persistence, save_persistence_diagram,
        get_trans_prob_mat, plot_persistence_diagrams,
        plot_relative_bottleneck,
)

from face_whisker import load_model

import plotly.graph_objects as go

import persim

import os

import pickle as pkl


def toy_example():
    """
    Test how the persistent homology compares
    two example transition probability matrices,
    derived from two different descriptions of
    going to get coffee. 
    """

    # Scenario 1
    # 1: Writing thesis
    # 2: Walking through the hallway
    # 3: Getting coffee
    # 4: Speak to family
    # Scenario 2
    # 1: Writing theiss
    # 2: Opening the door
    # 3: Walking through the hallway
    # 4: Get a cup from the cupboard 
    # 5: Pour coffee
    # 6: Speak to family
    # 7: Stare out the window


    # Initialize storage for behavior labels
    scenario_1 = []
    scenario_2 = []

    # Go through a session
    # First we write the thesis for 60 seconds
    scenario_1.extend([1]*60)
    scenario_2.extend([1]*60)
    # Then we decide to get coffee, and go to the kitchen
    scenario_1.extend([2]*5)
    scenario_2.extend([2]*2)
    scenario_2.extend([3]*3)
    # Pouring in coffee
    scenario_1.extend([3]*20)
    scenario_1.extend([4]*5)
    scenario_1.extend([3]*15)
    scenario_2.extend([4]*3)
    scenario_2.extend([5]*3)
    scenario_2.extend([7]*14)
    scenario_2.extend([6]*5)
    scenario_2.extend([5]*15)
    # Go back to writing
    scenario_1.extend([2]*5)
    scenario_2.extend([3]*3)
    scenario_2.extend([2]*2)
    # Write thesis again
    scenario_1.extend([1]*10)
    scenario_2.extend([1]*10)

    # Add noise
    np.random.seed(1)
    n_1 = len(scenario_1)
    noise_1 = np.random.randint(np.min(scenario_1), np.max(scenario_1)+1, size=n_1)
    scenario_1[::20] = noise_1[::20]
    n_2 = len(scenario_2)
    noise_2 = np.random.randint(np.min(scenario_2), np.max(scenario_2)+1, size=n_2)
    scenario_2[::20] = noise_1[::20]

    # Compute probability matrices
    # Scenario 1
    trans_mat_1 = get_trans_prob_mat(scenario_1)
    np.fill_diagonal(trans_mat_1, 0)
    trans_mat_1 = trans_mat_1 / np.sum(trans_mat_1, axis = 1,
                                       keepdims = True)
    trans_mat_1 = np.ones(trans_mat_1.shape) - trans_mat_1
    np.fill_diagonal(trans_mat_1, 0)

    # Scenario 2
    trans_mat_2 = get_trans_prob_mat(scenario_2)
    np.fill_diagonal(trans_mat_2, 0)
    trans_mat_2 = trans_mat_2 / np.sum(trans_mat_2, axis = 1,
                                       keepdims = True)
    trans_mat_2 = np.ones(trans_mat_2.shape) - trans_mat_2
    np.fill_diagonal(trans_mat_2, 0)

    pers_1, pers_trans_1 = compute_persistence(scenario_1)
    pers_2, pers_trans_2 = compute_persistence(scenario_2)
    pers_noise, pers_trans_noise = compute_persistence(noise_1)

    distance_bottleneck, matching = persim.bottleneck(pers_trans_1[0], pers_trans_2[0], matching=True)



    plt.figure(figsize = (3.4, 3.4))
    plot_persistence_diagrams([pers_trans_1[0], pers_trans_2[0]],
                              title = "",
                              labels = ["Scenario 1", "Scenario 2"])

    plt.savefig("./figures/persistence_diagrams/toy_example.png",
                bbox_inches="tight", dpi = 600)
#     plt.show()


    seaborn.set_theme(font = "serif", font_scale = 0.75)
    plt.figure(figsize = (3.4, 3.4))
    ax = plt.axes()
    persim.bottleneck_matching(pers_trans_1[0,:,:-1], pers_trans_2[0,:,:-1],
                               matching, labels = [
                                   "Scenario 1", "Scenario 2"])
    plots = ax.get_children()
    sc = plots[1]
    new_marker = MarkerStyle("X")
    sc.set_paths((new_marker.get_path(),))
    sc.set_sizes([40])
    plt.legend(loc = "lower right")
    plt.legend(prop={'family': 'serif'})
    plt.savefig("./figures/persistence_diagrams/matching.png",
                bbox_inches = "tight", dpi = 600)
    # plt.show()



def plot_all_persistence_diagrams():
    """
    Plot and save all persistence diagrams 
    for the trained models varying the bandwidth 
    parameter.
    """

    # Get filenames
    cwd = os.getcwd()
    filenames = [path.split("/")[-1] for path in
                 os.listdir(cwd + "/persistence_pairs/")]

    for filename in filenames:
        path = "./persistence_pairs/" + filename
        with open(path, "rb") as file:
            dgm = pkl.load(file)
        
        dgm = dgm[0]
        dgm0 = dgm[np.where(dgm[:,-1] == 0)]
        dgm1 = dgm[np.where(dgm[:,-1] == 1)]
        
        name = filename.split(".pkl")[0]
        title = "Bandwidth: " + name.split("_")[-1]

        seaborn.set_theme(font = "serif", font_scale = 0.75)
        plt.figure(figsize = (2.42, 2.42))
        persim.plot_diagrams([dgm0, dgm1], size = 11,
                                  labels = [r'$H_0$', r'$H_1$'])
        plt.title(title, fontsize = 9)
        plt.legend(prop={'family': 'serif'}, loc = "lower right",
                   fontsize = 9)
        savepath = "./figures/persistence_diagrams/"
        savepath += name + "_pyplot.png"
        # plt.show()
        plt.savefig(savepath, bbox_inches = "tight",
                    dpi = 600)
        plt.close()
        

def plot_bottleneck():
    """
    Create plots showing how the
    bottleneck distance varies as 
    a function of the difference in 
    the bandwidth parameter.
    """

    # Get filenames
    cwd = os.getcwd()
    filenames = [path.split("/")[-1] for path in
                 os.listdir(cwd + "/persistence_pairs/")]

    # Storage for persistence diagrams
    ft_dgms = {}
    post_dgms = {}

    # Retrieve the persistence diagrams
    for filename in filenames:
        path = "./persistence_pairs/" + filename
        with open(path, "rb") as file:
            dgm = pkl.load(file)

        model = filename.split("_")[0]
        bw = float(filename.split("_")[-1].split(".pkl")[0])
        if model == "post":
            post_dgms[bw] = dgm
        elif model == "ft":
            ft_dgms[bw] = dgm

    ## Plot for the postural model
    bandwidths = np.sort(np.array(list(post_dgms.keys())))
    bottlenecks = np.zeros(len(bandwidths))
    ref_bw = bandwidths[len(bandwidths) // 2]
    # Compute the bottleneck distance for each of the 
    # bandwidths, relative to the reference model
    for i in range(len(bandwidths)):
        ref_dgm = post_dgms[ref_bw][0]
        comp_dgm = post_dgms[bandwidths[i]][0]
        bottlenecks[i] = persim.bottleneck(ref_dgm, comp_dgm)

    

    plt.figure(figsize = (3.4, 7))
    plt.subplot(211)
    plot_relative_bottleneck(bandwidths, bottlenecks, ref_bw)
    # plt.savefig("./figures/bottleneck/ft_bw_" + str(ref_bw) + ".png",
                # bbox_inches = "tight", dpi = 600)
    # plt.show()
    # plt.close()

    ## Plot for the facial tracking model
    bandwidths = np.sort(np.array(list(ft_dgms.keys())))
    bottlenecks = np.zeros(len(bandwidths))
    ref_bw = bandwidths[len(bandwidths) // 2]
    # Compute the bottleneck distance for each of the 
    # bandwidths, relative to the reference model
    for i in range(len(bandwidths)):
        ref_dgm = ft_dgms[ref_bw][0]
        comp_dgm = ft_dgms[bandwidths[i]][0]
        bottlenecks[i] = persim.bottleneck(ref_dgm, comp_dgm)

    plt.subplot(212)
    plot_relative_bottleneck(bandwidths, bottlenecks, ref_bw)
    # plt.savefig("./figures/bottleneck/post_bw_" + str(ref_bw) + ".png",
                # bbox_inches = "tight", dpi = 600)
    plt.savefig("./figures/bottleneck/together.png", dpi = 600,
                bbox_inches = "tight")
    # plt.show()
    plt.close()


def plot_theory_example():
    """
    Plot two persistance diagrams together
    with the bottleneck distance maching,
    to be uses as a theory illustration.
    """

    dgm1 = np.array([
        [0.6, 0.9],
        [0.53, 0.8],
        [0.5, 0.54]
        ])

    dgm2 = np.array([
        [0.55, 0.92],
        [0.7, 0.8]
        ])

    d, matching = persim.bottleneck(
            dgm1,
            dgm2,
            matching = True
        )

    plt.figure(figsize = (4.85, 4.85))
    ax = plt.axes()
    persim.bottleneck_matching(dgm1, dgm2, matching, labels = ["Dgm 1", "Dgm 2"], ax = ax)
    plots = ax.get_children()
    sc = plots[1]
    new_marker = MarkerStyle("X")
    sc.set_paths((new_marker.get_path(),))
    sc.set_sizes([20])
    # ax.legend(loc = "lower right")
    plt.legend(prop={'family': 'serif'}, loc = "lower right")
    plt.savefig("./figures/persistence_diagrams/theory.png", dpi = 600,
                bbox_inches = "tight")
    # plt.show()

def main():
    seaborn.set_theme(font = "serif", font_scale = 0.75)
    # toy_example()
    # plot_all_persistence_diagrams()
    plot_bottleneck()
    # plot_theory_example()


if __name__ == "__main__":
    main()

