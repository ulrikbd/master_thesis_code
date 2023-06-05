import seaborn
import numpy as np

import matplotlib.pyplot as plt
#Options
params = {'text.usetex' : False,
          'font.size' : 9,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params)

from face_whisker import (
        load_face, load_post,
        load_model,
)

from helper_functions import (
        describe_pipeline, plot_watershed_heat,
        get_contours, plot_ethogram,
        get_trans_prob_mat, plot_cum_pca,
        plot_pca_embedding, plot_colored_tsne_map,
        colored_time_map, plot_partial_labels,
        get_partial_labels,
)


def compare_models(bc1, bc2, name1, name2):
    """
    Function which compares two models,
    both descriptively and visually

    Arguments:
        bc1, bc2 (pipeline): Trained models
        name1, name2 (str): Names for saving
            comparison plots
    """

    compare_descriptives(bc1, bc2, name1, name2)
    compare_heatmaps(bc1, bc2, name1, name2)
    compare_ethograms(bc1, bc2, name1, name2)
    compare_transition_matrices(bc1, bc2, name1, name2)
    compare_pca(bc1, bc2, name1, name2)
    compare_pc_embedding(bc1, bc2, name1, name2)
    compare_time_in_behaviors(bc1, bc2, name1, name2)
    compare_partial_labels(bc1, bc2, name1, name2)



def compare_descriptives(bc1, bc2, name1, name2):
    """
    Load the models with/without whisker data
    and compare basic statistics of the model,
    number of principal components, found behaviors,
    etc.
    """


    print("Behaviors found:")
    print(name1, "-", len(np.unique(bc1.ws_labels)))
    print(name2, "-", len(np.unique(bc2.ws_labels)))

    print("Principal components:")
    print(name1, "-", bc1.n_pca)
    print(name1, "-", bc2.n_pca)



def compare_heatmaps(bc1, bc2, name1, name2):
    """
    Plot the heatmaps side by side
    """

    fig = plt.figure()
    plt.subplot(121)
    contours_post = get_contours(bc1.kde, bc1.ws_labels)
    plot_watershed_heat(bc1.embedded, bc1.kde,
                        contours_post, bc1.border)
    plt.subplot(122)
    contours_face = get_contours(bc2.kde, bc2.ws_labels)
    plot_watershed_heat(bc2.embedded, bc2.kde,
                        contours_face, bc2.border)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # plt.show()
    plt.savefig("./figures/comparisons/" + name1 + "_" + name2 + "_" + "heatmap_comparison.pdf", 
                bbox_inches = "tight")


def compare_ethograms(bc1, bc2, name1, name2):
    """
    Plot the ethograms in a certain time period side by side
    """
    
    t1 = 4000
    t2 = 8000
    t = np.arange(t1, t2) / bc1.capture_framerate_post

    plt.figure(figsize = (12, 5))
    plt.subplot(121)
    plot_ethogram(bc1.beh_labels[t1:t2], t)
    plt.xlabel("t [s]")
    plt.subplot(122)
    plot_ethogram(bc2.beh_labels[t1:t2], t)
    plt.xlabel("t [s]")
    # plt.show()
    plt.savefig("./figures/comparisons/" + name1 + "_" + name2 + "_" + "ethogram_comparison.pdf", 
                bbox_inches = "tight")

    plt.figure(figsize = (8, 5))
    plot_ethogram(bc1.beh_labels[t1:t2], t)
    plot_ethogram(bc2.beh_labels[t1:t2], t)
    plt.xlabel("t [s]")
    plt.legend([name1, name2])
    # plt.show()
    plt.savefig("./figures/comparisons/" + name1 + "_" + name2 + "_" + "ethogram_doubled.pdf", 
                bbox_inches = "tight")


def compare_transition_matrices(bc1, bc2, name1, name2):
    """
    Compute and compare the transition
    probability matrices describing 
    movements between behaviors in the
    two models, with/without facial data.
    """

    t_mat_post = get_trans_prob_mat(
            bc1.beh_labels)
    t_mat_face = get_trans_prob_mat(
            bc2.beh_labels)
    diag_post = np.diagonal(t_mat_post)
    diag_face = np.diagonal(t_mat_face)

    print("Mean diagonal:", np.mean(diag_post), np.mean(diag_face))
    
    print("Non-zero elements in rows")
    non_zero_post = np.count_nonzero(t_mat_post, axis = 1)
    print("Postural:", non_zero_post)
    print("Average:", np.mean(non_zero_post))
    non_zero_face = np.count_nonzero(t_mat_face, axis = 1)
    print("Whisker:", non_zero_face)
    print("Average:", np.mean(non_zero_face))


def compare_pca(bc1, bc2, name1, name2):
    """
    Compare explained variance of the 
    used principal components for
    both models
    """

    pca_post = bc1.var_pca
    pca_face = bc2.var_pca

    plt.figure(figsize = (4.85, 2.2))
    plt.tight_layout()
    ax1 = plt.subplot(121)
    plot_cum_pca(pca_post)
    # plt.title(name1)
    ax2 = plt.subplot(122, sharey = ax1)
    plot_cum_pca(pca_face)
    plt.ylabel("")
    # plt.title(name2)
    # plt.show()
    plt.savefig("./figures/comparisons/" + name1 + "_" + name2 + "_" + "compare_pca_variance.png",
                bbox_inches = "tight", dpi = 1200)


def compare_pc_embedding(bc1, bc2, name1, name2):
    """
    Compare the two-dimensional PC 
    embeddings.
    """
    
    plt.figure(figsize = (15, 10))
    plt.subplot(231)
    plot_pca_embedding(bc1, 1, 2)
    plt.title(name1)
    plt.subplot(232)
    plot_pca_embedding(bc1, 1, 3)
    plt.title(name1)
    plt.subplot(233)
    plot_pca_embedding(bc1, 2, 3)
    plt.title(name1)
    plt.subplot(234)
    plot_pca_embedding(bc2, 1, 2)
    plt.title(name2)
    plt.subplot(235)
    plot_pca_embedding(bc2, 1, 3)
    plt.title(name2)
    plt.subplot(236)
    plot_pca_embedding(bc2, 2, 3)
    plt.title(name2)
    # plt.show()
    plt.savefig("./figures/comparisons/" + name1 + "_" + name2 + "_" + "pc_embedding.pdf",
                bbox_inches = "tight")


def compare_time_in_behaviors(bc1, bc2, name1, name2):
    """
    Plot the colored t-SNE map showing proportion
    of time spent in each region for both models
    side by side.
    """

    time_map_1 = colored_time_map(bc1)
    time_map_2 = colored_time_map(bc2)

    plt.figure(figsize = (12, 5))
    plt.subplot(121)
    plot_colored_tsne_map(bc1, time_map_1)
    # plt.gca().images[0].colorbar.remove()
    plt.subplot(122)
    plot_colored_tsne_map(bc2, time_map_2)
    # plt.gca().images[0].colorbar.remove()
    # plt.show()
    plt.savefig("./figures/comparisons/" + name1 + "_" + name2 + "_" + "time_duration_comp.pdf",
                bbox_inches = "tight")


def compare_partial_labels(bc1, bc2, name1, name2):
    """
    Plot how the partial labelled data are clustered
    """

    part_lab = get_partial_labels(bc1)

    plt.figure(figsize = (12, 5))
    plt.subplot(121)
    plot_partial_labels(bc1, part_lab)
    plt.title(name1)
    plt.subplot(122)
    plot_partial_labels(bc2, part_lab)
    plt.title(name2)
    # plt.show()
    plt.savefig("./figures/comparisons/" + name1 + "_" + name2 + "_" + "partial_labels.pdf",
                bbox_inches = "tight")


def main():
    seaborn.set_theme(font = "serif", font_scale = 0.65)

    # compare_models(load_post(), load_face(), "postural", "whisker")
    # compare_models(load_face(), load_model("./models/face_high_freq.pkl"), "equal_freq", "high_facial_freq")
    # compare_models(load_post(), load_model("./models/face_high_freq.pkl"), "postural", "high_facial_freq")
    # compare_models(load_model("./models/face_50_maxfreq.pkl"), load_model("./models/face_high_freq.pkl"), "50_max_freq", "high_facial_freq")
    # compare_models(load_model("./models/face_high_freq.pkl"), load_model("./models/face_pca_cutoff.pkl"), "high_facial_freq", "pca_cutoff")
    # compare_models(load_post(), load_model("./models/whiskers_only.pkl"), "postural", "whisker_only")
    # compare_models(load_model("./models/whiskers_only.pkl"), load_model("./models/face_high_freq.pkl"), "whisker_only", "face_high_freq")
    # compare_models(load_post(), load_model("./models/face_narrow_freq.pkl"), "postural", "narrow_freq")
    # compare_models(load_post(), load_model("./models/separate_pca.pkl"), "postural", "separate_pca")

    compare_pca(load_post(), load_face(), "post", "face")



if __name__ == "__main__":
    main()


