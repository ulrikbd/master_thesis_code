import pickle as pkl
import numpy as np

import matplotlib.pyplot as plt
#Options
params = {'text.usetex' : False,
          'font.size' : 9,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params)

import seaborn
import os

from behavioral_clustering import BehavioralClustering
from helper_functions import (
        plot_scaleogram, assign_labels,
        scale_power_spectrum, estimate_pdf,
        get_contours, get_watershed_labels,
        plot_watershed_heat, describe_pipeline,
        perplexity_tuning, plot_colored_tsne_map,
        get_average_power, get_average_value,
        perplexity_tuning, plot_partial_labels,
        get_partial_labels, get_partial_labels_whisk_only,
        plot_behavior_labels, create_csv,
        plot_session_distribution, colored_time_map,
        plot_model_summary, compute_persistence,
        save_persistence_diagram,
)


from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


sessions = ["C4_1", "C4_2", "C4_3",
            "C5_1", "C6_1", "C6_2", 
            "C7_1", "C7_2", "C8_2",
            "C8_3", "C8_4", "C9_2", 
            "C9_3", "C10_1", "C10_2", 
            "C10_3", "C11_1", "C11_2",
            "C11_3", "C12_1"]

def get_post_pipeline():
    """
    Perform the behavioral clustering on the data
    NOT including the whisker movements
    """

    extracted_file_path = "./extracted_whisker_data/"

    filenames_post = ["C4_1_all_post.pkl",
                 "C4_2_all_post.pkl",
                 "C4_3_neuro_posture_whiskers_post.pkl",
                 "C5_1_neuro_posture_whiskers_post.pkl",
                 "C6_1_neuro_posture_whiskers_post.pkl",
                 "C6_2_neuro_posture_whiskers_post.pkl",
                 "C7_1_neuro_posture_whiskers_post.pkl",
                 "C7_2_neuro_posture_whiskers_post.pkl",
                 "C8_2_neuro_posture_whiskers_post.pkl",
                 "C8_3_neuro_posture_whiskers_post.pkl",
                 "C8_4_neuro_posture_whiskers_post.pkl",
                 "C9_2_neuro_posture_whiskers_post.pkl",
                 "C9_3_neuro_posture_whiskers_post.pkl",
                 "C10_1_neuro_posture_whiskers_post.pkl",
                 "C10_2_neuro_posture_whiskers_post.pkl",
                 "C10_3_neuro_posture_whiskers_post.pkl",
                 "C11_1_neuro_posture_whiskers_post.pkl",
                 "C11_2_neuro_posture_whiskers_post.pkl",
                 "C11_3_neuro_posture_whiskers_post.pkl",
                 "C12_1_neuro_posture_whiskers_post.pkl",
                 ]

    bc = BehavioralClustering()
    bc.set_extracted_file_path(extracted_file_path)
    bc.load_relevant_features(filenames_post)
    bc.remove_nan()
    bc.detrend()
    bc.time_frequency_analysis()
    bc.standardize_features()
    bc.pca(300000)
    bc.tsne()
    bc.pre_embedding()
    bc.kernel_density_estimation(500j)
    bc.watershed_segmentation()
    bc.classify()

    return bc

def get_face_pipeline():
    """
    Perform the behavioral clustering on the data
    including the whisker movements
    """

    extracted_file_path = "./extracted_whisker_data/"

    filenames_face = ["C4_1_all_whisker.pkl",
                 "C4_2_all_whisker.pkl",
                 "C4_3_neuro_posture_whiskers_whisker.pkl",
                 "C5_1_neuro_posture_whiskers_whisker.pkl",
                 "C6_1_neuro_posture_whiskers_whisker.pkl",
                 "C6_2_neuro_posture_whiskers_whisker.pkl",
                 "C7_1_neuro_posture_whiskers_whisker.pkl",
                 "C7_2_neuro_posture_whiskers_whisker.pkl",
                 "C8_2_neuro_posture_whiskers_whisker.pkl",
                 "C8_3_neuro_posture_whiskers_whisker.pkl",
                 "C8_4_neuro_posture_whiskers_whisker.pkl",
                 "C9_2_neuro_posture_whiskers_whisker.pkl",
                 "C9_3_neuro_posture_whiskers_whisker.pkl",
                 "C10_1_neuro_posture_whiskers_whisker.pkl",
                 "C10_2_neuro_posture_whiskers_whisker.pkl",
                 "C10_3_neuro_posture_whiskers_whisker.pkl",
                 "C11_1_neuro_posture_whiskers_whisker.pkl",
                 "C11_2_neuro_posture_whiskers_whisker.pkl",
                 "C11_3_neuro_posture_whiskers_whisker.pkl",
                 "C12_1_neuro_posture_whiskers_whisker.pkl",
                 ]

    filenames_post = ["C4_1_all_post.pkl",
                 "C4_2_all_post.pkl",
                 "C4_3_neuro_posture_whiskers_post.pkl",
                 "C5_1_neuro_posture_whiskers_post.pkl",
                 "C6_1_neuro_posture_whiskers_post.pkl",
                 "C6_2_neuro_posture_whiskers_post.pkl",
                 "C7_1_neuro_posture_whiskers_post.pkl",
                 "C7_2_neuro_posture_whiskers_post.pkl",
                 "C8_2_neuro_posture_whiskers_post.pkl",
                 "C8_3_neuro_posture_whiskers_post.pkl",
                 "C8_4_neuro_posture_whiskers_post.pkl",
                 "C9_2_neuro_posture_whiskers_post.pkl",
                 "C9_3_neuro_posture_whiskers_post.pkl",
                 "C10_1_neuro_posture_whiskers_post.pkl",
                 "C10_2_neuro_posture_whiskers_post.pkl",
                 "C10_3_neuro_posture_whiskers_post.pkl",
                 "C11_1_neuro_posture_whiskers_post.pkl",
                 "C11_2_neuro_posture_whiskers_post.pkl",
                 "C11_3_neuro_posture_whiskers_post.pkl",
                 "C12_1_neuro_posture_whiskers_post.pkl",
                 ]

    bc = BehavioralClustering()
    bc.set_extracted_file_path(extracted_file_path)
    bc.load_relevant_features(filenames_post, filenames_face)
    bc.remove_nan()
    bc.detrend()
    bc.time_frequency_analysis()
    bc.standardize_features()
    bc.align_time_points()
    bc.pca(300000)
    bc.tsne()
    bc.pre_embedding()
    bc.kernel_density_estimation(500j)
    bc.watershed_segmentation()
    bc.classify()

    return bc


def pickle_data():
    """
    Extract the raw postural/whisker data and pickle it
    """

    original_filepath = "./whisker_data/"
    extracted_file_path = "./extracted_whisker_data/"
    filenames = ["C4_1_all.pkl",
                 "C4_2_all.pkl",
                 "C4_3_neuro_posture_whiskers.pkl",
                 "C5_1_neuro_posture_whiskers.pkl",
                 "C6_1_neuro_posture_whiskers.pkl",
                 "C6_2_neuro_posture_whiskers.pkl",
                 "C7_1_neuro_posture_whiskers.pkl",
                 "C7_2_neuro_posture_whiskers.pkl",
                 "C8_2_neuro_posture_whiskers.pkl",
                 "C8_3_neuro_posture_whiskers.pkl",
                 "C8_4_neuro_posture_whiskers.pkl",
                 "C9_2_neuro_posture_whiskers.pkl",
                 "C9_3_neuro_posture_whiskers.pkl",
                 "C10_1_neuro_posture_whiskers.pkl",
                 "C10_2_neuro_posture_whiskers.pkl",
                 "C10_3_neuro_posture_whiskers.pkl",
                 "C11_1_neuro_posture_whiskers.pkl",
                 "C11_2_neuro_posture_whiskers.pkl",
                 "C11_3_neuro_posture_whiskers.pkl",
                 "C12_1_neuro_posture_whiskers.pkl",
                 ]

    for filename in filenames:
        # Load data
        with open(original_filepath + filename, "rb") as file:
            data = pkl.load(file)
        
        # Extract the features which we will use
        keys = list(data["ft_raw"].keys())
        feat_names = keys[0:8] + keys[10:18]
        relevant_features = np.zeros((len(data["ft_raw"][feat_names[0]]),
                                     len(feat_names)))
        for i in range(len(feat_names)):
            relevant_features[:,i] = data["ft_raw"][feat_names[i]]

        df = {"relevant_features": relevant_features,
              "feature_names": feat_names}

        with open(extracted_file_path + filename.split(".")[0] + "_whisker_extracted.pkl", "wb") as file:
            pkl.dump(df, file)

    for filename in filenames:
        # Load data
        with open(original_filepath + filename, "rb") as file:
            data = pkl.load(file)
        
        # Extract the features which we will use
        data = data["possiblecovariates"]
        keys = list(data.keys())
        feat_names = ["B Speeds", "G Neck_elevation",
                "K Ego3_Head_roll", "L Ego3_Head_pitch",
                "M Ego3_Head_azimuth",
                "N Back_pitch", "O Back_azimuth"]

        relevant_features = np.zeros((len(data[feat_names[0]]),
                                     len(feat_names)))
        for i in range(len(feat_names)):
            relevant_features[:,i] = data[feat_names[i]]

        df = {"relevant_features": relevant_features,
              "feature_names": feat_names}

        with open(extracted_file_path + filename.split(".")[0] + "_post_extracted.pkl", "wb") as file:
            pkl.dump(df, file)


def pickle_face():
    """
    Pickle face/whisker model to be easily retrieved
    """

    path = "./models/face.pkl"
    bc = get_face_pipeline()

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def pickle_post():
    """
    Pickle postural model to be easily retrieved
    """

    path = "./models/post.pkl"
    bc = get_post_pipeline()

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def load_face():
    """
    Loads the pickled face/whisker model
    """
    path = "./models/face.pkl"
    
    return load_model(path)


def load_post():
    """
    Loads the pickled postural model
    """
    path = "./models/post.pkl"
    
    return load_model(path)


def load_model(path):
    """
    Load pickled model
    """

    with open(path, "rb") as file:
        bc = pkl.load(file)
    
    return bc


def explore_data():
    """
    Initial exploration of the face/whisker data. Max/min,
    NaN, etc.
    """

    filenames = ["C4_1_all",
                 "C4_2_all",
                 "C4_3_neuro_posture_whiskers",
                 "C5_1_neuro_posture_whiskers",
                 "C6_1_neuro_posture_whiskers",
                 "C6_2_neuro_posture_whiskers",
                 "C7_1_neuro_posture_whiskers",
                 "C7_2_neuro_posture_whiskers",
                 "C8_2_neuro_posture_whiskers",
                 "C8_3_neuro_posture_whiskers",
                 "C8_4_neuro_posture_whiskers",
                 "C9_2_neuro_posture_whiskers",
                 "C9_3_neuro_posture_whiskers",
                 "C10_1_neuro_posture_whiskers",
                 "C10_2_neuro_posture_whiskers",
                 "C10_3_neuro_posture_whiskers",
                 "C11_1_neuro_posture_whiskers",
                 "C11_2_neuro_posture_whiskers",
                 "C11_3_neuro_posture_whiskers",
                 "C12_1_neuro_posture_whiskers",
                 ]

    n_points_post = 0
    n_points_face = 0
    n_nan_points = 0

    for i in range(len(filenames)):
        # Posture data
        print("Postural data:", filenames[i])
        post_path = "./extracted_whisker_data/" + filenames[i] + "_post_extracted.pkl"
        with open(post_path, "rb") as file:
            postural_data = pkl.load(file)

        
        post_data = postural_data["relevant_features"]
        print("# Postural points:", len(post_data))
        post_names = postural_data["feature_names"]
        max_nan = 0
        for j in range(post_data.shape[1]):
            # print("Covariate:", post_names[j])
            data = post_data[:,j]
            # print("Min:", round(np.nanmin(data),2), "Max:", 
                  # round(np.nanmax(data),2))
            # print("#NaN:", np.sum(np.isnan(data)), "%NaN:",
                  # round(np.sum(np.isnan(data))/len(data)*100, 2))
            nan = np.sum(np.isnan(data))/len(data) 
            if nan > max_nan:
                max_nan = nan

        n_points_post += len(post_data)
        n_nan_points += max_nan*len(post_data)
        # face/whisker data
        face_path = "./extracted_whisker_data/" + filenames[i] + "_whisker_extracted.pkl"

        print("Whisker data:", filenames[i])

        with open(face_path, "rb") as file:
            facial_data = pkl.load(file)
        
        face_data = facial_data["relevant_features"]
        print("# Facial tracking points:", len(face_data))
        face_names = facial_data["feature_names"]
        for j in range(face_data.shape[1]):
            # print("Covariate:", face_names[j])
            data = face_data[:,j]
            # print("Min:", round(np.nanmin(data),2), "Max:", 
                  # round(np.nanmax(data),2))
            # print("#NaN:", np.sum(np.isnan(data)), "%NaN:",
                  # round(np.sum(np.isnan(data))/len(data)*100, 2))

        n_points_face += len(face_data)

    print("# points postural:", n_points_post)
    print("# points facial:", n_points_face)
    print("Average time:", n_points_post / 120 / 20)
    print("Approximate NaN ratio:", n_points_post / n_points_post)


def bandwidth_tuning(bc, model_name):
    """
    Plot the resulting clusters for several values of
    the bandwidth parameter used in the kernel 
    density estimation.
    """

    bandwidths = np.linspace(0.05, 0.15, 10)

    for bandwidth in bandwidths:
        bc.bw = bandwidth

        bc.kernel_density_estimation(500j)
        bc.watershed_segmentation()
        bc.classify()

        part_lab = get_partial_labels(bc)

        contours = get_contours(bc.kde, bc.ws_labels)
        
        plt.figure(figsize = (4.85, 4.85))
        plt.title("Bandwidth = " + str(round(bc.bw, 5)) + ", n behaviors = " + str(bc.n_behaviors) + ", Model: " + model_name)
        plot_partial_labels(bc, part_lab)
        plt.savefig("./figures/bandwidth_tuning/" + model_name + "_bw_" + str(round(bc.bw, 5)) + ".pdf", 
                bbox_inches = "tight", dpi = 600)
        plt.close()


def pickle_high_freq():
    """
    Test whether the model improves/changes
    significantly when we use higher frequencies 
    in the time frequency analysis for the 
    whisker data.
    """

    bc_high = load_face()

    # Change frequency domain
    bc_high.min_freq_face = 1
    bc_high.max_freq_face = 30
    bc_high.dj_face = 1/(bc_high.num_freq - 1)*np.log2(
                bc_high.max_freq_face/bc_high.min_freq_face)
    bc_high.aligned_features = []
    bc_high.features_face = []
    bc_high.features_post = []

    # Retrain model
    bc_high.time_frequency_analysis()
    bc_high.standardize_features()
    bc_high.align_time_points()
    bc_high.pca(300000)
    bc_high.tsne()
    bc_high.pre_embedding()
    bc_high.kernel_density_estimation(500j)
    bc_high.watershed_segmentation()
    bc_high.classify()

    path = "./models/face_high_freq.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc_high, file)


def pickle_high_freq_50():
    """
    Test whether the model improves/changes
    significantly when we use higher frequencies 
    in the time frequency analysis for the 
    whisker data.
    """

    bc_high = load_face()

    # Change frequency domain
    bc_high.min_freq_face = 1
    bc_high.max_freq_face = 50
    bc_high.dj_face = 1/(bc_high.num_freq - 1)*np.log2(
                bc_high.max_freq_face/bc_high.min_freq_face)
    bc_high.aligned_features = []
    bc_high.features_face = []
    bc_high.features_post = []

    # Retrain model
    bc_high.time_frequency_analysis()
    bc_high.standardize_features()
    bc_high.align_time_points()
    bc_high.pca(300000)
    bc_high.tsne()
    bc_high.pre_embedding()
    bc_high.kernel_density_estimation(500j)
    bc_high.watershed_segmentation()
    bc_high.classify()

    path = "./models/face_50_maxfreq.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc_high, file)


def pickle_narrow_freq():
    """
    Train model using a narrower 
    frequency interval in the 
    time-frequency analysis.
    """

    bc_high = load_face()

    # Change frequency domain
    bc_high.min_freq_face = 1
    bc_high.max_freq_face = 10
    bc_high.dj_face = 1/(bc_high.num_freq - 1)*np.log2(
                bc_high.max_freq_face/bc_high.min_freq_face)
    bc_high.aligned_features = []
    bc_high.features_face = []
    bc_high.features_post = []

    # Retrain model
    bc_high.time_frequency_analysis()
    bc_high.standardize_features()
    bc_high.align_time_points()
    bc_high.pca(300000)
    bc_high.tsne()
    bc_high.pre_embedding()
    bc_high.kernel_density_estimation(500j)
    bc_high.watershed_segmentation()
    bc_high.classify()

    path = "./models/face_narrow_freq.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc_high, file)


def pickle_pca_cutoff():
    """
    Pickle model including whisker features,
    but only using the first 50 principal components 
    in the t-SNE dimensionality reduction.
    """

    bc_high = load_model("./models/face_high_freq.pkl")

    # Concatenate features
    features = np.concatenate(bc_high.features_post, axis = 0)

    aligned_features = np.concatenate(bc_high.aligned_features,
                                      axis = 0)
    features = np.concatenate([features, aligned_features],
                                axis = 1)
        

    pca = PCA(n_components = 50)
    pca.fit(features[::len(features) // 300000,:])
    bc_high.fit_pca = pca.transform(features)

    bc_high.tsne()
    bc_high.pre_embedding()
    bc_high.kernel_density_estimation(500j)
    bc_high.watershed_segmentation()
    bc_high.classify()

    path = "./models/face_pca_cutoff.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc_high, file)


def pickle_separate_pca():
    """
    Train model using separate PCA for 
    the postural and whisker features.
    Then concatenate before tsne.
    """

    bc_high = load_model("./models/face_high_freq.pkl")

    # Concatenate features
    features = np.concatenate(bc_high.features_post, axis = 0)

    aligned_features = np.concatenate(bc_high.aligned_features,
                                      axis = 0)
        
    features_ds = features[::len(features) // 300000,:]
    aligned_features_ds = aligned_features[::len(aligned_features) // 300000,:]

    pca_post = PCA(n_components = 50)
    pca_post.fit(features_ds)
    fit_pca_post = pca_post.transform(features)

    pca_face = PCA(n_components = 10)
    pca_face.fit(aligned_features_ds)
    fit_pca_face = pca_face.transform(aligned_features)

    bc_high.fit_pca = np.concatenate([fit_pca_post, fit_pca_face],
                                     axis = 1)

    bc_high.tsne()
    bc_high.pre_embedding()
    bc_high.kernel_density_estimation(500j)
    bc_high.watershed_segmentation()
    bc_high.classify()

    path = "./models/separate_pca.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc_high, file)


def pickle_post_final():
    """
    Train the final model chosen by hyperparameter
    tuning. Perp: 50, Bw: 0.117
    """

    bc = load_post()

    bc.perp = 50
    bc.bw = 0.117

    # Train model
    bc.tsne()
    bc.pre_embedding()
    bc.kernel_density_estimation(500j)
    bc.watershed_segmentation()
    bc.classify()

    path = "./models/post_final.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def pickle_ft_final():
    """
    Train the final model chosen by hyperparameter
    tuning. Perp: 50, Bw: 0.117
    """

    bc = load_model("./models/separate_pca.pkl")

    bc.perp = 80
    bc.bw = 0.108

    # Train model
    bc.tsne()
    bc.pre_embedding()
    bc.kernel_density_estimation(500j)
    bc.watershed_segmentation()
    bc.classify()

    path = "./models/ft_final.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def pickle_whisker_only():
    """
    Train model ONLY using the whisker features
    """
    bc = load_model("./models/face_high_freq.pkl")

    pca = PCA()
    features = np.concatenate(bc.features_face, axis = 0)
    pca.fit(features[::len(features)//300000,:])
    bc.var_pca = pca.explained_variance_ratio_
    bc.n_pca = np.argmax(np.cumsum(bc.var_pca) > 0.95)

    pca = PCA(n_components = bc.n_pca)
    pca.fit(features[::len(features)//300000,:])
    bc.fit_pca = pca.transform(features)
    
    bc.ds_rate = 30000 * bc.capture_framerate_face / len(bc.fit_pca)

    bc.tsne_ind = np.arange(0, len(bc.fit_pca),
                            int(bc.capture_framerate_face / bc.ds_rate))
    train = bc.fit_pca[bc.tsne_ind,:]
    bc.embedded_train = TSNE(n_components = 2,
                             perplexity = bc.perp,
                             init = "pca").fit_transform(train)

    bc.pre_embedding()
    bc.kernel_density_estimation(500j)
    bc.watershed_segmentation()
    bc.classify()

    path = "./models/whiskers_only.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def characteristics(bc, power_spec, name, freq_range = "high"):
    """
    Plot the different behavioral regions
    in the t-SNE plane, color coded by 
    the spectral power or values of the 
    various features

    Arguments:
        bc: Behavioral pipeline
        power_spec (bool): True if we use the 
            power spectrum as the charateristic
        name (str): Model name to be used for saving
            plots
        freq_range (str): "high"/"low". Power from high 
            or low frequencies, divided in the middle
    """

    labels = np.unique(bc.beh_labels)

    face = False
    for i in range(bc.n_features_post):
        power_map = {0.0: 0.0}
        for lab in labels[1:]:
            if power_spec:
                power_map[lab] = get_average_power(bc, lab,
                                                   i, face,
                                                   freq_range)
            else:
                power_map[lab] = get_average_value(bc, lab,
                                                   i, face)
        char = bc.ws_labels.astype(object)
        char = np.vectorize(power_map.get)(char)

        plt.figure(figsize = (4.85, 4.85))
        plot_colored_tsne_map(bc, char)
        # plt.show()
        save_txt = "./figures/colored_regions/" + name
        save_txt += "_posture_"
        if power_spec:
            save_txt += freq_range 
            save_txt += "_power_feature_"
        else:
            save_txt += "value_feature_"
        save_txt += str(i) + ".png" 
        plt.savefig(save_txt, dpi = 600,
                    bbox_inches = "tight")

    if bc.face:
        face = True
        for i in range(bc.n_features_face):
            power_map = {0.0: 0.0}
            for lab in labels[1:]:
                power_map[lab] = get_average_power(bc, lab,
                                               i, face,
                                                   freq_range)
            char = bc.ws_labels.astype(object)
            char = np.vectorize(power_map.get)(char)

            plt.figure(figsize = (4.85, 4.85))
            plot_colored_tsne_map(bc, char)
            # plt.show()
            save_txt = "./figures/colored_regions/" + name
            save_txt += "_whisker_"
            if power_spec:
                save_txt += freq_range
                save_txt += "_power_feature_"
            else:
                save_txt += "value_feature_"
            save_txt += str(i) + ".png" 
            plt.savefig(save_txt, dpi = 600,
                        bbox_inches = "tight")


def vary_perplexity(bc, name):
    """
    Compute behaviors varying the perplexity
    parameter for t-SNE.
    """

    perp = [10, 20, 30, 40, 50, 80, 100,
            130, 160, 200, 300]

    for i in range(len(perp)):
        # Train model
        bc.perp = perp[i]
        bc.tsne()
        bc.pre_embedding()
        bc.kernel_density_estimation(500j)
        bc.watershed_segmentation()
        bc.classify()

        contours = get_contours(bc.kde, bc.ws_labels)
        n = len(np.unique(bc.beh_labels)) - 1
        plt.figure(figsize = (4.85, 4.85))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)
        plt.title("Perp = " + str(perp[i]) + ", n = " + str(n))
        # plt.show()
        save_txt = "./figures/perplexity_tuning/" + name
        save_txt += "_perp_" + str(perp[i]) + ".pdf"
        plt.savefig(save_txt, dpi = 600, bbox_inches = "tight")


def vary_kde_resolution(bc, name):
    """
    Compute behaviors varuing the resulution
    (grid) we compute the kernel density estimation
    in.
    """

    res = [20j, 40j, 60j, 80j, 100j,
           120j, 140j, 160j, 180j,
           200j, 300j]

    for i in range(len(res)):
        # Train model
        bc.kernel_density_estimation(res[i])
        bc.watershed_segmentation()
        bc.classify()

        contours = get_contours(bc.kde, bc.ws_labels)
        n = len(np.unique(bc.beh_labels)) - 1
        plt.figure(figsize = (4.85, 4.85))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)
        plt.title("KDE res = " + str(res[i]) + ", n = " + str(n))
        # plt.show()
        save_txt = "./figures/kde_resolution/" + name
        save_txt += "_kde_res_" + str(res[i]) + ".pdf"
        plt.savefig(save_txt, dpi = 600, bbox_inches = "tight")


def vary_num_freqencies(bc, name):
    """
    Compute behaviors varying the number of 
    frequencies used in the time-frequency analysis.
    """

    freqs = np.arange(3, 50, 3)

    for i in range(len(freqs)):
        bc.num_freq = freqs[i]
        bc.dj_post = 1/(bc.num_freq - 1)*np.log2(
                    bc.max_freq_post/bc.min_freq_post)
        bc.dj_face = 1/(bc.num_freq - 1)*np.log2(
                    bc.max_freq_face/bc.min_freq_face)
        bc.features_post = []
        bc.features_face = []
        bc.aligned_features = []

        # Train model
        bc.time_frequency_analysis()
        bc.standardize_features()
        bc.align_time_points()
        bc.pca(300000)
        bc.tsne()
        bc.pre_embedding()
        bc.kernel_density_estimation(100j)
        bc.watershed_segmentation()
        bc.classify()

        contours = get_contours(bc.kde, bc.ws_labels)
        n = len(np.unique(bc.beh_labels)) - 1
        plt.figure(figsize = (4.85, 4.85))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)
        plt.title("# frequncies = " + str(freqs[i]) + ", n = " + str(n))
        # plt.show()
        save_txt = "./figures/num_freqs/" + name
        save_txt += "_num_freqs_" + str(freqs[i]) + ".pdf"
        plt.savefig(save_txt, dpi = 600, bbox_inches = "tight")


def test_tsne_randomness():
    """
    Train model three times with the same parameters
    to check how much the final result differ
    """

    bc1 = load_model("./models/face_high_freq.pkl")
    bc2 = load_model("./models/face_high_freq.pkl")
    bc3 = load_model("./models/face_high_freq.pkl")

    # Retrain models with t-SNE
    bc1.tsne()
    bc1.pre_embedding()
    bc1.kernel_density_estimation(200j)
    bc1.watershed_segmentation()
    bc1.classify()
    bc2.tsne()
    bc2.pre_embedding()
    bc2.kernel_density_estimation(200j)
    bc2.watershed_segmentation()
    bc2.classify()
    bc3.tsne()
    bc3.pre_embedding()
    bc3.kernel_density_estimation(200j)
    bc3.watershed_segmentation()
    bc3.classify()

    
    contours1 = get_contours(bc1.kde, bc1.ws_labels)
    contours2 = get_contours(bc2.kde, bc2.ws_labels)
    contours3 = get_contours(bc3.kde, bc3.ws_labels)
    n1 = len(np.unique(bc1.beh_labels)) - 1
    n2 = len(np.unique(bc2.beh_labels)) - 1
    n3 = len(np.unique(bc3.beh_labels)) - 1
    plt.figure(figsize = (4.85, 2.2))
    plt.subplot(131)
    plot_watershed_heat(bc1.embedded, bc1.kde,
                        contours1, bc1.border)
    plt.title("n = " + str(n1))
    plt.subplot(132)
    plot_watershed_heat(bc2.embedded, bc2.kde,
                        contours2, bc2.border)
    plt.title("n = " + str(n2))
    plt.subplot(133)
    plot_watershed_heat(bc3.embedded, bc3.kde,
                        contours3, bc3.border)
    plt.title("n = " + str(n3))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # plt.show()
    save_txt = "./figures/test_tsne_randomness.pdf"
    plt.savefig(save_txt, dpi = 600, bbox_inches = "tight")


def perplexity_partial_labelling(bc, name):
    """
    Tune the whisker model perplexity
    value by plotting the partial labels
    in t-SNE plane for every value
    """

    perp = [10, 20, 30, 40, 50, 80, 100,
            130, 160, 200, 300]

    part_lab = get_partial_labels(bc)

    for i in range(len(perp)):
        # Train model
        bc.perp = perp[i]
        bc.tsne()
        bc.pre_embedding()
        bc.kernel_density_estimation(200j)
        bc.watershed_segmentation()
        bc.classify()

        contours = get_contours(bc.kde, bc.ws_labels)
        n = len(np.unique(bc.beh_labels)) - 1
        plt.figure(figsize = (4.85, 4.85))
        plot_partial_labels(bc, part_lab)
        plt.title("Perp = " + str(perp[i]) + ", n = " + str(n))
        # plt.show()
        save_txt = "./figures/partial_labelling/" + name
        save_txt += "_perp_" + str(perp[i]) + ".pdf"
        plt.savefig(save_txt, dpi = 600, bbox_inches = "tight")


def whisker_only_partial_labels():
    """
    Plot the partial labels for the whisker only model,
    without downsampling to 120Hz
    """
    
    bc = load_model("./models/whiskers_only.pkl")

    part_lab = get_partial_labels_whisk_only(bc)

    save_txt = "./figures/partial_labelling/whisker_only.png"
    plt.figure(figsize = (4.85, 4.85))
    plot_partial_labels(bc, part_lab)
    # plt.title("Model: " + "whisker_only")
    # plt.show()
    plt.savefig(save_txt, dpi = 1200, bbox_inches = "tight")


def save_partial_label_plot(bc, name):
    """
    Create partial label plot,
    not the whisker only model
    """

    part_lab = get_partial_labels(bc)

    save_txt = "./figures/partial_labelling/" + name + ".png"
    plt.figure(figsize = (4.85, 4.85))
    plot_partial_labels(bc, part_lab)
    # plt.title("Model:" + name)
    plt.savefig(save_txt, dpi = 1200, bbox_inches = "tight")


def save_time_in_behavior_plot(bc, name):
    """
    Plot colored in time in behavior
    """

    time_map = colored_time_map(bc)

    save_path = "./figures/time_in_behavior/" + name + ".pdf"

    plt.figure(figsize = (4.85, 4.85))
    plot_colored_tsne_map(bc, time_map)
    plt.title("Model: " + name)
    plt.savefig(save_path, dpi = 600, bbox_inches = "tight")


def analyse_model(bc, model_name):
    """
    Plot wanted plots, compute csv files,
    perform perplexity tuning, etc.

    Arguments:
        bc: Clustering pipeline
        model_name (str): Name used for storing plots etc.
    """

    print("Analysing model", model_name)

    # Create csv files
    print("Creating csv files")
    create_csv(bc, model_name)

    # Plot colored regions
    print("Plotting colored regions")
    characteristics(bc, True, model_name, "high")
    characteristics(bc, True, model_name, "low")
    characteristics(bc, False, model_name, "high")
    characteristics(bc, False, model_name, "low")
    
    save_partial_label_plot(bc, model_name)

    plt.figure(figsize = (4.85, 4.85))
    plot_behavior_labels(bc)
    plt.savefig("./figures/behavior_labels/" + model_name +
                "_labels.pdf", dpi = 600, bbox_inches = "tight")

    # print("Start perplexity tuning using partial labels")
    # perplexity_partial_labelling(bc, model_name)


def create_all_csv_files():
    create_csv(load_face(), "face")
    create_csv(load_post(), "post")
    create_csv(load_model("./models/face_high_freq.pkl"), "high_freq")
    create_csv(load_model("./models/separate_pca.pkl"), "sep_pca")
    create_csv(load_model("./models/face_narrow_freq.pkl"), "narrow_freq")
    create_csv(load_model("./models/face_pca_cutoff.pkl"), "cutoff_pca")


def analyse_all_models():
    analyse_model(load_face(), "face")
    analyse_model(load_post(), "post")
    analyse_model(load_model("./models/face_high_freq.pkl"), "high_freq")
    analyse_model(load_model("./models/separate_pca.pkl"), "sep_pca")
    analyse_model(load_model("./models/face_narrow_freq.pkl"), "narrow_freq")
    analyse_model(load_model("./models/face_pca_cutoff.pkl"), "cutoff_pca")
    

def plot_all_session_distributions():
    plot_session_distribution(load_face(), "face")
    plot_session_distribution(load_post(), "post")
    plot_session_distribution(load_model("./models/face_high_freq.pkl"), "high_freq")
    plot_session_distribution(load_model("./models/separate_pca.pkl"), "sep_pca")
    plot_session_distribution(load_model("./models/face_narrow_freq.pkl"), "narrow_freq")
    plot_session_distribution(load_model("./models/face_pca_cutoff.pkl"), "cutoff_pca")


def plot_all_partial_label_plots():
    save_partial_label_plot(load_face(), "face")
    save_partial_label_plot(load_post(), "post")
    save_partial_label_plot(load_model("./models/face_high_freq.pkl"), "high_freq")
    save_partial_label_plot(load_model("./models/separate_pca.pkl"), "sep_pca")
    save_partial_label_plot(load_model("./models/face_narrow_freq.pkl"), "narrow_freq")
    save_partial_label_plot(load_model("./models/face_pca_cutoff.pkl"), "cutoff_pca")


def plot_all_time_in_behavior():
    save_time_in_behavior_plot(load_face(), "face")
    save_time_in_behavior_plot(load_post(), "post")
    save_time_in_behavior_plot(load_model("./models/face_high_freq.pkl"), "high_freq")
    save_time_in_behavior_plot(load_model("./models/separate_pca.pkl"), "sep_pca")
    save_time_in_behavior_plot(load_model("./models/face_narrow_freq.pkl"), "narrow_freq")
    save_time_in_behavior_plot(load_model("./models/face_pca_cutoff.pkl"), "cutoff_pca")


def bandwidth_tuning_all():
    print("Tuning face")
    bandwidth_tuning(load_face(), "face")
    print("Tuning post")
    bandwidth_tuning(load_post(), "post")
    print("Tuning high_freq")
    bandwidth_tuning(load_model("./models/face_high_freq.pkl"), "high_freq")
    print("Tuning sep_pca")
    bandwidth_tuning(load_model("./models/separate_pca.pkl"), "sep_pca")
    bandwidth_tuning(load_model("./models/face_narrow_freq.pkl"), "narrow_freq")
    bandwidth_tuning(load_model("./models/face_pca_cutoff.pkl"), "cutoff_pca")
    
    
def pickle_all_models():
    pickle_post()
    pickle_face()
    pickle_high_freq()
    pickle_whisker_only()
    pickle_pca_cutoff()
    pickle_high_freq_50()
    pickle_narrow_freq()
    pickle_separate_pca()


def describe_all_pipelines():
    describe_pipeline(load_face(), "face")
    describe_pipeline(load_post(), "post")
    describe_pipeline(load_model("./models/face_high_freq.pkl"), "high_freq")
    describe_pipeline(load_model("./models/separate_pca.pkl"), "sep_pca")
    describe_pipeline(load_model("./models/face_narrow_freq.pkl"), "narrow_freq")
    describe_pipeline(load_model("./models/face_pca_cutoff.pkl"), "cutoff_pca")
    describe_pipeline(load_model("./models/whiskers_only.pkl"), "whisk_only")


def plot_all_summary_plots():
    plot_model_summary(load_face(), "face")
    plot_model_summary(load_post(), "post")
    plot_model_summary(load_model("./models/face_high_freq.pkl"), "high_freq")
    plot_model_summary(load_model("./models/separate_pca.pkl"), "sep_pca")
    plot_model_summary(load_model("./models/face_narrow_freq.pkl"), "narrow_freq")
    plot_model_summary(load_model("./models/face_pca_cutoff.pkl"), "cutoff_pca")
    plot_model_summary(load_model("./models/whiskers_only.pkl"), "whisk_only", True)


def model_tuning(bc, name):
    """
    Tune the model by varying both
    the bandwidth and perplexity 
    parameters. Plot partial labels
    for each model.
    """

    perp = np.linspace(20, 80, 7, endpoint = True)
    bandwidth = np.linspace(0.1, 0.15, 7, endpoint = True)

    part_lab = get_partial_labels(bc)

    for i in range(2, len(perp)):
        for j in range(len(bandwidth)):
            bc.perp = perp[i]
            bc.bw = bandwidth[j]

            # Train model
            bc.tsne()
            bc.pre_embedding()
            bc.kernel_density_estimation(500j)
            bc.watershed_segmentation()
            bc.classify()

            contours = get_contours(bc.kde, bc.ws_labels)
            n = len(np.unique(bc.beh_labels)) - 1
            plt.figure(figsize = (4.85, 4.85))
            plot_partial_labels(bc, part_lab)
            plt.title("Perp: " + str(perp[i]) + ", Bw: " + str(round(bandwidth[j],3)) + ", n: " + str(n))
            # plt.show()
            save_txt = "./figures/tuning/" + name
            save_txt += "_perp_" + str(perp[i])
            save_txt += "_bw_" + str(bandwidth[j]) +  ".png"
            plt.savefig(save_txt, dpi = 1200, bbox_inches = "tight")


def compute_persistence_diagrams(bc, name):
    """
    Trains the model for various bandwidth values,
    saving the persistence diagram for each one.
    """

    bandwidth = np.linspace(0.1, 0.15)
    bandwidth = np.round(bandwidth, 3)
    
    for i in range(len(bandwidth)):
        bc.bw = bandwidth[i]

        # Train model
        bc.tsne()
        bc.pre_embedding()
        bc.kernel_density_estimation(500j)
        bc.watershed_segmentation()
        bc.classify()

        pers, pers_trans = compute_persistence(bc.beh_labels)

        save_path = "./persistence_pairs/" + name
        save_path += "_bw_" + str(bandwidth[i]) + ".pkl"
        
        with open(save_path, "wb") as file:
            pkl.dump(pers_trans, file)

        path = "./figures/persistence_diagrams/"
        path += name + "_bw_" + str(bandwidth[i])
        path += ".png"

        title = "Bandwidth: " + str(bandwidth[i])

        save_persistence_diagram(pers, pers_trans, path,
                                 title)


def main():
    seaborn.set_theme(font = "serif", font_scale = 0.75)

    # pickle_data()
    explore_data()
    # bc = get_post_pipeline()
    
    # bandwidth_tuning()
    # vary_num_freqencies(load_model("./models/face_high_freq.pkl"),
    #                 "high_freq")


    
    # pickle_all_models()
    # create_all_csv_files()
    # plot_all_partial_label_plots()
    # whisker_only_partial_labels()
    # plot_all_time_in_behavior()
    # plot_all_session_distributions()
    # plot_all_time_in_behavior()
    # pickle_face()
    # describe_all_pipelines()
    
    # plot_all_summary_plots()
    # bandwidth_tuning_all()
    # analyse_all_models()

    # model_tuning(load_model("./models/separate_pca.pkl"), "sep_pca")

    # pickle_post_final()
    # pickle_ft_final()

    # describe_pipeline(load_model("./models/post_final.pkl"), "post_final")
    # describe_pipeline(load_model("./models/ft_final.pkl"), "ft_final")
    # plot_model_summary(load_model("./models/post_final.pkl"), "post_final")
    # plot_model_summary(load_model("./models/ft_final.pkl"), "ft_final")
    # save_time_in_behavior_plot(load_model("./models/post_final.pkl"), "post_final")
    # save_time_in_behavior_plot(load_model("./models/ft_final.pkl"), "ft_final")
    # analyse_model(load_model("./models/post_final.pkl"), "post_final")
    # analyse_model(load_model("./models/ft_final.pkl"), "ft_final")

    # compute_persistence_diagrams(load_face(), "face")

    # compute_persistence_diagrams(load_model("./models/post_final.pkl"), "post_final")
    # compute_persistence_diagrams(load_model("./models/ft_final.pkl"), "ft_final")


    

if __name__ == "__main__":
    main()


