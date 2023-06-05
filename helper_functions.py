import numpy as np
import scipy
import math
import vg
import pickle

from scipy.interpolate import LSQUnivariateSpline
from scipy.stats import gaussian_kde
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.markers import MarkerStyle

import numpy.random as rd


from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import cv2 as cv
import imutils

import pycwt as wavelet
from pycwt.helpers import find

from sklearn.manifold import TSNE

from gtda.homology import FlagserPersistence

import plotly.graph_objects as go

import persim

def estimate_pdf(array, bw, border, pixels):
    """
    Estimate the probability density
    of a 2D array.
    Applies a kernel density estimator
    with given bandwidth

    Parameters:
        array (np.ndarray(float)): Array to be 
            used for the estimation
        bw (float): Bandwidth used in the kde algorithm
        border (float): Padding around data to make
            watershed segmentation easier
        pixels (int): Pixelation along each axis
            to compute the pdf on

    Returns:
        kde (np.ndarray): Pdf estimated by kernel
            density estimation on a grid defined
            by xmin, ymin, dx, dy
        grid (list(np.ndarray)): List containing 
            information (X, Y) about the grid where 
            kde were applied
    """


    kernel = gaussian_kde(array.T, bw_method = bw)

    xmax, ymax = np.max(array, axis = 0) + border
    xmin, ymin = np.min(array, axis = 0) - border
    X, Y = np.mgrid[xmin:xmax:pixels, ymin:ymax:pixels]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde = np.reshape(kernel(positions).T, X.shape)
    kde = np.rot90(kde)
    # Truncate low values to zero, such that we get an
    # outer border
    kde[np.abs(kde) < 1e-5] = 0
    

    return kde, [X, Y]


def get_watershed_labels(image):
    """
    Performs watershed segmentation on
    an image.

    Parameters:
        image (ndarray): nxm image array where
            high values are far from the border.

    Returns:
        labels (np.ndarray): Label array
            given by the watershed segmentation
            algorithm
    """

    max_coords = peak_local_max(image)
    local_maxima = np.zeros_like(image, dtype = bool)
    local_maxima[tuple(max_coords.T)] = True
    markers = ndi.label(local_maxima)[0]
    labels = watershed(-image, markers, mask = image)

    return labels


def get_contours(image, labels):
    """
    Obtain the contours for an image,
    here a watershed segmentation.

    Parameters:
        image (ndarray): nxm image array where
            high values are far from the border.
        labels (ndarray): nxm array with the 
            labels found by watershed segmentation

    Returns:
        countours (ndarray): nxm array which is
            zero everywhere except at countours,
            where it is one.
    """

    unique_labels = np.unique(labels)
    # Create space for the contours
    contours = np.zeros(image.shape)

    
    # Iterate over the clusters
    for i in range(1, len(unique_labels)):
        # Create image only showing the single cluster
        mask = np.zeros(image.shape, dtype = "uint8")
        mask[labels == unique_labels[i]] = 255

        # Find the contours
        cnts = cv.findContours(mask.copy(),
                                cv.RETR_LIST,
                                cv.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)[0]
        
        for pair in cnts:
            contours[pair[0][0], pair[0][1]] = 1

    return contours.T



def assign_labels(data, labels, grid):
    """
    Classify data points to labels given by
    a watershed segmentation.

    Parameters:
        data (np.ndarray): Collection of 
            data points (t-SNE embeddings)
            to be classified.
        labels (np.ndarray): Label array
            given by the watershed segmentation
            algorithm
        grid (list(np.ndarry)): X and Y coordinates
            to which the kernel density estimation was
            applied.

    Returns:
        data_labels (np.ndarray): Array
            containing the assigned labels
    """

    # Rotate the watershed image 270 degrees 
    # to match the embedding coordinates
    labels = np.rot90(labels, 3)

    X = grid[0]
    Y = grid[1]

    # Create storage for the labels
    data_labels = np.zeros(len(data))

    # Iterate over every data point
    for i in range(len(data)):
        # Find the closest grid point
        xi = (np.abs(X[:,0] - data[i,0])).argmin()
        yi = (np.abs(Y[0,:] - data[i,1])).argmin()
        
        # Assign label
        data_labels[i] = int(labels[xi, yi])

    return data_labels

        
def plot_watershed_heat(data, image, contours, border):
    """
    Plots the segmented heatmat of the pdf obtained
    via kernel density estimation of the t-SNE
    embedding.

    Parameters:
        data (ndarray):
    """


    xmax, ymax = np.max(data, axis = 0) + border
    xmin, ymin = np.min(data, axis = 0) - border

    outside = np.ones(image.shape)
    outside[image == 0] = 0

    plt.imshow(image, cmap = "coolwarm", alpha = outside,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.imshow(np.zeros(contours.shape), alpha = contours,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.grid(None)



def spline_regression(y, dim, freq, knot_freq):
    """
    Performs spline regression on a time series y(t).
    The internal knots are centered to get close to 
    equal distance to the endpoints.

    Parameters:
        y (np.ndarray): The time series values
        dim (int): Order of the spline
        freq (float): Frequency of the time points.
        knot_freq (float): Chosen frequency of the
            internal knots.

    Returns:
        (np.ndarray): The regression curve at time 
            points t
    """

    t = np.arange(len(y))
    
    # Calculate the interior knots
    space = int(freq / knot_freq)
    rem = len(t) % space
    knots = np.arange(freq + rem // 2, len(t), space)
    
    spl = LSQUnivariateSpline(t, y, knots, k = dim)
    
    return spl(t)




def pickle_relevant_features(original_filepath, new_filepath):
    """
    Pickle the relevant features contained in the data
    found in the given filepath

    Parameters:
        filepath (string): path to the original pickled data
    """

    # Get the pickled data for one rat
    with open(original_filepath, "rb") as file:
        data = pickle.load(file)
    
    raw_features = extract_relevant_features(data)

    with open(new_filepath, "wb") as file:
        pickle.dump(raw_features, file)


def extract_relevant_features(data):
    """
    Collect the relevant time series which will be used 
    as raw features in the analysis.
    Features extracted:
     - Egocentric head actions relative to body in 3D:
        roll (X), pitch (Y), azimuth (Z)
     - Speed in the XY plane
     - Back angles: pitch (Y), azimuth (Z)
     - Sorted point data??


    Parameters:
        data (dict): All the provided data on one rat

    Returns:
        relevant_features (numpy.ndarray): 
            The relevant data
    """
    
    n_features = 7
    speeds = np.array(data["speeds"][:,2])
    ego3_rotm = np.array(data["ego3_rotm"])

    n_time_points = len(speeds)

    ego3q = np.zeros((n_time_points, 3))
    for i in range(n_time_points):
        ego3q[i,:] = rot2expmap(ego3_rotm[i,:,:])
    
    relevant_features = np.zeros((n_time_points, n_features))
    relevant_features[:,:3] = ego3q
    relevant_features[:,3] = speeds
    relevant_features[:,4:6] = data["back_ang"]
    relevant_features[:,6] = data["sorted_point_data"][:,4,2]
    
    return relevant_features


def rot2expmap(rot_mat):
    """
    Converts rotation matrix to quaternions
    Stolen from github
    """

    expmap = np.zeros(3)
    if np.sum(np.isfinite(rot_mat)) < 9:
        expmap[:] = np.nan
    else:
        d = rot_mat - np.transpose(rot_mat)
        if scipy.linalg.norm(d) > 0.01:
            r0 = np.zeros(3)
            r0[0] = -d[1, 2]
            r0[1] = d[0, 2]
            r0[2] = -d[0, 1]
            sintheta = scipy.linalg.norm(r0) / 2.
            costheta = (np.trace(rot_mat) - 1.) / 2.
            theta = math.atan2(sintheta, costheta)
            r0 = r0 / scipy.linalg.norm(r0)
        else:
            eigval, eigvec = scipy.linalg.eig(rot_mat)
            eigval = np.real(eigval)
            r_idx = np.argmin(np.abs(eigval - 1))
            r0 = np.real(eigvec[:, r_idx])
            theta = vg.angle(r0, np.dot(rot_mat, r0))

        theta = np.fmod(theta + 2*math.pi, 2*math.pi) # Remainder after dividsion (modulo operation)
        if theta > math.pi:
            theta = 2*math.pi - theta
            r0 = -r0
        expmap = r0*theta

    return expmap



def plot_scaleogram(power, t, freqs, scales):
    """
    Plots the spectrogram of the power density
    achived using a continuous wavelet transform

    Parameters:
        power (np.ndarray): Estimated power for all 
            time points across multiple scales
    """


    # Find appropriate contour levels
    min_level = np.nanmin(np.log2(power)) / 2
    max_level = np.nanmax([np.nanmax(np.log2(power)), 1])
    level_step = (max_level - min_level) / 9
    levels = np.arange(min_level, max_level + level_step, level_step)

    cnf = plt.contourf(t, np.log2(freqs), np.log2(power),
                 levels = levels, extend="both",
                 cmap=plt.cm.viridis)
    cbar = plt.colorbar(cnf)
    cbar.set_ticks(levels)
    cbar.set_ticklabels(np.char.mod("%.1e", 2.**levels))
    y_ticks = 2**np.arange(np.ceil(np.log2(freqs.min())),
                           np.ceil(np.log2(freqs).max()))
    ax = plt.gca()
    ax.set_yticks(np.log2(y_ticks))
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("Time [" + r'$s$' + "]")
    ax.set_ylabel("Frequency [Hz]")


def generate_simulated_data():
    """
    Generate simulated data to test the implementation
    choices. Each distinct behaviour is characterised
    by a set of features, each a superpostion of 
    sine waves corresponding frequencies and amplitudes.
    """

    rd.seed(666)

    # Number of distinct behaviours
    n_b = 10
    # Number of features
    n_f = 5
    # Number of sine-waves per feature
    n_s = 4
    # Length of recorded time series, in seconds
    t_max = 600
    # Capture framerate
    fr = 120
    # Example data
    data = np.zeros(shape = (fr * t_max, n_f))
    # Store labels
    labels = np.zeros(len(data))
    # Time points
    t = np.arange(t_max * fr) / fr
    # Expected length of a behaviour, in seconds
    length = 3
    # Lower frequency bound, Hz
    w_lower = 0.5
    # Upper frequency bound, Hz
    w_upper = 20
    # Mu amplitude parameter
    a_mu = 1 
    # Sigma amplitude parameter
    a_sig = 0.5
    # Determine the corresponding frequencies
    w = w_lower + (w_upper - w_lower) * rd.rand(n_b, n_f, n_s)
    # Determine the corresponding amplitudes
    a = rd.lognormal(mean = a_mu, sigma = a_sig, size = (n_b, n_f, n_s))
    # Gaussian noise added to the features
    noise_std = 0.2

    # Define feature generating function given amplitudes and frequencies
    def feature(freq, ampl, t):
        val = 0
        for i in range(len(freq)):
            val += ampl[i]*np.sin(2*np.pi*freq[i]*t)
        return val + rd.normal(0, noise_std)


    ## Simulate data
    # Choose behavioural changing times
    t_change = np.sort(rd.randint(0, len(data), int(len(data) / length / fr)))
    t_change = np.append(t_change, len(data))
    t_change = np.insert(t_change, 0, 0)
    # Choose behaviour labels
    behaviours = rd.randint(0, n_b, len(t_change))

    # Iterate through behaviours
    for i in range(1, len(t_change)):
        beh = behaviours[i]
        t_c = t_change[i]
        t_vals = t[t_change[i - 1]:t_change[i]]
        labels[t_change[i - 1]:t_change[i]] = np.ones(len(t_vals))*beh

        # Iterate over features
        for j in range(n_f):
            freq = w[beh, j, :]
            ampl = a[beh, j, :]
            temp = lambda time: feature(freq, ampl, time)
            data[t_change[i - 1]:t_change[i], j] = temp(t_vals)
            
    return data, labels, t_change, behaviours, w, t
        

def get_simulated_data():
    """
    Retrieve dictionary of simulated data
    """

    with open("./extracted_data/simulated_data.pkl", "rb") as file:
        df = pickle.load(file)

    return df


def pickle_simulated_data():
    """
    Generate simulated data and pickle it
    for easy retrieval elsewhere.
    """
    data, labels, t_change, behaviours, w, t = generate_simulated_data()
    df = {
            "data": data,
            "labels": labels,
            "t_change": t_change,
            "behaviours": behaviours,
            "freqencies": w,
            "time": t,
    }

    with open("./extracted_data/simulated_data.pkl", "wb") as file:
        pickle.dump(df, file)


def scale_power_spectrum(bc, sqrt = True, standardize = True):
    """
    Finds the power spectrum and scales it 
    corresponding to the variable scale:

    Arguments:
        bc (pipeline): Pipeline containing the data
        sqrt (bool): To take square root of the
            spectrum or not
        standardize (bool): To standardize the 
            spectrum or not
    Returns:
        embedding: t-SNE embedding found
    """
    bc.power = []
    bc.features = []

    # Iterate over animals
    for d in range(len(bc.data)):
        # Create storage for new feature vector
        x_d = np.zeros(shape = (len(bc.data[d]),
                                bc.n_features*
                                (bc.num_freq + 1)))
        # Create storage for power spectrum
        power_d = np.zeros(shape = (bc.n_features,
                                    bc.num_freq,
                                    len(bc.data[d])))

        # Iterate over raw features
        for i in range(bc.n_features):
            # Apply cwt
            wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
                    bc.data[d][:,i], bc.dt, bc.dj, 1/bc.max_freq,
                    bc.num_freq - 1, bc.mother)
            # Store the frequencies and scales
            if scales.all() != bc.scales.all():
                bc.scales = scales
                bc.freqs = freqs
            # Compute wavelet power spectrum
            power_i = np.abs(wave)**2
            # Normalize over scales (Liu et. al. 07)
            power_i /= scales[:, None]
            # Store power
            power_d[i] = power_i
            if sqrt:
                # Take the square root to ...
                power_i = np.sqrt(power_i)
            
            # Center and rescale trend
            trend = bc.trend[d][:,i]
            trend_std = np.std(trend)
            trend = (trend - np.mean(trend)) / trend_std


            # Store new features
            x_d[:,i] = trend
            x_d[:,bc.n_features + i*bc.num_freq:bc.n_features +(i + 1)*
                bc.num_freq] = power_i.T
    
        bc.features.append(x_d)
        bc.power.append(power_d)

    if standardize:
        bc.standardize_features()
    bc.pca()
    bc.tsne()

    return bc.embedded_train


def plot_methodology(bc):
    """
    Create plot showing the full methodology of
    a trained pipeline.
    """ 
    
    # Feature to plot
    feat = 1
    # Animal
    animal = 0
    # Time interval 
    time = [60*bc.capture_framerate, 120*bc.capture_framerate]
    # Data, 
    ts = bc.data[animal][time[0]:time[1],feat]
    detrend = bc.data_detrended[animal][time[0]:time[1],feat]
    trend = bc.trend[animal][time[0]:time[1],feat]
    # Scalogram
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
            bc.data[animal][:,feat], bc.dt, bc.dj, 1/bc.max_freq,
            bc.num_freq - 1, bc.mother)
    power = np.abs(wave)**2

    # Raw time series
    ax1 = plt.subplot(421)
    plt.plot(ts, c = "k", lw = 1)
    plt.plot(trend, c = "r", lw = 1)
    # Detrending
    ax2 = plt.subplot(423)
    plt.plot(detrend, c = "b", lw = 1)
    # Scaleogram
    ax3 = plt.subplot(222)
    t1 = 60*bc.capture_framerate
    t2 = 70*bc.capture_framerate
    t = np.arange(len(bc.data[animal]))/bc.capture_framerate
    plot_scaleogram(power[:,t1:t2], t[t1:t2], freqs, scales, ax3)
    # t-SNE embedding
    ax4 = plt.subplot(223)
    plt.scatter(bc.embedded[:,0], bc.embedded[:,1], s = 1)
    # Heatmap + segmentation
    ax5 = plt.subplot(224)
    contours = get_contours(bc.kde, bc.ws_labels)
    plot_watershed_heat(bc.embedded, bc.kde, contours,
                        bc.border - 15)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplots_adjust(hspace = 0.1, wspace = 0.1)


def describe_pipeline(bc, model_name):
    """
    Description of a trained clustering
    pipeline. Print descriptive statistics
    and plot relevant results.
    """

    print(model_name)
    print()

    print("POSTURAL FEATURES")
    print(f"Number of animals: {len(bc.data_post)}")
    print(f"Number of features: {bc.n_features_post}")
    total_timepoints = np.sum([len(d) for d in bc.raw_features_post])
    nonan_timepoints = np.sum([len(d) for d in bc.data_post])
    print(f"Number of input time points: {total_timepoints}")
    print(f"Non-NaN time points: {nonan_timepoints}")
    print(f"Number of NaN-timepoints: {total_timepoints - nonan_timepoints}")
    print(f"Average length of series: {nonan_timepoints/len(bc.data_post)/bc.capture_framerate_post} seconds")
    print(f"Capture framerate: {bc.capture_framerate_post} Hz")
    print(f"Training points for t-SNE: {len(bc.embedded_train)}")
    print(f"Downsampling rate: {bc.ds_rate} Hz")
    print(f"Number of found behaviors: {len(np.unique(bc.ws_labels - 1))}")
    print(f"Number of principal components used: {bc.n_pca}")

    print()
    if bc.face:
        print("FACIAL TRACKING FEATURES")
        print(f"Number of animals: {len(bc.data_face)}")
        print(f"Number of features: {bc.n_features_face}")
        total_timepoints = np.sum([len(d) for d in bc.raw_features_face])
        nonan_timepoints = np.sum([len(d) for d in bc.data_face])
        print(f"Number of input time points: {total_timepoints}")
        print(f"Non-NaN time points: {nonan_timepoints}")
        print(f"Number of NaN-timepoints: {total_timepoints - nonan_timepoints}")
        print(f"Average length of series: {nonan_timepoints/len(bc.data_face)/bc.capture_framerate_face} seconds")
        print(f"Capture framerate: {bc.capture_framerate_face} Hz")
        print(f"Training points for t-SNE: {len(bc.embedded_train)}")
        print(f"Downsampling rate: {bc.ds_rate} Hz")
        print(f"Number of found behaviors: {len(np.unique(bc.ws_labels - 1))}")
        print(f"Number of principal components used: {bc.n_pca}")

    print()


def perplexity_tuning(bc):
    """
    Tries a series of perplexity values on 
    an already trained pipeline.
    Plots the result.
    """
    
    # Perplexity values to be tested
    perp = [5, 30, 50, 100, 200, 500]

    # Iterate over chosen perplexities
    for i in range(len(perp)):
        bc.perp = perp[i]
        bc.tsne()
        bc.pre_embedding()
        bc.kernel_density_estimation(300j)
        bc.watershed_segmentation()

        contours = get_contours(bc.kde, bc.ws_labels)
        
        plt.subplot(2, 3, i + 1)
        plt.title("Perplexity = " + str(perp[i]))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)


def plot_ethogram(behavior_labels, t):
    """
    Plot movements between different behaviors
    """

    plt.plot(t, behavior_labels)
    

def get_trans_prob_mat(states):
    """
    Compute and return the one step 
    transition probability matrix.

    Arguments:
        states (list(int)): List type object with states
            in chronological order. States are indexed
            1,2,...
    """


    n_beh = len(np.unique(states))
    tmat = np.zeros(shape = (n_beh, n_beh))

    for (i,j) in zip(states, states[1:]):
        tmat[int(i-1), int(j-1)] += 1

    # Normalize rows
    tmat = tmat / np.linalg.norm(tmat, axis = 1,
                                 ord = 1, keepdims = True)

    return tmat


def plot_cum_pca(ratio):
    """
    Plot the cumulative explained variance
    of the principal components.

    Arguments:
        ratio (np.ndarray): Array of decreasing
           ordinal values representing ratio
           of explained variance
    """

    cum = np.cumsum(ratio)
    n = np.arange(1,len(cum) + 1)
    cutoff = np.argmax(cum > 0.95)
    plt.scatter(n, cum, marker = ".", s = 1)
    plt.xlabel("# included PCs")
    plt.ylabel("Explained variance")
    # plt.vlines(cutoff, np.min(cum), np.max(cum),
               # color = "red", linestyle = "dashed",
               # lw = 1)
    plt.axvline(x = cutoff, color = "red", linestyle = "dashed",
                lw = 1)
    plt.axhline(y = 0.95, color = "red", linestyle = "dashed",
                lw = 1)
    # plt.hlines(0.95, np.min(n), np.max(n),
               # colors = "red", linestyle = "dashed",
               # lw = 1)
    

def plot_pca_embedding(bc, n1, n2):
    """
    Plot the points in the 2-dimensional
    plane given by the first two principal 
    components. Color points by their 
    classification.

    Arguments:
        bc (pipeline): Trained behavioral clustering
            model
        n1 (int): First PC
        n2 (int): Second PC
    """

    feat = bc.fit_pca

    pc1 = feat[:,n1-1]
    pc2 = feat[:,n2-1]
    labels = bc.beh_labels.astype(int)

    # Downsample (only show part of the points)
    ind = np.arange(0, len(pc1), len(pc1) // 5000)

    scatter = plt.scatter(pc1[ind], pc2[ind], c = labels[ind],
                s = 1, cmap = "Paired")
    plt.legend(*scatter.legend_elements(num=10))
    plt.xlabel("PC" + str(n1))
    plt.ylabel("PC" + str(n2))


def plot_colored_tsne_map(bc, char, hide_tics = False,
                          col_bar = True):
    """
    Plots the t-SNE mapping where each
    region is colored by degree of 
    some characteristic specified 
    in func.

    Arguments:
        bc: Behavioral pipeline (trained)
        char (np.ndarray): Pre-computed characteristics
            for each regions which will be plotted
    """

    contours = get_contours(bc.kde, bc.ws_labels)

    outside = np.ones(bc.kde.shape)
    outside[bc.kde == 0] = 0

    xmax, ymax = np.quantile(bc.embedded, 0.97, axis = 0) + bc.border
    xmin, ymin = np.quantile(bc.embedded, 0.03, axis = 0) - bc.border
    
    pos = plt.imshow(char, cmap = "YlOrRd", alpha = outside,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.imshow(np.zeros(contours.shape), alpha = contours,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.grid(None)
    if hide_tics:
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    if col_bar:
        plt.colorbar(pos, shrink = 0.85)


def get_average_value(bc, label, feat_num, face):
    """
    Returns the average value for a feature in a certain 
    region.

    Arguments:
        bc: Behavioral pipeline
        label (int): Which region to analyze
        feat_num (int): Which feature to analyze
        face (bool): True if whisker feature
    """

    if face: 
        feat = np.concatenate(bc.aligned_features,
                              axis = 0)
    else:
        feat = np.concatenate(bc.features_post,
                              axis = 0)

    # Find timepoints in given region
    ind = np.asarray(bc.beh_labels == label).nonzero()
    reg_feat = feat[ind]

    # Retrieve relevant feature values
    reg_val = reg_feat[:,feat_num] 

    return np.mean(reg_val)


def get_average_power(bc, label, feat_num, face, freq_range):
    """
    Returns the average power for a feature in a certain 
    region.

    Arguments:
        bc: Behavioral pipeline
        label (int): Which region to analyze
        feat_num (int): Which feature to analyze
        face (bool): True if whisker feature
        freq_range (str): "high"/"low". Power from high 
            or low frequencies, divided in the middle
    """

    if face: 
        feat = np.concatenate(bc.aligned_features,
                              axis = 0)
    else:
        feat = np.concatenate(bc.features_post,
                              axis = 0)

    # Find timepoints in given region
    ind = np.asarray(bc.beh_labels == label).nonzero()
    reg_feat = feat[ind]

    # Retrieve relevant feature power spectrum
    if face:
        reg_spec = reg_feat[:,bc.n_features_face + 
                            feat_num*bc.num_freq:
                            bc.n_features_face + 
                            (feat_num + 1)*bc.num_freq]
    else:
        reg_spec = reg_feat[:,bc.n_features_post + 
                            feat_num*bc.num_freq:
                            bc.n_features_post + 
                            (feat_num + 1)*bc.num_freq]
    # Divide in half
    mid = reg_spec.shape[1] // 2
    if freq_range == "high":
        reg_spec = reg_spec[:,mid:]
    else:
        reg_spec = reg_spec[:,:mid]

    return np.mean(reg_spec)


def get_time_in_behavior(bc, label):
    """
    Returns the proportion of time spent in a
    given behavior.

    Arguments:
        bc: Behavioral pipeline
        label (int): Behavior label
    """

    # Find timepoints in given region
    ind = np.asarray(bc.beh_labels == label).nonzero()
    reg_feat = bc.beh_labels[ind]

    return len(reg_feat) / len(bc.beh_labels)


def colored_time_map(bc):
    """
    Retrieve full colored in t-SNE
    mapping showing proportion of time
    spent in each behavior.
    """

    labels = np.unique(bc.beh_labels)
    time_map = {0.0: 0.0}
    for lab in labels:
        time_map[lab] = get_time_in_behavior(bc, lab)

    prop = bc.ws_labels.astype(object)
    prop = np.vectorize(time_map.get)(prop)

    return prop

    
def plot_partial_labels(bc, part_lab):
    """
    Plot segmented heatmap in t-SNE plane with
    partial labels shown.
    """

    ind = np.nonzero(part_lab)

    contours = get_contours(bc.kde, bc.ws_labels)

    outside = np.ones(bc.kde.shape)
    outside[bc.kde == 0] = 0

    xmax, ymax = np.max(bc.embedded, axis = 0) + bc.border
    xmin, ymin = np.min(bc.embedded, axis = 0) - bc.border
    
    pos = plt.imshow(bc.kde, cmap = "coolwarm", alpha = outside * 0.5,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.imshow(np.zeros(contours.shape), alpha = contours,
               extent = [xmin, xmax, ymin, ymax]) 

    css_col = mcolors.CSS4_COLORS
    col = [css_col["maroon"], css_col["tomato"], css_col["darkorange"],
           css_col["forestgreen"], css_col["gold"], css_col["royalblue"],
           css_col["dodgerblue"], css_col["darkorchid"], css_col["magenta"],
           css_col["hotpink"], css_col["grey"], css_col["silver"]]
    mark = [".", "o", "v", "1", "s", "P",
            "*", "+", "X", "D", "x", "p"]
    behaviors = np.unique(part_lab)[1:].astype(int)
    labels = ["Rearing wall 1", "Rearing wall 2", 
              "Rearing open", "Eating", "Grooming face",
              "Freeze 1", "Freeze 2", "Touching object 1",
              "Touching object 2", "Touching object 3",
              "Running across 1", "Running across 2"]

    for beh in behaviors:
        # Find corresponding indices
        ind = np.asarray(part_lab == beh).nonzero()
        plt.scatter(bc.embedded[ind,0], bc.embedded[ind,1], c = col[beh-1],
                    marker = mark[beh-1], s = 9,
                    label = labels[beh-1])

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.legend(fontsize = 6)
    ax = plt.gca()
    label_params = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    plt.grid(None)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig = plt.gcf()

    # Plot legend in separate plot
    figl, axl = plt.subplots()
    axl.axis(False)
    axl.legend(*label_params, bbox_to_anchor=(0.5, 0.5),
               loc="center", markerscale = 2)
    figl.savefig("./figures/legend.pdf", bbox_inches = "tight")
    # Return to previous figure
    plt.figure(fig)


def get_partial_labels(bc):
    """
    Assign labels to some of the data.
    0: No assigned behavior
    1: Rearing wall 1—Simple whisking upward
    2: Rearing wall 2—More complex whisking 
    3: Rearing open
    4: Eating
    5: Grooming face
    6: Freeze 1—Short freeze
    7: Freeze 1—Long pause, some movement
    8: Touching object 1—Mostly L whisker engaged
    9: Touching object 1—Both whiskers engaged
    10: Touching object 1—Both whiskers engaged + 
        animal climbing on/around
    11: Running across 1—Whiskers forward, medium speed
    12: Running across 2—Whiskers forward, fast jumpy run


    Returns np.ndarray with the assigned
    labels on the same dimension as 
    bc.beh_labels
    """

    # Create initial space for the partial labels
    part_lab = []
    for i in range(len(bc.used_indices_post)):
        part_lab.append(np.zeros(len(bc.used_indices_post[i])))

    # Manually insert labels for rearing wall
    part_lab[0][15200:15300] = 1
    part_lab[0][17740:18140] = 2
    part_lab[0][32870:33290] = 2
    part_lab[1][47255:47454] = 1
    part_lab[2][8995:9235] = 1
    part_lab[2][26935:27465] = 2
    part_lab[2][55005:56105] = 2
    part_lab[3][17240:17410] = 1
    part_lab[3][11964:12150] = 2
    part_lab[4][7800:7995] = 1
    part_lab[9][14370:14695] = 1
    part_lab[9][1960:2285] = 2
    part_lab[16][835:1290] = 1
    part_lab[19][19034:19205] = 1
    part_lab[19][19330:19540] = 1
    part_lab[19][67270:67440] = 1
    part_lab[19][6430:6830] = 2
    part_lab[19][21690:22245] = 2
    part_lab[19][26225:26850] = 2
    # Manually insert labels for rearing open
    part_lab[1][38775:38860] = 3
    part_lab[9][16970:17440] = 3
    part_lab[19][1160:1385] = 3
    # Manually insert labels for eating
    part_lab[0][430:13300] = 4
    part_lab[1][15815:27800] = 4
    part_lab[2][295:5915] = 4
    part_lab[2][30885:35415] = 4
    part_lab[3][59600:61558] = 4
    part_lab[4][35760:42680] = 4
    part_lab[9][48114:51985] = 4
    part_lab[16][11680:16350] = 4
    part_lab[16][16960:19380] = 4
    part_lab[19][29955:35330] = 4
    # Manually insert labels for grooming face
    part_lab[0][15840:17640] = 5
    part_lab[0][19320:21660] = 5
    part_lab[0][41760:44040] = 5
    part_lab[1][29950:31600] = 5
    part_lab[2][11685:13375] = 5
    part_lab[2][45455:46305] = 5
    part_lab[2][46555:46995] = 5
    part_lab[3][4860:6262] = 5
    part_lab[4][11845:12985] = 5
    part_lab[4][28455:32200] = 5
    part_lab[9][6845:7822] = 5
    part_lab[16][6530:8180] = 5
    part_lab[19][15380:16015] = 5
    part_lab[19][38600:39210] = 5
    # Manually insert labels for freeze 
    part_lab[0][25880:26220] = 6
    part_lab[0][26395:26810] = 6
    part_lab[1][12990:13220] = 6
    part_lab[1][36271:36349] = 6
    part_lab[2][15493:15655] = 6
    part_lab[2][16145:16265] = 6
    part_lab[2][22765:24985] = 7
    part_lab[2][57170:58305] = 7
    part_lab[3][13600:13645] = 6
    part_lab[3][57366:57620] = 6
    part_lab[3][12449:12878] = 7
    part_lab[3][17650:18567] = 7
    part_lab[4][447:830] = 6
    part_lab[9][27270:27860] = 7
    part_lab[16][5180:5230] = 6
    part_lab[19][16840:17015] = 6
    part_lab[19][27190:27345] = 6
    # Manually insert labels for touching object
    part_lab[2][6550:6587] = 8
    part_lab[2][6410:6490] = 9
    part_lab[2][7682:7962] = 9
    part_lab[2][28183:28229] = 9
    part_lab[2][6654:6790] = 10
    part_lab[2][27710:27950] = 10
    part_lab[9][34240:34390] = 9 
    part_lab[9][54445:54489] = 9
    part_lab[9][54574:54725] = 9
    part_lab[9][34425:35400] = 10
    part_lab[16][38920:39205] = 9
    part_lab[16][58480:58617] = 10
    part_lab[19][53510:54690] = 8
    part_lab[19][36530:36645] = 9
    part_lab[19][37115:37150] = 9
    part_lab[19][52615:53440] = 9
    part_lab[19][28995:29680] = 10
    part_lab[19][37265:37410] = 10
    # Manually insert labels for running across
    part_lab[0][24060:24170] = 11
    part_lab[0][56710:56850] = 11
    part_lab[1][37640:37800] = 11
    part_lab[1][38040:38133] = 12
    part_lab[1][44825:45045] = 12
    part_lab[2][25515:25635] = 11
    part_lab[2][43995:44085] = 12
    part_lab[2][50665:50775] = 12
    part_lab[3][14110:14225] = 11
    part_lab[4][5255:5400] = 12
    part_lab[9][16670:16815] = 11
    part_lab[9][46547:46795] = 11
    part_lab[16][59713:59855] = 11
    part_lab[19][9840:10090] = 11
    part_lab[19][28740:28870] = 11
    part_lab[19][47900:48035] = 12

    # Filter the data
    for i in range(len(part_lab)):
        part_lab[i] = part_lab[i][bc.used_indices_post[i]]

    # Concatenate to desired dimension
    part_lab = np.concatenate(part_lab, axis = 0)

    return part_lab


def get_partial_labels_whisk_only(bc):
    """
    Assign labels to some of the data.
    0: No assigned behavior
    1: Rearing wall 1—Simple whisking upward
    2: Rearing wall 2—More complex whisking 
    3: Rearing open
    4: Eating
    5: Grooming face
    6: Freeze 1—Short freeze
    7: Freeze 1—Long pause, some movement
    8: Touching object 1—Mostly L whisker engaged
    9: Touching object 1—Both whiskers engaged
    10: Touching object 1—Both whiskers engaged + 
        animal climbing on/around
    11: Running across 1—Whiskers forward, medium speed
    12: Running across 2—Whiskers forward, fast jumpy run

    Returns np.ndarray with the assigned
    labels on the same dimension as 
    bc.beh_labels
    """

    # Create initial space for the partial labels
    part_lab = []
    for i in range(len(bc.used_indices_face)):
        part_lab.append(np.zeros(len(bc.used_indices_face[i])))

    cp = bc.capture_framerate_face / bc.capture_framerate_post

    # Manually insert labels for rearing wall
    part_lab[0][int(15200*cp):int(15300*cp)] = 1
    part_lab[0][int(17740*cp):int(18140*cp)] = 2
    part_lab[0][int(32870*cp):int(33290*cp)] = 2
    part_lab[1][int(47255*cp):int(47454*cp)] = 1
    part_lab[2][int(8995*cp):int(9235*cp)] = 1
    part_lab[2][int(26935*cp):int(27465*cp)] = 2
    part_lab[2][int(55005*cp):int(56105*cp)] = 2
    part_lab[3][int(17240*cp):int(17410*cp)] = 1
    part_lab[3][int(11964*cp):int(12150*cp)] = 2
    part_lab[4][int(7800*cp):int(7995*cp)] = 1
    part_lab[9][int(14370*cp):int(14695*cp)] = 1
    part_lab[9][int(1960*cp):int(2285*cp)] = 2
    part_lab[16][int(835*cp):int(1290*cp)] = 1
    part_lab[19][int(19034*cp):int(19205*cp)] = 1
    part_lab[19][int(19330*cp):int(19540*cp)] = 1
    part_lab[19][int(67270*cp):int(67440*cp)] = 1
    part_lab[19][int(6430*cp):int(6830*cp)] = 2
    part_lab[19][int(21690*cp):int(22245*cp)] = 2
    part_lab[19][int(26225*cp):int(26850*cp)] = 2
    # Manually insert labels for rearing open
    part_lab[1][int(38775*cp):int(38860*cp)] = 3
    part_lab[9][int(16970*cp):int(17440*cp)] = 3
    part_lab[19][int(1160*cp):int(1385*cp)] = 3
    # Manually insert labels for eating
    part_lab[0][int(430*cp):int(13300*cp)] = 4
    part_lab[1][int(15815*cp):int(27800*cp)] = 4
    part_lab[2][int(295*cp):int(5915*cp)] = 4
    part_lab[2][int(30885*cp):int(35415*cp)] = 4
    part_lab[3][int(59600*cp):int(61558*cp)] = 4
    part_lab[4][int(35760*cp):int(42680*cp)] = 4
    part_lab[9][int(48114*cp):int(51985*cp)] = 4
    part_lab[16][int(11680*cp):int(16350*cp)] = 4
    part_lab[16][int(16960*cp):int(19380*cp)] = 4
    part_lab[19][int(29955*cp):int(35330*cp)] = 4
    # Manually insert labels for grooming face
    part_lab[0][int(15840*cp):int(17640*cp)] = 5
    part_lab[0][int(19320*cp):int(21660*cp)] = 5
    part_lab[0][int(41760*cp):int(44040*cp)] = 5
    part_lab[1][int(29950*cp):int(31600*cp)] = 5
    part_lab[2][int(11685*cp):int(13375*cp)] = 5
    part_lab[2][int(45455*cp):int(46305*cp)] = 5
    part_lab[2][int(46555*cp):int(46995*cp)] = 5
    part_lab[3][int(4860*cp):int(6262*cp)] = 5
    part_lab[4][int(11845*cp):int(12985*cp)] = 5
    part_lab[4][int(28455*cp):int(32200*cp)] = 5
    part_lab[9][int(6845*cp):int(7822*cp)] = 5
    part_lab[16][int(6530*cp):int(8180*cp)] = 5
    part_lab[19][int(15380*cp):int(16015*cp)] = 5
    part_lab[19][int(38600*cp):int(39210*cp)] = 5
    # Manually insert labels for freeze 
    part_lab[0][int(25880*cp):int(26220*cp)] = 6
    part_lab[0][int(26395*cp):int(26810*cp)] = 6
    part_lab[1][int(12990*cp):int(13220*cp)] = 6
    part_lab[1][int(36271*cp):int(36349*cp)] = 6
    part_lab[2][int(15493*cp):int(15655*cp)] = 6
    part_lab[2][int(16145*cp):int(16265*cp)] = 6
    part_lab[2][int(22765*cp):int(24985*cp)] = 7
    part_lab[2][int(57170*cp):int(58305*cp)] = 7
    part_lab[3][int(13600*cp):int(13645*cp)] = 6
    part_lab[3][int(57366*cp):int(57620*cp)] = 6
    part_lab[3][int(12449*cp):int(12878*cp)] = 7
    part_lab[3][int(17650*cp):int(18567*cp)] = 7
    part_lab[4][int(447*cp):int(830*cp)] = 6
    part_lab[9][int(27270*cp):int(27860*cp)] = 7
    part_lab[16][int(5180*cp):int(5230*cp)] = 6
    part_lab[19][int(16840*cp):int(17015*cp)] = 6
    part_lab[19][int(27190*cp):int(27345*cp)] = 6
    # Manually insert labels for touching object
    part_lab[2][int(6550*cp):int(6587*cp)] = 8
    part_lab[2][int(6410*cp):int(6490*cp)] = 9
    part_lab[2][int(7682*cp):int(7962*cp)] = 9
    part_lab[2][int(28183*cp):int(28229*cp)] = 9
    part_lab[2][int(6654*cp):int(6790*cp)] = 10
    part_lab[2][int(27710*cp):int(27950*cp)] = 10
    part_lab[9][int(34240*cp):int(34390*cp)] = 9 
    part_lab[9][int(54445*cp):int(54489*cp)] = 9
    part_lab[9][int(54574*cp):int(54725*cp)] = 9
    part_lab[9][int(34425*cp):int(35400*cp)] = 10
    part_lab[16][int(38920*cp):int(39205*cp)] = 9
    part_lab[16][int(58480*cp):int(58617*cp)] = 10
    part_lab[19][int(53510*cp):int(54690*cp)] = 8
    part_lab[19][int(36530*cp):int(36645*cp)] = 9
    part_lab[19][int(37115*cp):int(37150*cp)] = 9
    part_lab[19][int(52615*cp):int(53440*cp)] = 9
    part_lab[19][int(28995*cp):int(29680*cp)] = 10
    part_lab[19][int(37265*cp):int(37410*cp)] = 10
    # Manually insert labels for running across
    part_lab[0][int(24060*cp):int(24170*cp)] = 11
    part_lab[0][int(56710*cp):int(56850*cp)] = 11
    part_lab[1][int(37640*cp):int(37800*cp)] = 11
    part_lab[1][int(38040*cp):int(38133*cp)] = 12
    part_lab[1][int(44825*cp):int(45045*cp)] = 12
    part_lab[2][int(25515*cp):int(25635*cp)] = 11
    part_lab[2][int(43995*cp):int(44085*cp)] = 12
    part_lab[2][int(50665*cp):int(50775*cp)] = 12
    part_lab[3][int(14110*cp):int(14225*cp)] = 11
    part_lab[4][int(5255*cp):int(5400*cp)] = 12
    part_lab[9][int(16670*cp):int(16815*cp)] = 11
    part_lab[9][int(46547*cp):int(46795*cp)] = 11
    part_lab[16][int(59713*cp):int(59855*cp)] = 11
    part_lab[19][int(9840*cp):int(10090*cp)] = 11
    part_lab[19][int(28740*cp):int(28870*cp)] = 11
    part_lab[19][int(47900*cp):int(48035*cp)] = 12

    # Filter the data
    for i in range(len(part_lab)):
        part_lab[i] = part_lab[i][bc.used_indices_face[i]]

    # Concatenate to desired dimension
    part_lab = np.concatenate(part_lab, axis = 0)

    return part_lab


def plot_behavior_labels(bc):
    """
    Shows the found behavior number found 
    plotted over the t-SNE map, showing
    where the behaviors lie in the plane.
    """

    contours = get_contours(bc.kde, bc.ws_labels)

    outside = np.ones(bc.kde.shape)
    outside[bc.kde == 0] = 0

    xmax, ymax = np.max(bc.embedded, axis = 0) + bc.border
    xmin, ymin = np.min(bc.embedded, axis = 0) - bc.border

    
    pos = plt.imshow(bc.kde, cmap = "coolwarm", alpha = outside * 0.5,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.imshow(np.zeros(contours.shape), alpha = contours,
               extent = [xmin, xmax, ymin, ymax]) 

    # Rotate the watershed image 270 degrees 
    # to match the embedding coordinates
    labels = np.rot90(bc.ws_labels, 3)

    for beh_num in np.unique(labels)[1:]:
        ind = np.nonzero(labels == beh_num)
        x_ind = np.mean(ind[0])
        y_ind = np.mean(ind[1])
        beta_x = x_ind / (len(labels) - x_ind)
        beta_y = y_ind / (len(labels) - y_ind)
        x = 1 / (1 + beta_x) * (beta_x * xmax + xmin)
        y = 1 / (1 + beta_y) * (beta_y * ymax + ymin)
        plt.text(x, y, str(beh_num), fontsize = 8,
                 horizontalalignment = "center",
                 verticalalignment = "center")

    
    plt.grid(None)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # plt.show()


def create_csv(bc, model_name):
    """
    Create csv files for each behavior and 
    session, showing the start and end frames.
    This can be used with the Fiji GUI to
    check how the behaviors look.

    Arguments:
        bc: Clustering pipeline
        model_name (str): Name used for storing
            csv files
    """
    
    # Number of sessions
    ns = len(bc.train_file_names_post)
    # Session names
    sessions = [s.split("_")[0] + "_" + s.split("_")[1]
                for s in bc.train_file_names_post]

    # Iterate over sessions
    start_ind = 0
    for i in range(ns):
        # Total number of frames
        nf = len(bc.raw_features_post[i])
        # Retrive relevant behavioral labels
        end_ind = start_ind + len(bc.data_post[i])
        emb = bc.beh_labels[start_ind:end_ind]
        used = np.argwhere(bc.used_indices_post[i])
        start_ind = end_ind

        # Create storage for intervals
        intervals = [[] for _ in range(len(np.unique(bc.beh_labels)))]

        # Iterate over frames
        beh = int(emb[0])
        start_frame = 0
        for j in range(1, len(emb)):
            # Check if we have a change
            if emb[j] != emb[j-1]:
                # Find the original frames
                # corresponding to start/stop
                start = int(used[start_frame])
                end = int(used[j-1])
                # Save the interval
                intervals[beh-1].append([start, end])
                # Update to new behavior interval
                beh = int(emb[j])
                start_frame = j

        # Write to csv file                
        for j in range(1, len(np.unique(bc.beh_labels))):
            path = "./csv_files/" + model_name
            path += "_" + sessions[i]
            path += "_beh_" + str(j) + ".csv" 
            np.savetxt(path, intervals[j-1], fmt="%d", delimiter=",")


def plot_session_distribution(bc, model_name):
    """
    Plot relative prevalence of behavior in each 
    behavioral region. 
    """

    # Number of sessions
    ns = len(bc.train_file_names_post)
    # Session names
    sessions = [s.split("_")[0] + "_" + s.split("_")[1]
                for s in bc.train_file_names_post]

    # Iterate over sessions
    start_ind = 0
    for i in range(ns):
        # Retrive relevant behavioral labels
        end_ind = start_ind + len(bc.data_post[i])
        emb = bc.beh_labels[start_ind:end_ind]
        start_ind = end_ind

        labels = np.unique(bc.ws_labels)
        # Compute average time spent in 
        # each behavior in session i,
        # relative to total average
        rel_prev = {0.0: 0.0}
        for lab in labels[1:]:
            # This session
            ind_ses = np.asarray(emb == lab).nonzero()
            reg_feat_ses = emb[ind_ses]
            rel_time_ses = len(reg_feat_ses) / len(emb)
            # Total
            ind_tot = np.asarray(bc.beh_labels == lab).nonzero()
            reg_feat_tot = bc.beh_labels[ind_tot]
            rel_time_tot = len(reg_feat_tot) / len(bc.beh_labels)

            rel_prev[lab] = rel_time_ses / rel_time_tot

        prop = bc.ws_labels.astype(object)
        prop = np.vectorize(rel_prev.get)(prop)

        # Plot the relative prevalence in t-SNE plane

        plt.figure(figsize = (4.85, 4.85))
        plot_colored_tsne_map(bc, prop)
        plt.title("Session " + sessions[i])

        labels = np.rot90(bc.ws_labels, 3)
        xmax, ymax = np.quantile(bc.embedded, 0.97, axis = 0) + bc.border
        xmin, ymin = np.quantile(bc.embedded, 0.03, axis = 0) - bc.border
    
        for beh_num in np.unique(labels)[1:]:
            ind = np.nonzero(labels == beh_num)
            x_ind = np.mean(ind[0])
            y_ind = np.mean(ind[1])
            beta_x = x_ind / (len(labels) - x_ind)
            beta_y = y_ind / (len(labels) - y_ind)
            x = 1 / (1 + beta_x) * (beta_x * xmax + xmin)
            y = 1 / (1 + beta_y) * (beta_y * ymax + ymin)
            plt.text(x, y, str(beh_num), fontsize = "x-small",
                     horizontalalignment = "center",
                     verticalalignment = "center")
    
        # plt.show()
        path = "./figures/session_distribution/" + model_name
        path += "_" + sessions[i]
        path += "_session_dist.pdf" 
        plt.savefig(path, bbox_inches = "tight")
        plt.close()


def plot_model_summary(bc, model_name, face = False):
    """
    Create pretty plot to be used for displaying
    the various models in the thesis.
    Including a pure heatmap, partial labelling,
    time in behavior, and numbering.
    """

    contours = get_contours(bc.kde, bc.ws_labels)

    time_map = colored_time_map(bc)

    if face:
        part_lab = get_partial_labels_whisk_only(bc)
    else:
        part_lab = get_partial_labels(bc)
    
    # Create a 2x2 subplot structure
    fig = plt.figure(figsize = (4.85, 4.7))
    ax1 = plt.subplot(221)
    plot_watershed_heat(bc.embedded, bc.kde,
                        contours, bc.border)

    ax2 = plt.subplot(222)
    plot_colored_tsne_map(bc, time_map, col_bar = False)

    ax3 = plt.subplot(223)
    plot_behavior_labels(bc)
    
    ax4 = plt.subplot(224)
    plot_partial_labels(bc, part_lab)
    
    fig.tight_layout()
    plt.savefig("./figures/summary_plots/" + model_name + "_summary.png", dpi = 1200,
                bbox_inches = "tight")


def compute_persistence(beh_labels):
    """
    Computes the persistence pairs 
    in dimension 1 and 2, from the 
    found behavior labels.
    """

    trans_mat = get_trans_prob_mat(beh_labels)
    np.fill_diagonal(trans_mat, 0)
    trans_mat = trans_mat / np.sum(trans_mat, axis = 1,
                                       keepdims = True)

    trans_mat = np.ones(trans_mat.shape) - trans_mat
    np.fill_diagonal(trans_mat, 0)


    pers = FlagserPersistence()
    pers_trans = pers.fit_transform(trans_mat.reshape(1, *trans_mat.shape))

    return pers, pers_trans


def save_persistence_diagram(pers, pers_trans, path, title):
    """
    Compute the persistence diagram using plotly,
    and save to path as png.
    """

    fig = pers.plot(pers_trans)
    fig.update_layout(
            title_text=title,
            font=dict(
                family="serif",
                size=27,
                color="black"
            )
        )
    fig.write_image(path)


def plot_persistence_diagrams(persistence_pairs, title, labels):
    """
    Use the persim package to plot persistance diagrams.
    Can compare several plots.
    """
    
    ax = plt.axes()
    persim.plot_diagrams(persistence_pairs, labels = labels, ax = ax)
    plots = ax.get_children()
    sc = plots[1]
    new_marker = MarkerStyle("X")
    sc.set_paths((new_marker.get_path(),))
    sc.set_sizes([40])
    plt.legend(loc = "lower right")
    plt.legend(prop={'family': 'serif'})
    plt.title(title)


def plot_relative_bottleneck(bw, bottleneck, ref_bw):
    """
    Plot the bottleneck distance to other models 
    trained with different bandwidth parameter.
    """

    x = bw - ref_bw
    y = bottleneck

    plt.plot(x, y, ls = "--", marker = "o",
             lw = 1, ms = 2)
    plt.xlabel("Bandwidth difference")
    plt.ylabel("Bottleneck distance")


def main():
    pickle_simulated_data()


if __name__ == "__main__":
    main()
