import numpy as np
import pickle as pkl

from helper_functions import (
        pickle_relevant_features, spline_regression,
        plot_scaleogram, estimate_pdf, 
        get_watershed_labels, assign_labels,
)

import pycwt as wavelet
from pycwt.helpers import find

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d


class BehavioralClustering():
    """
    Wrapper class for clustering distinct behaviours
    in rats from recorded postural dynamics.

    Attributes:
        face (bool): True if face/whisker data is included
            in the analysis
        train_file_names_post (List(str)): Name of raw data 
            to be used in the clustering
        train_file_names_face (List(str)): Name of raw data 
            to be used in the clustering
        original_file_path (str): Location of the training
            files.
        extracted_file_path (str): Location of the pickled 
            files containing only relevant features.
        raw_features_post (List(np.ndarray)): The features relevant to 
            the analysis, straight from the postural input files.
        raw_features_face (List(np.ndarray)): The features relevant to 
            the analysis, straight from the face/whisker input files.
        knot_frequency (float): Interior spacing of the knots
            used in the spline regression
        spline_dim (int): Dimension of the regression spline
            used in detrending the data
        trend_post (list(np.ndarray)): The trend found in the 
            time series by spline regression
        data_detrended_post (list(np.ndarray)): The detrended
            time series.
        data_detrended_face (list(np.ndarray)): The detrended
            time series.
        feature_names_post (list(str)): Names of the features
            used
        feature_names_face (list(str)): Names of the features
            used
        n_features_post (int): Number of different recorded
            postural features, i.e., number of movements recorded
            per animal.
        n_features_face (int): Number of different recorded
            whisker features
        capture_framerate_post(int): Frequency (Hz) of the 
            recorded postural time series.
        capture_framerate_face(int): Frequency (Hz) of the 
            recorded whisker time series.
        dt_post (float): Time interval in the postural 
            time series
        dt_face (float): Time interval in the whisker 
            time series
        used_indices_post (list(np.ndarray)): Indices of non-NaN values
        used_indices_face (list(np.ndarray)): Indices of non-NaN values
        num_freq (int): Number of dyadically spaced frequency
            scales used in the CWT
        min_freq (float): Minumum frequency scale used in 
            the CWT
        max_freq (float): Maximum frequency scale used in 
            the CWT
        dj (float): The number of sub-octaves per octave,
            logarithmic spacing for the scales used
            in the CWT
        mother (str): Mother wavelet used in the CWT
        power (list(np.ndarray)): The computed power
            spectrum from the cwt
        features (list(np.ndarray): Features extracted 
            from the time frequency analysis
        scales (np.ndarray): Stores the cwt scales
        freqs (np.ndarray): Stores the cwt frequencies
        var_pca (np.ndarray): Percentage ratios of the 
            explained variance in the principal component
            analysis
        n_pca (int): Number of principal components
            explaining more than 95% of the data
        fit_pca (np.ndarray): The transformed (reduced) 
            features using principal component analysis
        ds_rate (int): Downsampling frequency in [Hz],
            used as t-SNE memory complexity is O(n^2)
        perp (float): Perplexity parameter used in t-SNE
        embedded_train (np.ndarray): Two dimensional embedding
            obtained by t-SNE on the downsampled data
        embedded (np.ndarray): The low dimensional embedding
            of all points
        tsne_ind (np.ndarray): Indices of time points used
            when finding the t-SNE embedding
        kde (np.ndarray): Kernel density estimation of
            the t-SNE embedding
        border (int): Border around the t-SNE embedding
            for improved watershed segmentation, and 
            visualization
        grid (list(np.ndarray)): Grid on which the 
            kernel density estimation is applied
        bw (float/str): Bandwidth method used in the kernel
            density estimation
        ws_labels (np.ndarray): Assigned clusters
            found by watershed segmentation
        beh_labels (np.ndarry): Final classification of
            the non-nan time points
        n_behaviors (int): Number of unique behaviors found
    """

    def __init__(self):
        self.face = False
        self.train_file_names_post = []
        self.train_file_names_face = []
        self.original_file_path = None
        self.extracted_file_path = None
        self.raw_features_post = [] 
        self.raw_features_face = [] 
        self.knot_frequency = 0.5
        self.spline_dim = 3
        self.trend_post = []
        self.trend_face = []
        self.data_detrended_post = []
        self.data_detrended_face = []
        self.feature_names_post = ["exp0", "exp1", "exp2",
                              "speed2", "BackPitch",
                              "BackAzimuth", "NeckElevation"]
        self.feature_names_face = None
        self.n_features_post = len(self.feature_names_post)
        self.n_features_face = None
        self.capture_framerate_post= 120 
        self.capture_framerate_face= 210 
        self.dt_post = 1/self.capture_framerate_post
        self.dt_face = 1/self.capture_framerate_face
        self.used_indices_post = []
        self.used_indices_face = []
        self.data_post = []
        self.data_face = []
        self.num_freq = 18
        self.min_freq_post = 0.5
        self.max_freq_post = 20
        self.min_freq_face = 0.5
        self.max_freq_face = 20
        self.dj_post = 1/(self.num_freq - 1)*np.log2(
                    self.max_freq_post/self.min_freq_post)
        self.dj_face = 1/(self.num_freq - 1)*np.log2(
                    self.max_freq_face/self.min_freq_face)
        self.mother = "morlet"
        self.power_post = []
        self.power_face = []
        self.features_post = []
        self.features_face = []
        self.aligned_features = []
        self.scales_post = np.zeros(self.num_freq)
        self.scales_face = np.zeros(self.num_freq)
        self.freqs_post = np.zeros(self.num_freq)
        self.freqs_face = np.zeros(self.num_freq)
        self.var_pca = None
        self.n_pca = None
        self.fit_pca = None
        self.ds_rate = 2
        self.perp = 30
        self.embedded_train = None
        self.embedded = None
        self.kde = None
        self.border = None
        self.grid = None
        self.bw = "scott" 
        self.ws_labels = None
        self.beh_labels = None


    def remove_nan(self):
        """
        Removes each time point where one or more value
        in the time series contain a NaN value.
        Stores the new time series data in 
        an attribute, together with the used row indices.
        """
                
        # Iterate over animals
        for i in range(len(self.raw_features_post)):
            # Find the indices of usable rows
            self.used_indices_post.append(~np.isnan(
                self.raw_features_post[i]).any(axis =1))
            # Filter the data and add to an attribute
            self.data_post.append(self.raw_features_post[i][self.used_indices_post[i]])

        if self.face:
            # Iterate over animals
            for i in range(len(self.raw_features_face)):
                # Find the indices of usable rows
                self.used_indices_face.append(~np.isnan(
                    self.raw_features_face[i]).any(axis =1))
                # Filter the data and add to an attribute
                self.data_face.append(self.raw_features_face[i][self.used_indices_face[i]])
    



    def detrend(self):
        """
        Detrends the time series individually
        using spline regression, and stores the
        trend and detrended data as attributes
        """
       
        # Iterate over animals
        for d in range(len(self.data_post)):
            # Perform spline regression on each time series
            trend = [spline_regression(y, self.spline_dim,
                        self.capture_framerate_post
                            ,
                        self.knot_frequency) for y in self.data_post[d].T]
            self.trend_post.append(np.array(trend).T)
            self.data_detrended_post.append(self.data_post[d] - self.trend_post[d])
            # Normalize data
            std = np.std(self.data_detrended_post[-1], axis = 0)
            self.data_detrended_post[-1] = self.data_detrended_post[-1] / std

        if self.face:
            # Iterate over animals
            for d in range(len(self.data_face)):
                # Perform spline regression on each time series
                trend = [spline_regression(y, self.spline_dim,
                            self.capture_framerate_face
                                ,
                            self.knot_frequency) for y in self.data_face[d].T]
                self.trend_face.append(np.array(trend).T)
                self.data_detrended_face.append(self.data_face[d] - self.trend_face[d])
                # Normalize data
                std = np.std(self.data_detrended_face[-1], axis = 0)
                self.data_detrended_face[-1] = self.data_detrended_face[-1] / std


    def time_frequency_analysis(self):
        """
        Perform time frequency analysis
        using the continuous wavelet transform on the 
        detrended time series data. The power spectrum
        is computed using the pycwt python package,
        and divided by the scales. Taking the square root,
        centering and rescaling by the trend standard 
        deviation, the extended time series are concatenated
        with the normalized trend data creating new
        features. We store the power matrix, scales and 
        frequencies to be used for plotting scaleograms.
        """ 

        # Iterate over animals
        for d in range(len(self.data_post)):
            # Create storage for new feature vector
            x_d = np.zeros(shape = (len(self.data_post[d]),
                                    self.n_features_post*
                                    (self.num_freq + 1)))

            # Iterate over raw features
            self.power_post.append([])
            for i in range(self.n_features_post):
                # Apply cwt
                wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
                        self.data_post[d][:,i], self.dt_post, self.dj_post, 1/self.max_freq_post,
                        self.num_freq - 1, self.mother)
                # Store the frequencies and scales
                if scales.all() != self.scales_post.all():
                    self.scales_post = scales
                    self.freqs_post = freqs
                # Compute wavelet power spectrum
                power_i = np.abs(wave)**2
                self.power_post[d].append(power_i)
                # Normalize over scales (Liu et. al. 07)
                power_i /= scales[:, None]
                # Take the square root to ...
                power_i = np.sqrt(power_i)
                
                # Center and rescale trend
                trend = self.trend_post[d][:,i]
                trend_std = np.std(trend)
                trend = (trend - np.mean(trend)) / trend_std

                # Center and rescale power spectrum 
                power_i = (power_i - np.mean(power_i)) / trend_std    
                # Store new features
                x_d[:,i] = trend
                x_d[:,self.n_features_post + i*self.num_freq:self.n_features_post +(i + 1)*
                    self.num_freq] = power_i.T
        
            self.features_post.append(x_d)

        if self.face:
            # Iterate over animals
            for d in range(len(self.data_face)):
                # Create storage for new feature vector
                x_d = np.zeros(shape = (len(self.data_face[d]),
                                        self.n_features_face*
                                        (self.num_freq + 1)))

                # Iterate over raw features
                for i in range(self.n_features_face):
                    # Apply cwt
                    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
                            self.data_face[d][:,i], self.dt_face, self.dj_face, 1/self.max_freq_face,
                            self.num_freq - 1, self.mother)
                    # Store the frequencies and scales
                    if scales.all() != self.scales_face.all():
                        self.scales_face = scales
                        self.freqs_face = freqs
                    # Compute wavelet power spectrum
                    power_i = np.abs(wave)**2
                    # Normalize over scales (Liu et. al. 07)
                    power_i /= scales[:, None]
                    # Take the square root to ...
                    power_i = np.sqrt(power_i)
                    
                    # Center and rescale trend
                    trend = self.trend_face[d][:,i]
                    trend_std = np.std(trend)
                    trend = (trend - np.mean(trend)) / trend_std

                    # Center and rescale power spectrum 
                    power_i = (power_i - np.mean(power_i)) / trend_std    
                    # Store new features
                    x_d[:,i] = trend
                    x_d[:,self.n_features_face + i*self.num_freq:self.n_features_face +(i + 1)*
                        self.num_freq] = power_i.T
            
                self.features_face.append(x_d)

    
    def standardize_features(self):
        """
        Standardizes the features extracted via CWT such that they can
        be used in PCA, i.e., with mean zero and variance one.
        """
        
        # Iterate over animals
        for i in range(len(self.data_post)):
            self.features_post[i] = ((self.features_post[i] - np.mean(self.features_post[i],
                                                            axis = 0)) /
                                np.std(self.features_post[i], axis = 0))

        if self.face:
            for i in range(len(self.data_face)):
                self.features_face[i] = ((self.features_face[i] - np.mean(self.features_face[i],
                                                                axis = 0)) /
                                    np.std(self.features_face[i], axis = 0))


    def align_time_points(self):
        """
        Downsample the high frequency recording of the 
        facial data to match the postural data.
        Using linear interpolation we can approximate 
        the facial features at the time points needed.
        """

        # Iterate over animals 
        for i in range(len(self.data_post)):
            # Linear interpolation
            t = np.arange(0, len(self.data_face[i]))*self.dt_face
            f = interp1d(t, self.features_face[i],
                         axis = 0, kind = "linear",
                         bounds_error = False, fill_value = 0)
            t_post = np.arange(0, len(self.raw_features_post[i])*self.dt_post,
                               self.dt_post)
            t_post = np.arange(0, len(self.raw_features_post[i]))*self.dt_post
            # Find the used time points
            t_used = t_post[self.used_indices_post[i]]
            self.aligned_features.append(f(t_used))
            # Standardize
            self.aligned_features[i] = ((self.aligned_features[i] - np.mean(self.aligned_features[i],
                                    axis = 0)) / np.std(self.aligned_features[i], axis = 0))

        
    def pca(self, n_train = 0):
        """
        Compute the principal components explaining 95% of
        the variance in features extracted via CWT.
        Stores the new reduced features as an attribute.

        Arguments:
            n_train (int): Approximate number of training 
                points to use in the PCA (because of memory restrictions).
                If 0 all data is used
        """

        # Concatenate features
        features = np.concatenate(self.features_post, axis = 0)

        if self.face:
            aligned_features = np.concatenate(self.aligned_features,
                                      axis = 0)
            features = np.concatenate([features, aligned_features],
                                  axis = 1)
        
        # Find principal components
        pca = PCA()
        if n_train:
            # Train on downsampled data
            pca.fit(features[::len(features) // n_train,:])
        else:
            # Train on all features
            pca.fit(features)

        # Find number of features explaning 95% of the variance
        self.var_pca = pca.explained_variance_ratio_
        self.n_pca = np.argmax(np.cumsum(self.var_pca) > 0.95) + 1

        # Apply the transformation using the sufficient
        # number of principal components
        pca = PCA(n_components = self.n_pca)
        if n_train:
            # Train on downsampled data
            pca.fit(features[::len(features) // n_train,:])
            # Transfrom all data
            self.fit_pca = pca.transform(features)
        else:
            self.fit_pca = pca.fit_transform(features)
          

    def tsne(self):
        """
        Embed the principal component scores onto a 
        two dimensional manifold using t-SNE.
        Previous to the dimensionality reduction 
        we downsample to reduce memory complexity in
        the algorithm.
        """

        # Compute the downsampling rate to 
        # give approx 30000 training points
        self.ds_rate = 30000 * self.capture_framerate_post / len(self.fit_pca)

        # Downsample the principal component scores
        self.tsne_ind = np.arange(0, len(self.fit_pca),
                        int(self.capture_framerate_post
                / self.ds_rate))
        train = self.fit_pca[self.tsne_ind,:] 

        # Perform t-SNE
        self.embedded_train = TSNE(n_components = 2,
                                perplexity = self.perp,
                                init = "pca").fit_transform(train)
         

    def pre_embedding(self):
        """
        Embeds all data points into the t-SNE plane
        by choosing the components of the nearest 
        neighbor (in the training set) by euclidean
        distance in the PCA space.
        """ 

        # Principal component scores for 
        # points used to find t-SNE embedding.
        pca_train = self.fit_pca[self.tsne_ind,:]

        # Create storage for embeddings
        self.embedded = np.zeros(
                shape = (len(self.fit_pca), 2))

        # Iterate over all time points
        for i in range(len(self.fit_pca)):
            # Find closest time point in PCA space
            dist = cdist(self.fit_pca[i,:][np.newaxis,:],
                         pca_train)

            # Choose embedding corresponding to this
            # time point
            self.embedded[i] = self.embedded_train[dist.argmin()]

    
    def kernel_density_estimation(self, pixels):
        """
        Perform kernel density estimation on the t-SNE 
        embedding, estimating the pdf using Gaussian kernels.

        Arguments:
            pixels (int): Pixelation aloach each axis which
                the pdf is computed on
        """

        self.border = np.max(np.abs(self.embedded))/5

        # Outer border set for better visualizations
        self.kde, self.grid = estimate_pdf(
                self.embedded, self.bw, self.border, pixels)


    def watershed_segmentation(self):
        """
        Perform watershed segmentation on the 
        kernel density estimation pdf.
        """

        self.ws_labels = get_watershed_labels(self.kde)   


    def classify(self):
        """
        Assigns a behavioral action label to 
        all (non-nan) time points in the data set.
        """

        # Classify the embedded points
        self.beh_labels = assign_labels(self.embedded, 
                                   self.ws_labels,
                                   self.grid)

        # Store number of unique behaviors
        self.n_behaviors = len(np.unique(self.beh_labels))
         

    def set_original_file_path(self, original_file_path):
        self.original_file_path = original_file_path


    def set_extracted_file_path(self, extracted_file_path):
        self.extracted_file_path = extracted_file_path


    def pickle_relevant_features(self, file_names):
        """
        Extract the relevant raw features from the original
        data, pickle it, and dump it in the location
        specified in the extraced_file_path attribute

        Parameters:
            file_names (List(str)): File names of original data
        """
        for name in file_names:
            pickle_relevant_features(self.original_file_path + name,
                                     self.extracted_file_path +
                                     name.split(".")[0] + 
                                     "_extracted.pkl")


    def load_relevant_features(self, train_files_post,
                               train_files_face = None):
        """
        Loads the pickled raw relevant features into the 
        instance attribute to be used for training.
        Stores number of features as attribute.

        Parameters:
            train_files_post (List(str)): Names of training files
                for the postural parameters
            train_files_face (List(str)): Names of training files
                for the face/whisker parameters
        """

        self.train_file_names_post = train_files_post
        for filename in train_files_post:
            with open(self.extracted_file_path + 
                      filename.split(".")[0] + "_extracted.pkl",
                      "rb") as file:
                data = pkl.load(file)
                if isinstance(data, dict):
                    self.raw_features_post.append(data["relevant_features"])
                    self.feature_names_post = data["feature_names"]
                else:
                    self.raw_features_post.append(data)


        # Store the number of features
        self.n_features_post = self.raw_features_post[0].shape[1]

        if train_files_face != None:
            self.face = True
            self.train_file_names_face = train_files_face
            for filename in train_files_face:
                with open(self.extracted_file_path + 
                          filename.split(".")[0] + "_extracted.pkl",
                          "rb") as file:
                    data = pkl.load(file)
                    self.raw_features_face.append(data["relevant_features"])
                    self.feature_names_face = data["feature_names"]

            # Store the number of features
            self.n_features_face = self.raw_features_face[0].shape[1]

    
