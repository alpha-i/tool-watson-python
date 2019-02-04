import logging
import numpy as np
import scipy.fftpack as scifft
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from alphai_watson.transformer import AbstractDataTransformer

D_TYPE = np.float32
KEY_MEAN = 'mean'
KEY_SIGMA = 'sigma'
TEMPORAL_AXIS = 2  # Which dimension of the raw data corresponds to time
SENSOR_AXIS = 1
MIN_CLIP_VALUE = -10  # Limit on size of largest numbers in normalised data
MAX_CLIP_VALUE = 10
USE_SCIKIT_NORM = True


class FourierTransformer(AbstractDataTransformer):
    """
    Transformer which transforms the data from the time domain to the frequency domain.
    """

    def __init__(self, number_of_timesteps, number_of_sensors, downsample_factor=4, do_log_power=False,
                 do_local_normalisation=False, perform_pca=False, enable_fft=True):
        super().__init__(number_of_timesteps, number_of_sensors)
        self._sample_length = number_of_timesteps
        self.normalisation = None
        self.downsample_factor = downsample_factor
        self.do_log = do_log_power
        self.do_local_normalisation = do_local_normalisation
        self.perform_pca = perform_pca
        self.enable_fft = enable_fft
        self.quantile_transformer = None
        self.robust_scaler = RobustScaler()

    def apply_global_normalisation(self, data):
        """
        Set the data to have zero mean and unit variance across all chunks

        :param ndarray data: Shape [n_chunks, n_timesteps, n_sensors]
        :return:
        """

        if self.normalisation is None:
            logging.info("Fitting normalisation parameters.")
            self.fit_normalisation(data)
            do_fitting = True
        else:
            do_fitting = False

        if USE_SCIKIT_NORM:
            original_shape = data.shape
            reordered_data = self.reorder_data_for_scikit(data)
            data = self.quantile_transformer.transform(X=reordered_data)
            data = data.reshape(original_shape)
        else:
            if not self.enable_fft:
                data -= self.normalisation[KEY_MEAN]
            data = data / self.normalisation[KEY_SIGMA]

        if self.perform_pca:
            if do_fitting:
                logging.info("Fitting PCA.")
                self.fit_pca(data)
            logging.info("Applying PCA.")
            data = self.apply_pca_transform(data)

        data = np.clip(data, a_min=MIN_CLIP_VALUE, a_max=MAX_CLIP_VALUE)

        return data

    def reorder_data_for_scikit(self, data):
        """ Converts data of type [n_chunks, n_sensors, n_timesteps] into [n_samples, n_features]. """

        n_chunks = data.shape[0]
        n_timesteps = data.shape[2]

        if self.enable_fft:
            reordered_data = data.reshape((n_chunks, -1))
        else:
            reordered_data = data.swapaxes(1, 2)
            reordered_data = reordered_data.reshape((n_chunks * n_timesteps, -1))

        return reordered_data

    def reshape_data_from_scikit(self, data):
        """  Converts data of type [n_samples, n_features] into [n_chunks, n_sensors, n_timesteps].
        Inverse of the process defined by reorder_data_for_scikit.

        :param data: Two dimensional array
        :return: Three dimensional array
        """

        if self.enable_fft:
            n_chunks = data.shape[0]
            reordered_data = data.reshape((n_chunks, self.number_of_sensors, self.number_of_timesteps))
        else:
            denominator = (self.number_of_timesteps // self.downsample_factor)
            n_chunks = data.shape[0] // denominator
            reordered_data = data.reshape((n_chunks, denominator, self.number_of_sensors))
            reordered_data = reordered_data.swapaxes(1, 2)

        return reordered_data

    def apply_local_normalisation(self, data):
        """  Set the data to have zero mean and unit variance within each chunk (and each sensor independently)
        Unlike global norm, here we don't need to store global normalisation constants.
        :param data:
        :return:
        """

        if not self.enable_fft: # Using power spectra enforces data > 0, so we can avoid mean subtraction
            data -= np.mean(data, axis=TEMPORAL_AXIS, keepdims=True)
        data /= np.std(data, axis=TEMPORAL_AXIS, keepdims=True)

        if self.perform_pca:
            if self.normalisation is None:
                logging.info("Fitting PCA.")
                self.fit_pca(data)
                self.normalisation = 1.0

            logging.info("Applying PCA.")
            data = self.apply_pca_transform(data)

        data = np.clip(data, a_min=MIN_CLIP_VALUE, a_max=MAX_CLIP_VALUE)

        return data

    def reshape(self, data):
        return data

    def invert_normalisation(self, normalised_data):
        """ For visualisation purposes we may wish to invert the

        :param normalised_data:
        :return:
        """

        reordered_normalised_data = self.reorder_data_for_scikit(normalised_data)
        data = self.quantile_transformer.inverse_transform(reordered_normalised_data)
        data = self.reshape_data_from_scikit(data)

        return data

    def fit_normalisation(self, data):
        """  Finds the mean and standard deviation of the data, for each individual input element x_ij.
        :return:
        """

        self.normalisation = {}
        self.normalisation[KEY_MEAN] = data.mean(axis=0, dtype=D_TYPE)

        if USE_SCIKIT_NORM:
            self.quantile_transformer = QuantileTransformer(n_quantiles=10, random_state=0)
            reordered_data = self.reorder_data_for_scikit(data)
            self.quantile_transformer.fit(X=reordered_data)
        elif self.enable_fft:  # Ensure range is enforced from 0 to 1
            self.normalisation[KEY_SIGMA] = data.max(axis=0)
        else:
            self.normalisation[KEY_SIGMA] = data.std(axis=0, dtype=D_TYPE)

    def _trim_sample(self, sample):
        """ Removes end of segment to ensure integer multiples of feature_length is available. """

        len_segment = sample.shape[1]

        n_total_chunks = len_segment // self._sample_length
        max_index = n_total_chunks * self._sample_length

        return sample[:, 0:max_index]

    def sample_processor(self, sample, normalise_each_sample=False):
        """ For each flight we shall normalise each sensors overall signal to zero mean and unity rms. """

        if normalise_each_sample:
            n_sensors = sample.shape[0]
            logging.debug("Applying intra-sample normalisation on {} time series".format(n_sensors))
            sample = sample.astype(np.float64, copy=False) # Surprisingly sensitive to precision level
            sample = self.robust_scaler.fit_transform(X=sample.T) # Scikit wants samples in first dimension
            sample = sample.T

        return sample.astype(np.float32)

    def process_stacked_samples(self, full_data):
        """  Prepare list of large data samples for entry into network.
        :param full_data: array of [n_sensors, n_timesteps]
        :return: 4D nparray of shape [n_chunks, feature_length, n_sensors]
        """

        full_data = full_data.astype(np.float32, copy=False)

        n_sensors = full_data.shape[0]
        full_data = np.swapaxes(full_data, 0, 1)  # [n_all_timesteps, n_sensors]
        full_data = np.reshape(full_data, (-1, self._sample_length, n_sensors))  # [n_chunks, n_timesteps, n_sensors]
        full_data = full_data.swapaxes(1, 2)  # [n_chunks, n_sensors, n_timesteps]

        if self.enable_fft:
            full_data = self._batch_power_spectra(full_data)  # Perform multiple fft to prevent memory errors

        if self.do_log:
            full_data = np.log(1.0 + full_data)

        if self.downsample_factor > 1:
            temp = full_data.reshape((full_data.shape[0], full_data.shape[1],
                                      full_data.shape[2] // self.downsample_factor, self.downsample_factor))
            full_data = np.mean(temp, axis=-1, dtype=D_TYPE)

        if self.do_local_normalisation:
            full_data = self.apply_local_normalisation(full_data)
        else:
            full_data = self.apply_global_normalisation(full_data)

        return full_data  # [n_chunks, feature_length, n_sensors]

    @staticmethod
    def _batch_power_spectra(data, n_batches=100):
        """ Aplying fft on full set of data can trigger memory errors"""

        n_samples = data.shape[0]
        batch_size = n_samples // n_batches

        complete = False
        i = 0

        if batch_size == 0:
            input_batch = data
            fft_batch = scifft.fft(input_batch, axis=TEMPORAL_AXIS)
            data = np.sqrt(np.abs(fft_batch) ** 2)
        else:
            while not complete:
                start = i * batch_size
                end = start + batch_size

                input_batch = data[start:end]
                fft_batch = scifft.fft(input_batch, axis=TEMPORAL_AXIS)

                data[start:end] = np.sqrt(np.abs(fft_batch) ** 2)

                if end >= (n_samples - 1):
                    complete = True

                i += 1

        return data
