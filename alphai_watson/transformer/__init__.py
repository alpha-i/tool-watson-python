from abc import ABCMeta, abstractmethod

from sklearn.decomposition import PCA
import numpy as np


class AbstractDataTransformer(metaclass=ABCMeta):
    """
    This class implements the Interface of the Transformer class.
    The Transformer is an object which is passed to the datasource and its responsibility is to transform the raw data
    fetched from the datasource according to some pre-defined specification.
    """
    def __init__(self, number_of_timesteps, number_of_sensors):
        """

        :param number_of_timesteps:  defines the number of timesteps expected in the data
        :param number_of_sensors: define the number of sensors in the data
        """
        self.number_of_sensors = number_of_sensors
        self.number_of_timesteps = number_of_timesteps
        self.pca = PCA(n_components=None, whiten=True)

    def process_sample(self, sample):
        """
        This method process a single sample (single flight)
        :param sample:
        :return:
        """
        sample = self._trim_sample(sample)
        return self.sample_processor(sample)

    def reshape(self, data):
        """
        This method reshapes the data according to the number_of_sensors, number_of_timesteps
        :param data:
        :return:
        """
        return data.reshape(self.number_of_sensors, -1, self.number_of_timesteps).swapaxes(0, 1)

    def _trim_sample(self, sample):
        """
        This method makes the length of the data divisible by the number of timesteps
        :param sample:
        :return:
        """
        sample_size = sample.shape[1]
        new_size = int(sample_size - (sample_size % self.number_of_timesteps))
        return sample[:, 0:new_size]

    def fit_pca(self, training_data):  # Expects data of shape (n_samples, n_features)
        """
        This function fit the PCA
        :param training_data:
        :return:
        """
        n_samples = training_data.shape[0]
        return self.pca.fit(training_data.reshape((n_samples, -1)))

    def apply_pca_transform(self, data):
        """
        This function apply PCA
        :param data:
        :return:
        """
        original_data_shape = data.shape
        n_samples = original_data_shape[0]
        self.pca.transform(data.reshape((n_samples, -1)))

        return data.reshape(original_data_shape)

    def invert_pca_transform(self, data):
        """
        Inverts the PCA
        :param data:
        :return:
        """
        return self.pca.inverse_transform(data)

    @abstractmethod
    def sample_processor(self, sample):
        """
        This method is the interface called by the DataSource. The implementation must process a single sample
        :param sample:
        """
        raise NotImplementedError

    @abstractmethod
    def process_stacked_samples(self, stacked_samples):
        """
        This method is the interface called by the datasource and process all the samples together.

        Prepare list of large data samples for entry into network.
        :param stacked_samples:
        """
        raise NotImplementedError


class NullTransformer(AbstractDataTransformer):
    """
    Concrete transformer which doesn't apply any transformation
    """
    def sample_processor(self, sample):
        return sample

    def process_stacked_samples(self, stacked_samples):
        return stacked_samples


class NormalizeTransformer(AbstractDataTransformer):
    """
    Concrete Transformer which normalize data
    """
    def sample_processor(self, sample):
        """ Removes end of segment to ensure integer multiples of feature_length is available. """

        data = sample - np.mean(sample.flatten(), dtype=np.float32)
        return data / data.flatten().std(dtype=np.float32)

    def process_stacked_samples(self, stacked_samples):
        """ Prepare list of large data samples for entry into network. """

        return stacked_samples
