import datetime
import logging
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

DEFAULT_ANOMALY_PRIOR = 0.1


class AbstractDetective(metaclass=ABCMeta):
    """
    AbstractDetective

    This abstract class define the interface that a detective should implements in order to be pluggable inside
    the Controller and the ADS platform.

    The Interface is very simple and clean, the internal implementation might vary depending on the underlying model.

    It' strongly suggested to implement the ML Model on a different class.

    """
    @abstractmethod
    def train(self, train_sample):
        """
        Performs the training of the model

        :param alphai_watson.datasource.Sample train_sample:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def detect(self, test_sample):
        """
        Performs the detection of the model

        :param alphai_watson.datasource.Sample test_sample:
        :return:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def configuration(self):
        """
        Return a dict with the used configuration, to be saved during testing phase
        :return:
        """


class DetectionResult:
    """
    This class models the result of a Detection. It contains the data as a np.array
    and a series of methods to extrapolate more information on the result as well a method to
    transform the raw result in a list of probabilities

    """
    def __init__(self, data, n_timesteps_in_chunk, original_sample_rate):
        """
        Detection Result class

        :param data: 1-d np array, each value is the score of a chunk which contains n_timesteps_in_sample of values of
                    the original data
        :param n_timesteps_in_chunk: number of timesteps used to give 1 single value of result
        :param original_sample_rate: the original sample rate of the data in input
        """
        self._data = data
        self._number_timesteps_in_chunk = n_timesteps_in_chunk
        self._original_sample_rate = original_sample_rate

    def get_chunk_duration(self):
        milliseconds_in_second = 1000

        return (milliseconds_in_second * self._number_timesteps_in_chunk) / self.original_sample_rate

    def get_millisecond_for_chunk(self, chunk_index):
        return chunk_index * self.get_chunk_duration()

    def get_timedelta_for_chunk(self, chunk_index):
        return datetime.timedelta(milliseconds=self.get_millisecond_for_chunk(chunk_index))

    def get_probabilities(self, anomaly_prior=DEFAULT_ANOMALY_PRIOR, k=-0.06, x0=-20.0):
        """ Initial implicit prior on an anomaly is 0.5, due to the nature of GAN training.
        This may not reflect our prior belief in real test situations, where anomalies are relatively rare.
        Thus we can make an adjustment based on a supplied anomaly prior.
        e.g. if it is believed that 1 in 1000 flights are typically anomalous.

        :param anomaly_prior:
        :param float k: multiplicative scaling of log likelihood
        :param float x0: additive scaling of log likelihood
        :return: Array of probabilities that the data is anomalous.
        """

        # (1) p(a|D) = p(D|a)p(a) / p(D)
        # p(n|D) = p(D|n)p(n) / p(D)
        # (2) 1 - p(a|D) = p(D|n) (1 - p(a)) / p(D)
        # Then (2)/(1) yields 1/p(a|D) - 1 = p(D|n)/p(D|a) (1/p(a) - 1)
        # p(a|D) = p(a) / [p(D|n)/p(D|a) (1 - p(a)) + p(a)]
        # Meanwhile the likelihood ratio (D|n)/p(D|a) = 1 / default_probability - 1

        assert anomaly_prior >= 0 and anomaly_prior <= 1, 'Invalid prior.'

        default_posterior = self._estimate_calibrated_probabilities(k, x0)

        likelihood_ratio = 1 / default_posterior - 1

        posterior_probability = anomaly_prior / (anomaly_prior + likelihood_ratio * (1 - anomaly_prior))

        return posterior_probability

    def _estimate_calibrated_probabilities(self, k=-0.06, x0 = -20.0):
        """
        Determine probability of anomaly p(D|a) for the naive case p(a) = 0.5.
        Default calibration constants correspond to standard Rick & Morty GAN

        :param float k: multiplicative scaling of log likelihood
        :param float x0: additive scaling of log likelihood

        :return np.array:
        """
        return 1 / (1 + np.exp(-k * (self._data - x0)))

    @property
    def data(self):
        return self._data

    @property
    def original_sample_rate(self):
        return self._original_sample_rate

    @property
    def sample_rate(self):
        milliseconds_in_second = 1000

        return milliseconds_in_second / self.get_chunk_duration()

    def __repr__(self):
        return "<DetectionResult: {} samples at {}Hz>".format(len(self.data), self.sample_rate)


class DiagnosticResult:
    """
    This class models the result of a chunk diagnosis.
    """
    def __init__(self, chunk_index, chunk_timedelta, synthetic_chunk, original_chunk):
        self.chunk_index = chunk_index
        self.chunk_timedelta = chunk_timedelta
        self.synthetic_chunk = synthetic_chunk
        self.original_chunk = original_chunk

    @property
    def result(self):
        return self.original_chunk - self.synthetic_chunk

    def get_data_for_sensor_at_index(self, index):

        return {
            'original': self.original_chunk[index],
            'synthetic': self.synthetic_chunk[index]
        }


class SvmDetective(AbstractDetective):
    """
    Detective concrete class which wraps a SVM model
    """
    MAX_N_SAMPLES = 32000  # 1e3 samples and 1e3 features takes approx 1323 -  to train

    def __init__(self, max_number_of_samples=None, nu=0.1):
        self.max_number_of_samples = max_number_of_samples if max_number_of_samples else self.MAX_N_SAMPLES
        self.nu = nu
        self.classifier = svm.OneClassSVM(nu=self.nu, kernel="rbf", gamma=0.1)

    def train(self, train_sample):
        train_data = train_sample.data

        n_train_samples = train_data.shape[0]
        train_data = train_data.reshape(n_train_samples, -1)

        if n_train_samples > self.max_number_of_samples:
            logging.warning(
                'Discarding training data: using {} of {} chunks.'.format(self.max_number_of_samples, n_train_samples))
            train_data = self._subsample_data(train_data)

        self.classifier.fit(train_data)

    def detect(self, test_sample):
        data = test_sample.data.reshape(test_sample.data.shape[0], -1)
        anomaly_score = self.classifier.decision_function(data)

        return DetectionResult(
            np.squeeze(anomaly_score),
            test_sample.number_of_timesteps,
            test_sample.sample_rate
        )

    def _subsample_data(self, data):
        return data[np.random.choice(data.shape[0], self.max_number_of_samples, replace=False)]

    @property
    def configuration(self):
        return {
            'max_number_of_samples': self.max_number_of_samples,
            'nu': self.nu
        }

class IsolForestDetective(AbstractDetective):
    MAX_N_SAMPLES = 32000  # 1e3 samples and 1e3 features takes approx 1323 -  to train

    def __init__(self, max_number_of_samples=None, outliers_fraction=0.4, n_estimators=100):
        self.max_number_of_samples = max_number_of_samples if max_number_of_samples else self.MAX_N_SAMPLES
        self.outliers_fraction = outliers_fraction
        self.n_estimators = n_estimators
        self.classifier = IsolationForest(n_estimators=self.n_estimators,max_samples=self.max_number_of_samples, contamination=self.outliers_fraction,
                                          random_state=None)

    def train(self, train_sample):
        train_data = train_sample.data

        n_train_samples = train_data.shape[0]
        train_data = train_data.reshape(n_train_samples, -1)

        if n_train_samples > self.max_number_of_samples:
            logging.warning(
                'Discarding training data: using {} of {} chunks.'.format(self.max_number_of_samples, n_train_samples))
            train_data = self._subsample_data(train_data)

        self.classifier.fit(train_data)

    def detect(self, test_sample):
        data = test_sample.data.reshape(test_sample.data.shape[0], -1)
        anomaly_score = self.classifier.decision_function(data)

        return DetectionResult(
            anomaly_score,
            test_sample.number_of_timesteps,
            test_sample.sample_rate,
        )

    def _subsample_data(self, data):
        return data[np.random.choice(data.shape[0], self.max_number_of_samples, replace=False)]

    @property
    def configuration(self):
        return {
            'max_number_of_samples': self.max_number_of_samples,
            'outliers_fraction': self.outliers_fraction,
            'n_estimators': self.n_estimators
        }

class LocalOutlierFactorDetective(AbstractDetective):
    MAX_N_SAMPLES = 32000  # 1e3 samples and 1e3 features takes approx 1323 -  to train

    def __init__(self, max_number_of_samples=None, outliers_fraction=0.4, n_neighbors=100):
        self.max_number_of_samples = max_number_of_samples if max_number_of_samples else self.MAX_N_SAMPLES
        self.outliers_fraction = outliers_fraction
        self.n_neighbors = n_neighbors
        self.classifier = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.outliers_fraction)

    def train(self, train_sample):
        train_data = train_sample.data

        n_train_samples = train_data.shape[0]
        train_data = train_data.reshape(n_train_samples, -1)

        if n_train_samples > self.max_number_of_samples:
            logging.warning(
                'Discarding training data: using {} of {} chunks.'.format(self.max_number_of_samples, n_train_samples))
            train_data = self._subsample_data(train_data)

        self.classifier.fit(train_data)

    def detect(self, test_sample):
        data = test_sample.data.reshape(test_sample.data.shape[0], -1)
        anomaly_score = self.classifier._decision_function(data)

        return DetectionResult(
            anomaly_score,
            test_sample.number_of_timesteps,
            test_sample.sample_rate,
        )

    def _subsample_data(self, data):
        return data[np.random.choice(data.shape[0], self.max_number_of_samples, replace=False)]

    @property
    def configuration(self):
        return {
            'outliers_fraction': self.outliers_fraction,
            'n_estimators': self.n_estimators,
            'n_neighbors': self.n_neighbors
        }
