import abc
import enum

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from alphai_watson.detective import DetectionResult


class AnomalyTypes(enum.Enum):
    NORMAL = 1
    ABNORMAL = 0


class PerformanceAnalysis(metaclass=abc.ABCMeta):
    """
    Define the PerformanceAnalysis class interface.
    """
    @abc.abstractmethod
    def analyse(self, detection_result: np.ndarray, expected_truth: AnomalyTypes):
        raise NotImplementedError


class GANPerformanceAnalysis(PerformanceAnalysis):
    """
    Implements the Performance Analysis when a GAN model is used on a Detector
    It's underlying analysis is based on the Roc Score.
    """
    def __init__(self, configuration: dict):
        self._configuration = configuration

    def analyse(self, detection_result, expected_truth):
        """
        :param DetectionResult detection_result: the result of a dection
        :param np.ndarray expected_truth: the expected result
        """
        roc_score = roc_auc_score(expected_truth, detection_result)
        return roc_score


class SVMPerformanceAnalysis(PerformanceAnalysis):
    """
    Implements the PerformanceAnalysis for Detectives which uses SVM Models
    """
    def analyse(self, detection_result, expected_truth):
        """
        :param DetectionResult detection_result: the result of a dection
        :param np.ndarray expected_truth: the expected result
        """

        roc_score = roc_auc_score(expected_truth, detection_result)
        return roc_score

    def roc_curve(self, detection_result, expected_truth):
        """
        :param DetectionResult detection_result: the result of a dection
        :param np.ndarray expected_truth: the expected result
        """

        fpr, tpr, thresholds = roc_curve(expected_truth, detection_result)
        return fpr, tpr, thresholds

    def average_precision_score(self, detection_result, expected_truth):
        """
        :param DetectionResult detection_result: the result of a dection
        :param np.ndarray expected_truth: the expected result
        """

        avg_precision_score = average_precision_score(expected_truth, detection_result)
        return avg_precision_score


class SampleResult:
    """
    Simple Data Class which contains detection results and true values
    """
    def __init__(self, detection_result, true_value):

        self.detection_result = detection_result
        self.true_value = true_value

        self.chunk_scores = detection_result.data
        self.chunk_true_values = np.zeros(detection_result.data.shape) + true_value

        self.sample_score = np.mean(detection_result.data)
        self.sample_true_value = true_value

    @property
    def number_of_chunks(self):
        return len(self.chunk_scores)


class ResultCollector:
    """
    This class implements a Repository of result. This class facilitates grouping of the results for calculation.
    """
    def __init__(self):
        self._sample_list = []

    def add_result(self, index, detection_result, true_value):
        self._sample_list.insert(
            index,
            SampleResult(detection_result, true_value)
        )

    @property
    def sample_list(self):
        """
        Returns the sample list
        :return np.array:
        """
        return self._sample_list

    @property
    def sample_score(self):
        """
        return the list of scores, one for sample
        :return np.array:
        """
        return np.array([sample.sample_score for sample in self._sample_list])

    @property
    def sample_true_value(self):
        """
        Return the list of true values, one per sample. the index matches the sample_score property
        :return np.array:
        """
        return np.array([sample.sample_true_value for sample in self._sample_list])

    @property
    def chunk_score(self):
        """
        return a 1-d array containing all the score for all the chunks in the sample list, in order.
        :return np.array:
        """
        return np.hstack([sample.chunk_scores for sample in self._sample_list])

    @property
    def chunk_true_value(self):
        """
        return a 1-day array containin the true values for all the chunks in all the samples
        :return np.array:
        """
        return np.hstack([sample.chunk_true_values for sample in self._sample_list])

    @property
    def total_number_of_chunks(self):
        """
        Give the total number of chunk in the sample list
        :return int:
        """
        return sum([sample.number_of_chunks for sample in self._sample_list])

    def __str__(self):
        return "ResultCollector: total samples {}. total chunks {}".format(
            len(self._sample_list),
            self.total_number_of_chunks
        )
