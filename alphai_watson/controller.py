import logging
import os
import pickle
from time import strftime

from alphai_watson.detective import DiagnosticResult
from alphai_watson.performance import AnomalyTypes, ResultCollector


logging.getLogger(__name__).addHandler(logging.NullHandler())


class Controller:

    def __init__(self, detector, train_datasource, detect_datasource, performance_analysis, execution_dumper=None):
        """
        Controller for research workflow

        :param AbstractDetective detector: the detective.
        :param AbstractDataSource train_datasource: the source of the training data.
        :param AbstractDataSource detect_datasource: the source of the detect data.
        :param PerformanceAnalysis performance_analysis: the performance analyser.
        :param ExecutionDumper execution_dumper: the helper class implementing the writing of the execution results.

        """
        self._detector = detector
        self._train_datasource = train_datasource
        self._detect_datasource = detect_datasource
        self._performance_analysis = performance_analysis
        self.execution_dumper = execution_dumper
        self.is_trained = False

    def _train(self, train_sample_type):
        """
        performs the training
        :param str train_sample_type: define which type of sample has to be used to train the model
        """
        train_data = self._train_datasource.get_train_data(train_sample_type)
        self._detector.train(train_data)
        self.is_trained = True

    def diagnose_single_chunk(self, chunk_index):
        """
        Performs a diagnostic of a single chunk

        :param int chunk_index: the index which identify the location of the chunk

        :return DiagnosticResult
        """
        if not self.is_trained:
            self._train('NORMAL')

        test_samples = self._get_samples_for_testing()
        test_sample = test_samples[0]
        original_chunk = test_sample.get_chunk(chunk_index)
        chunk_timedelta = test_sample.get_timedelta_for_chunk(chunk_index)

        synthetic_chunk = self._detector.diagnose(original_chunk)

        return DiagnosticResult(
            chunk_index,
            chunk_timedelta,
            synthetic_chunk,
            original_chunk
        )

    def run(self, train_sample_type):
        """
        This is the main routine of the Controller
        In this method, the controller performs the following:

        1 - Trains the model with the data form the Training dataset
        2 - Performs a detection for each data contained in the test dataset and collect the result
        3 - performs a performance analysis on the result of the detection to determine the roc score of the model.

        :param str train_sample_type: define if training should be done using NORMAL or ABNORMAL data
        """
        self._train(train_sample_type)

        result_collector = ResultCollector()

        for i, sample in enumerate(self._get_samples_for_testing()):
            sample_type = sample.type
            true_value = AnomalyTypes[sample_type].value

            detection_result = self._detector.detect(sample)
            logging.info("Sample results: {}".format(detection_result._data))
            logging.info("Sample probabilities: {}".format(detection_result.get_probabilities()))

            result_collector.add_result(i, detection_result, true_value)

        sample_roc_score, chunk_roc_score = self._analyse_result(result_collector)

        if self.execution_dumper:
            self.execution_dumper.add_detective_configuration(self._detector.configuration)
            self.execution_dumper.add_result_collection(result_collector.sample_list)
            self.execution_dumper.dump()

        return sample_roc_score, chunk_roc_score

    def _analyse_result(self, result_collector):
        """
        Performs the analysis using the PerformanceAnalysis class
        :param ResultCollector result_collector:
        :return:
        """
        sample_roc_score = self._performance_analysis.analyse(result_collector.sample_score,
                                                              result_collector.sample_true_value)
        chunk_roc_score = self._performance_analysis.analyse(result_collector.chunk_score,
                                                             result_collector.chunk_true_value)
        logging.info("Tested {}".format(result_collector))
        logging.info("Chunk ROC SCORE: {}".format(chunk_roc_score))
        logging.info("Sample ROC SCORE: {}".format(sample_roc_score))

        return sample_roc_score, chunk_roc_score

    def _get_samples_for_testing(self):
        """
        Prepare the list of the sample for testing.
        :return:
        """
        normal_samples = [sample for sample in self._detect_datasource.get_test_data('NORMAL')]
        abnormal_samples = [sample for sample in self._detect_datasource.get_test_data('ABNORMAL')]

        return normal_samples + abnormal_samples


class ExecutionDumper:
    """
    Class that implements the writer of train-detect routine
    """
    def __init__(self, execution_name, output_path):
        """
        :param str execution_name: name of this run
        :param str output_path: path where to save the execution dump
        """
        self.execution_name = execution_name
        self.output_path = output_path
        self._detection_results = []
        self._detective_configuration = None

    def add_result_collection(self, sample_result_list):
        """
        add result to the collection
        :param list sample_result_list: list of SampleResult

        """
        for sample_result in sample_result_list:
            self._detection_results.append({
                'result': sample_result.detection_result,
                'true_value': sample_result.true_value
            })

    def add_detective_configuration(self, configuration):
        """
        Add detective configuration for later save
        :param dict configuration:

        """
        self._detective_configuration = configuration

    def dump(self):
        """
        Perform the dump

        """
        save_filename = "detective_execution_{}_{}.pickl".format(
            self.execution_name,
            strftime("%Y%m%d%H%M%S")
        )
        with open(os.path.join(self.output_path, save_filename), "wb") as f:
            pickle.dump({
                'detection_results': self._detection_results,
                'detective_configuration': self._detective_configuration
            }, f)
