import os

import numpy as np

from alphai_watson.datasource.brainwaves import BrainwavesDataSource
from alphai_watson.detective import DetectionResult, SvmDetective
from alphai_watson.transformer import NullTransformer
from tests.watson import RESOURCES_PATH


def test_detection_result_get_time_location_for_sample():
    original_sample_rate = 1024
    n_timesteps_in_sample = 128
    one_second_of_data_length = original_sample_rate // n_timesteps_in_sample
    result_data = np.zeros(one_second_of_data_length)

    result = DetectionResult(result_data, n_timesteps_in_sample, original_sample_rate)

    assert 125 == result.get_chunk_duration()
    assert 12500 == result.get_millisecond_for_chunk(100)

    original_sample_rate = 44100
    n_timesteps_in_sample = 128
    one_second_of_data_length = original_sample_rate // n_timesteps_in_sample
    result_data = np.zeros(one_second_of_data_length)

    result = DetectionResult(result_data, n_timesteps_in_sample, original_sample_rate)

    np.testing.assert_almost_equal(result.get_chunk_duration(), 2.90249, decimal=5)
    np.testing.assert_almost_equal(result.get_millisecond_for_chunk(100), 290.249, decimal=3)
    assert 290249 == result.get_timedelta_for_chunk(100).microseconds

    probabilities = result.get_probabilities()
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_svm_result():
    n_sensors = 16
    n_timesteps = 1
    transformer = NullTransformer(n_timesteps, n_sensors)

    train_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_sample_1.hd5')
    train_datasource = BrainwavesDataSource(train_data_file, transformer)

    detect_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_and_abnormal.hd5')
    detect_datasource = BrainwavesDataSource(detect_data_file, transformer)

    svm = SvmDetective(max_number_of_samples=4000)

    svm.train(train_datasource.get_train_data('NORMAL'))

    for sample in detect_datasource.get_test_data('NORMAL'):
        detection_result = svm.detect(sample)

        assert isinstance(detection_result, DetectionResult)
        detection_data = detection_result.data
        assert (detection_data >= -1).all() and (detection_data <= 1).all()
        probabilities = detection_result.get_probabilities()
        assert (probabilities >= 0).all() and (probabilities <= 1).all()
        adjusted_probabilities = detection_result.get_probabilities(anomaly_prior=0.01)
        assert (adjusted_probabilities >= 0).all() and (adjusted_probabilities <= 1).all()