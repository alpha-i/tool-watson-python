import os

import numpy as np

from alphai_watson.datasource import Sample
from alphai_watson.datasource.brainwaves import BrainwavesDataSource
from alphai_watson.transformer import NullTransformer

from tests.watson import RESOURCES_PATH


def test_brainwaves_datasource_get_train_data():
    # 1 sample(flight) with shape (239766, 16)
    test_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_and_abnormal.hd5')

    number_of_timesteps = 10
    number_of_sensors = 16

    data_source = BrainwavesDataSource(
        source_file=test_data_file,
        transformer=NullTransformer(number_of_timesteps, number_of_sensors)
    )

    train_sample = data_source.get_train_data('NORMAL')

    assert isinstance(train_sample, Sample)
    assert train_sample.sample_rate == 399.609756097561

    assert isinstance(train_sample.data, np.ndarray)
    assert train_sample.data.shape == (23976, number_of_sensors, number_of_timesteps)

    assert len(train_sample.data) == 23976


def test_brainwaves_datasource_get_test_data():
    # 1 sample(flight) with shape (239766, 16)
    test_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_and_abnormal.hd5')

    number_of_timesteps = 10
    number_of_sensors = 16

    data_source = BrainwavesDataSource(
        source_file=test_data_file,
        transformer=NullTransformer(number_of_timesteps, number_of_sensors)
    )

    test_samples = [sample for sample in data_source.get_test_data('NORMAL')]

    assert len(test_samples) == 1

    test_sample = test_samples[0]

    assert isinstance(test_sample, Sample)
    assert test_sample.sample_rate == 399.609756097561

    assert isinstance(test_sample.data, np.ndarray)
    assert test_sample.data.shape == (23976, number_of_sensors, number_of_timesteps)

    assert len(test_sample.data) == 23976


def test_sample_batch_generator():
    # 1 sample(flight) with shape (239766, 16)
    test_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_and_abnormal.hd5')

    number_of_timesteps = 10
    number_of_sensors = 16
    batch_size = 200

    data_source = BrainwavesDataSource(
        source_file=test_data_file,
        transformer=NullTransformer(number_of_timesteps, number_of_sensors)
    )

    test_samples = [sample for sample in data_source.get_test_data('NORMAL')]
    test_sample = test_samples[0]

    list_of_batches = [batch for batch in test_sample.get_batch_generator(batch_size)]

    assert len(list_of_batches) == (23976 // 200) + 1

    list_of_batches = [batch for batch in test_sample.get_batch_generator(batch_size, True)]

    assert len(list_of_batches) == (23976 // 200)


def test_random_batches_generator():
    # 1 sample(flight) with shape (239766, 16)
    test_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_and_abnormal.hd5')

    number_of_timesteps = 10
    number_of_sensors = 16
    batch_size = 200

    data_source = BrainwavesDataSource(
        source_file=test_data_file,
        transformer=NullTransformer(number_of_timesteps, number_of_sensors)
    )
    train_sample = data_source.get_train_data('NORMAL')

    random_batch_generator = train_sample.get_infinite_random_batch_generator(batch_size)

    batches = [next(random_batch_generator) for x in range(100)]

    assert len(batches) == 100
    batch = batches[0]

    assert isinstance(batch, np.ndarray)
