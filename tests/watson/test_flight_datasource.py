import os

from alphai_watson.datasource.flight import FlightDataSource
from alphai_watson.transformer import NullTransformer

test_file_path = os.path.join(os.path.dirname(__file__), '../resources/new_smaller_flight_store.h5')


def test_get_train_data():
    number_of_sensors = 8
    sample_rate = 1024
    number_of_timesteps = 10

    train_data_source = FlightDataSource(
        source_file=test_file_path,
        transformer=NullTransformer(
            number_of_timesteps,
            number_of_sensors
        )
    )

    # flight 1 contains 50 seconds of data
    assert train_data_source._raw_data['NORMAL'][0].shape == (number_of_sensors, sample_rate * 5 * 10)
    # flight 2 contains 100 seconds of data
    assert train_data_source._raw_data['NORMAL'][1].shape == (number_of_sensors, sample_rate * 5 * 20)
    # contains 75 seconds of data
    assert train_data_source._raw_data['NORMAL'][2].shape == (number_of_sensors, sample_rate * 5 * 15)

    train_data = train_data_source.get_train_data()

    expected_train_data_length = (
            train_data_source._raw_data['NORMAL'][0].shape[1] +
            train_data_source._raw_data['NORMAL'][1].shape[1] +
            train_data_source._raw_data['NORMAL'][2].shape[1]
    ) // number_of_timesteps

    assert train_data.data.shape == (expected_train_data_length, 8, 10)
