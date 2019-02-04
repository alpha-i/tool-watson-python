import logging

import numpy as np
import pandas as pd

from alphai_watson.datasource import AbstractDataSource, Sample

logging.getLogger(__name__).addHandler(logging.NullHandler())


class FlightDataSource(AbstractDataSource):
    """
    Implements datasource for HD5 Flights data
    """
    @property
    def sample_rate(self):
        return 1024

    def _read_samples(self):
        with pd.HDFStore(self._source_file, 'r') as store:
            flights = []
            for key in store.keys():
                logging.debug("Parsing flight {}".format(key))
                flight_dataframe = store.get(key)
                flight_array = flight_dataframe.values
                logging.debug("Start reshape for flight {}".format(key))
                reshaped_flight_data = self._reshape_flight_data(flight_array)
                logging.debug("End reshape for flight {}".format(key))
                flights.append(reshaped_flight_data)

        return {'NORMAL': flights}

    def _reshape_flight_data(self, flight_array):
        reshaped = flight_array.reshape(flight_array.shape[0], self._transformer.number_of_sensors, -1)
        swapped = reshaped.swapaxes(1, 2)
        re_reshaped = swapped.reshape(-1, self._transformer.number_of_sensors).astype(np.float16)
        transposed = re_reshaped.T
        return transposed

    def get_train_data(self, *args, **kwargs):
        raw_samples = self._raw_data['NORMAL']
        return Sample(
            data=self._extract_and_process_samples(raw_samples),
            sample_type='N/A',
            sample_rate=self.sample_rate,
            number_of_timesteps=self._transformer.number_of_timesteps
        )

    def get_test_data(self, *args, **kwargs):
        for sample in self._raw_data['NORMAL']:
            yield Sample(
                data=self._extract_and_process_samples([sample]),
                sample_type='N/A',
                sample_rate=self.sample_rate,
                number_of_timesteps=self._transformer.number_of_timesteps
            )
