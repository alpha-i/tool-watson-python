import logging

import h5py

from alphai_watson.datasource import AbstractDataSource


class BrainwavesDataSource(AbstractDataSource):
    """
    Implements a Datasource for Kaggle Brainwaves data
    """
    SAMPLE_TYPES = ['NORMAL', 'ABNORMAL']

    def __init__(self, source_file, transformer):
        super().__init__(source_file, transformer)

    @property
    def sample_rate(self):
        return self._sample_rate

    def _read_samples(self):
        """
        Parses the source file for a give sample_type.
        Every sample should have the shape of (number_of_sensors, data_length)
        """

        logging.debug("Start file parsing")
        samples = {}
        with h5py.File(self._source_file, 'r') as store:
            for sample_type in self.SAMPLE_TYPES:
                samples_of_type = store.get(sample_type)
                sample_list = samples_of_type.get('DATA')
                _data = [
                    sample_list.get(sample_id).value
                    for sample_id in list(sample_list.keys())
                ]
                self._sample_rate = samples_of_type.get('SAMPLE_RATE').value  # FIXME
                samples[sample_type] = _data

        logging.debug("end file parsing")

        return samples
