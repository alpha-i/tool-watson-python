import datetime
from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractDataSource(metaclass=ABCMeta):

    def __init__(self, source_file, transformer):
        """
        The source of the Data

        :param str source_file: the path of the source file
        :param AbstractDataTransformer transformer: the class to transform data
        """
        self._source_file = source_file
        self._transformer = transformer
        self._raw_data = self._read_samples()

    @property
    @abstractmethod
    def sample_rate(self):
        raise NotImplementedError

    @abstractmethod
    def _read_samples(self):
        """
        Parses the source file for a give sample_type.
        Every sample should have the shape of (number_of_sensors, data_length)
        """

        raise NotImplementedError

    def get_train_data(self, sample_type):
        raw_samples = self._raw_data[sample_type]

        assert len(raw_samples) > 0, "No training samples found."

        return Sample(
            data=self._extract_and_process_samples(raw_samples),
            sample_type=sample_type,
            sample_rate=self.sample_rate,
            number_of_timesteps=self._transformer.number_of_timesteps
        )

    def get_test_data(self, sample_type):
        for sample in self._raw_data[sample_type]:
            yield Sample(
                data=self._extract_and_process_samples([sample]),
                sample_type=sample_type,
                sample_rate=self.sample_rate,
                number_of_timesteps=self._transformer.number_of_timesteps
            )

    def _extract_and_process_samples(self, sample_list):
        """
        Extracts the sample of sample_type according to start and end index.
        Returns stacked samples
        """
        transformed_samples = np.hstack([self._transformer.process_sample(sample) for sample in sample_list])
        transformed_lists = self._transformer.process_stacked_samples(transformed_samples)

        return self._transformer.reshape(np.squeeze(transformed_lists))


class Sample:
    """
    This is a DataClass which decorate the np.array data.
    """
    def __init__(self, data, sample_type, sample_rate, number_of_timesteps):
        """

        :param float sample_rate: sample rate
        :param np.ndarray data: data of shape (number_of_chunks, number_of_sensor, number_of_timesteps)
        :param int number_of_timesteps: number of timesteps in a chunk
        """
        self.data = data
        self.type = sample_type
        self.sample_rate = sample_rate
        self.number_of_timesteps = number_of_timesteps

    @property
    def chunk_shape(self):
        return [self.data[0].shape[0], self.data[0].shape[1], 1]

    def get_chunk(self, index):
        return self.data[index]

    def get_timedelta_for_chunk(self, chunk_index):
        each_timestep = 1 / self.sample_rate
        chunk_duration = each_timestep * self.number_of_timesteps
        return datetime.timedelta(seconds=chunk_index * chunk_duration)

    def get_infinite_random_batch_generator(self, batch_size, strict=False):
        """
        Returns an infinite generator of random batches.

        Usage:
        >>> infinite_generator = source.get_infinite_random_batch_generator(200)
        >>> value = next(infinite_generator)


        :param int batch_size: The size of the random batch to fetch
        :param bool strict: Getting n batches each time means that we might have a remainder of length < n.
                            If strict is false, we'll give the last batch regardless.
        """

        def get_generator_of_random_batches():
            list_of_indexes = list(range(self.data.shape[0]))
            available_indexes = list_of_indexes

            used_indexes = []
            while len(used_indexes) != len(list_of_indexes):
                try:
                    random_indexes = np.random.choice(available_indexes, size=(batch_size,), replace=False)
                except ValueError:
                    if strict:
                        raise StopIteration
                    random_indexes = available_indexes
                finally:
                    used_indexes.extend(random_indexes)
                    available_indexes = list(set(available_indexes) - set(used_indexes))

                yield self.data[random_indexes]

        generator = get_generator_of_random_batches()
        while True:
            try:
                yield next(generator)
            except StopIteration:
                generator = get_generator_of_random_batches()
                yield next(generator)

    def get_batch_generator(self, batch_size, strict=False):
        """
        Return a generator for continuous batches of chunks of size batch_size

        :param int batch_size: the number of chunks in a batch
        :param strict: if True it doens't allow to have a smaller batch at the end of the data.

        :return generator:
        """

        for i in range(0, len(self.data), batch_size):
            chunk = self.data[i:i + batch_size]

            if len(chunk) < batch_size and strict:
                raise StopIteration

            yield chunk
