import numpy as np

from alphai_watson.transformer import NullTransformer


def test_transform_trim():
    transform_class = NullTransformer(10, 16)

    sample = np.ndarray([16, 105])

    trimmed_sample = transform_class.process_sample(sample)

    assert trimmed_sample.shape == (16, 100)
