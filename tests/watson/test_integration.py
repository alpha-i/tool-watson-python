import glob
import os
import tempfile

from alphai_watson.controller import Controller, ExecutionDumper
from alphai_watson.datasource.brainwaves import BrainwavesDataSource
from alphai_watson.detective import SvmDetective
from alphai_watson.performance import SVMPerformanceAnalysis
from alphai_watson.transformer import NullTransformer

from tests.watson import RESOURCES_PATH


def test_controller_flow():
    n_sensors = 16
    n_timesteps = 1
    transformer = NullTransformer(n_timesteps, n_sensors)

    train_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_sample_1.hd5')
    train_datasource = BrainwavesDataSource(train_data_file, transformer)

    detect_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_and_abnormal.hd5')
    detect_datasource = BrainwavesDataSource(detect_data_file, transformer)

    svm = SvmDetective(max_number_of_samples=1000)

    output_path = tempfile.TemporaryDirectory()

    dumper = ExecutionDumper('TEST', output_path.name)

    controller = Controller(
        svm,
        train_datasource,
        detect_datasource,
        SVMPerformanceAnalysis(),
        dumper
    )

    controller.run(train_sample_type='NORMAL')

    saved_file = glob.glob(os.path.join(output_path.name, "detective_execution*"))

    assert len(saved_file) == 1
