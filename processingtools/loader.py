from enum import Enum
from functools import reduce
from pathlib import Path
import h5py
from .datamodel import InputParamLogger
from .config import LoaderType


class Loader(InputParamLogger):
    def __init__(self, *, loader_name, loader_type):
        self.loader_name = loader_name
        self.loader_type = loader_type
        self.daily_path = None
        self.run_name = None

    def set_run(self, daily_path, run_name):
        self.daily_path = daily_path
        self.run_name = run_name

    def extract_data(self, datamodel):
        raise NotImplementedError

    def arrange_data(self, data):
        raise NotImplementedError

    @staticmethod
    def reduce_datasource_by_key(dataframe, keychain):
        data = reduce(lambda x, y: x[y], keychain.split('/'), dataframe)
        return data

    def load_shot(self, shot_num, datamodel):
        extracted_data = self.extract_data(datamodel)
        arranged_data = self.arrange_data(extracted_data)
        return arranged_data


class RawLoader(Loader):
    def __init__(self, *, loader_name, datastream_name, file_prefix):
        super(RawLoader, self).__init__(loader_name=loader_name, loader_type=LoaderType.RAW)
        self.datastream_name = datastream_name
        self.file_prefix = file_prefix
        self.data_path = None
        self.num_shots = None

    def set_run(self, daily_path, run_name):
        super(RawLoader, self).set_run(daily_path, run_name)
        self.data_path = Path(self.daily_path, 'data', self.run_name, self.datastream_name)
        self.num_shots = self.get_num_shots()

    def get_num_shots(self):
        file_list = list(self.data_path.glob('*.h5'))
        self.num_shots = len(file_list)
        return self.num_shots

    def get_file_path(self, shot_num):
        file_name = f'{self.file_prefix}_{shot_num:05d}.h5'
        file_path = Path(self.data_path, file_name)
        return file_path

    def load_shot(self, shot_num):
        raise NotImplementedError


class AbsorptionLoader(RawLoader):
    def __init__(self, *, loader_name, datastream_name, file_prefix,
                 atom_frame_keychain, bright_frame_keychain, dark_frame_keychain, roi_slice):
        super(AbsorptionLoader, self).__init__(loader_name=loader_name, datastream_name=datastream_name,
                                               file_prefix=file_prefix)
        self.atom_frame_keychain = atom_frame_keychain
        self.bright_frame_keychain = bright_frame_keychain
        self.dark_frame_keychain = dark_frame_keychain
        self.roi_slice = roi_slice

    def load_shot(self, shot_num):
        file_path = self.get_file_path(shot_num)
        with h5py.File(file_path, 'r') as data_h5:
            atom_frame = self.reduce_datasource_by_key(data_h5, self.atom_frame_keychain)
            bright_frame = self.reduce_datasource_by_key(data_h5, self.bright_frame_keychain)
            dark_frame = self.reduce_datasource_by_key(data_h5, self.dark_frame_keychain)
            atom_frame = atom_frame[self.roi_slice]
            bright_frame = bright_frame[self.roi_slice]
            dark_frame = dark_frame[self.roi_slice]
        return atom_frame, bright_frame, dark_frame


class LightLoader(Loader):
    def __init__(self, *, loader_name):
        super(LightLoader, self).__init__(loader_name=loader_name, loader_type=LoaderType.LIGHT)
        self.datamodel = None

    def set_datamodel(self, datamodel):
        self.datamodel = datamodel

    def load_shot(self, shot_num):
        raise NotImplementedError


class ProcessorResultsLoader(LightLoader):
    def __init__(self, *, loader_name, processor_name, result_keychain):
        super(ProcessorResultsLoader, self).__init__(loader_name=loader_name)
        self.processor_name = processor_name

    def load_shot(self, shot_num):
        data_dict = self.datamodel.data_dict
        result = data_dict


class HeavyProcessedLoader(Loader):
    pass


class RawDataStream(InputParamLogger):
    def __init__(self, *, datastream_name, file_prefix):
        self.datastream_name = datastream_name
        self.file_prefix = file_prefix
        self.data_path = None
        self.num_shots = None

    def set_run(self, daily_path, run_name):
        self.data_path = Path(daily_path, 'data', run_name, self.datastream_name)
        self.num_shots = self.get_num_shots()

    def get_file_path(self, shot_num):
        file_name = f'{self.file_prefix}_{shot_num:05d}.h5'
        file_path = Path(self.data_path, file_name)
        return file_path

    def get_num_shots(self):
        file_list = list(self.data_path.glob('*.h5'))
        self.num_shots = len(file_list)
        return self.num_shots