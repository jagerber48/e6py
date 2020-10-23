from pathlib import Path
import h5py
from .datamodel import InputParamLogger


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


def get_gagescope_trace(file_path, channel_name, segment_name):
    h5 = h5py.File(file_path, 'r')
    channel_data = h5[channel_name]
    sample_offset = channel_data.attrs['sample_offset']
    sample_res = channel_data.attrs['sample_res']
    sample_range = channel_data.attrs['input_range']
    offset_v = channel_data.attrs['dc_offset']
    data = channel_data[segment_name]
    scaled_data = ((sample_offset - data) / sample_res) * (sample_range / 2000.0) + offset_v
    dt = data.attrs['dx']
    return scaled_data, dt
