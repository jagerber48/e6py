import numpy as np
from pathlib import Path
import pickle
import h5py


class DataModel:
    def __init__(self, daily_path, run_name, num_points=1, datastream_list=None, analyzer_list=None, reset_hard=False):
        self.daily_path = daily_path
        self.run_name = run_name
        self.num_points = num_points
        self.analyzer_list = analyzer_list

        self.datastream_dict = dict()

        for datastream in datastream_list:
            datastream.set_run(self.daily_path, self.run_name)
            self.datastream_dict[datastream.datastream_name] = datastream

        self.num_shots = datastream_list[0].num_shots
        if not all([datastream.num_shots == self.num_shots for datastream in datastream_list]):
            print('Warning, data streams' +
                  ', '.join([datastream.datastream_name for datastream in datastream_list]) +
                  f' have incommensurate numbers of files. num_shots set to: {self.num_shots}')

        self.data_dict = DataModelDict(self.daily_path, self.run_name, reset_hard=reset_hard)
        self.set_shot_lists()

    def run_analyzers(self):
        for analyzer in self.analyzer_list:
            analyzer.analyze_run(self.data_dict)
        self.data_dict.save_dict()

    def set_shot_lists(self):
        self.data_dict['num_points'] = self.num_points
        self.data_dict['num_shots'] = self.num_shots
        self.data_dict['shot_list'] = dict()
        self.data_dict['loop_nums'] = dict()
        for point in range(self.num_points):
            key = f'point-{point:d}'
            point_shots, point_loops = get_shot_list_from_point(point, self.num_points, self.num_shots)
            self.data_dict['shot_list'][key] = point_shots
            self.data_dict['loop_nums'][key] = point_loops
        self.data_dict.save_dict()


class RawDataStream:
    def __init__(self, datastream_name, file_prefix):
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

    def load_shot_h5(self, shot_num=0):
        file_name = f'{self.file_prefix}_{shot_num:05d}.h5'
        h5_file = h5py.File(file_name, 'r')
        return h5_file

    def get_num_shots(self):
        file_list = list(self.data_path.glob('*.h5'))
        self.num_shots = len(file_list)
        return self.num_shots


class Analyzer:
    def __init__(self, output_field_list, analyzer_name, datastream):
        if not isinstance(output_field_list, (list, tuple)):
            output_field_list = [output_field_list]
            print('tolist')
        self.output_field_list = output_field_list
        self.analyzer_name = analyzer_name
        self.datastream = datastream

    def setup_analyzer_dict(self, num_points=1):
        analyzer_dict = dict()
        for field in self.output_field_list:
            analyzer_dict[field] = dict()
            for point in range(num_points):
                point_key = f'point-{point:d}'
                analyzer_dict[field][point_key] = []
        return analyzer_dict

    def analyze_shot(self, shot_num=0):
        raise NotImplementedError

    def analyze_run(self, data_dict):
        num_points = data_dict['num_points']
        num_shots = data_dict['num_shots']
        analyzer_dict = self.setup_analyzer_dict(num_points)
        data_dict[self.analyzer_name] = analyzer_dict

        result_lists_dict = dict()
        for field in self.output_field_list:
            result_lists_dict[field] = []

        for shot_num in range(num_shots):
            loop, point = shot_to_loop_and_point(shot_num, num_points)
            point_key = f'point-{point:d}'
            results_dict = self.analyze_shot(shot_num)
            for key, value in results_dict.items():
                analyzer_dict[key][point_key].append(value)


class DataModelDict:
    def __init__(self, daily_path, run_name, reset_hard=False):
        self.daily_path = daily_path
        self.run_name = run_name
        self.dir_path = Path(self.daily_path, 'analysis', self.run_name)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.filename = f'{run_name}-datamodel.p'
        self.file_path = Path(self.dir_path, self.filename)
        if not reset_hard:
            self.data_dict = self.load_dict()
        else:
            print('hard reset')
            self.data_dict = dict()
            self.save_dict()

    def load_dict(self):
        try:
            self.data_dict = pickle.load(open(self.file_path, 'rb'))
        except FileNotFoundError as e:
            print(e)
            print(f'Creating {self.filename} in {self.dir_path}')
            self.data_dict = dict()
            self.save_dict()
        return self.data_dict

    def save_dict(self):
        pickle.dump(self.data_dict, open(self.file_path, 'wb'))

    def __getitem__(self, item):
        return self.data_dict[item]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def __repr__(self):
        return self.data_dict.__repr__()

    def __str__(self):
        return self.data_dict.__str__()

    def __delitem__(self, key):
        del self.data_dict[key]

    def keys(self):
        return self.data_dict.keys()


def shot_to_loop_and_point(shot, num_points=1, shot_index_convention=0,
                           loop_index_convention=0, point_index_convention=0):
    """
    Convert shot number to loop and point using the number of points. Default assumption is indexing for
    shot, loop, and point all starts from zero with options for other conventions.
    """
    shot_ind = shot - shot_index_convention
    loop_ind = shot_ind // num_points
    point_ind = shot_ind % num_points
    loop = loop_ind + loop_index_convention
    point = point_ind + point_index_convention
    return loop, point

def get_shot_list_from_point(point, num_points, num_shots, start_shot=0, stop_shot=None):
    # TODO: Implement different conventions for shot and point start indices
    shots = np.arange(point, num_shots, num_points)
    start_mask = start_shot <= shots
    shots = shots[start_mask]
    if stop_shot is not None:
        stop_mask = shots <= stop_shot
        shots = shots[stop_mask]
    num_loops = len(shots)
    return shots, num_loops
