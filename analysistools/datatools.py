import numpy as np
from pathlib import Path
import pickle


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


def loop_and_point_to_shot(loop, point, num_points=1, shot_index_convention=0,
                           loop_index_convention=0, point_index_convention=0):
    """
    Convert loop and point to shot number using the number of points. Default assumption is indexing for
    shot, loop, and point all starts from zero with options for other conventions.
    """
    loop_ind = loop - loop_index_convention
    point_ind = point - point_index_convention
    shot_ind = loop_ind * num_points + point_ind
    shot = shot_ind + shot_index_convention
    return shot


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


def get_num_files(data_path):
    file_list = list(data_path.glob('*.h5'))
    num_files = len(file_list)
    return num_files


def get_datastream_path(daily_path, run_name, datastream_name):
    datastream_path = Path(daily_path, 'data', run_name, datastream_name)
    analysis_path = Path(daily_path, 'analysis', run_name)
    analysis_path.mkdir(parents=True, exist_ok=True)
    return datastream_path


class AnalysisDict:
    def __init__(self, daily_path, run_name):
        self.daily_path = daily_path
        self.run_name = run_name
        self.analysis_path = Path(self.daily_path, 'analysis', self.run_name)
        self.analysis_path.mkdir(parents=True, exist_ok=True)
        self.filename = f'{run_name}-analysis.p'
        self.filepath = Path(self.analysis_path, self.filename)
        self.analysis_dict = self.load_dict()

    def load_dict(self):
        try:
            self.analysis_dict = pickle.load(open(self.filepath, 'rb'))
        except FileNotFoundError as e:
            print(e)
            print(f'Creating {self.filename} in {self.analysis_path}')
            self.analysis_dict = dict()
            self.save_dict()
        return self.analysis_dict

    def save_dict(self):
        pickle.dump(self.analysis_dict, open(self.filepath, 'wb'))

    def set_shot_lists(self, num_shots, num_points=1, start_shot=0, stop_shot=None):
        self['num_points'] = num_points
        self['start_shot'] = start_shot
        self['num_shots'] = num_shots
        self['shot_list'] = dict()
        self['loop_nums'] = dict()
        for point in range(num_points):
            key = f'point-{point:d}'
            point_shots, point_loops = get_shot_list_from_point(point, num_points, num_shots,
                                                                start_shot=start_shot,
                                                                stop_shot=stop_shot)
            self['shot_list'][key] = point_shots
            self['loop_nums'][key] = point_loops
        self.save_dict()

    def __getitem__(self, item):
        return self.analysis_dict[item]

    def __setitem__(self, key, value):
        self.analysis_dict[key] = value

    def __repr__(self):
        return self.analysis_dict.__repr__()

    def __str__(self):
        return self.analysis_dict.__str__()

    def keys(self):
        return self.analysis_dict.keys()
