from functools import reduce
import numpy as np
from pathlib import Path
import pickle


def qprint(text, quiet=False):
    if not quiet:
        print(text)


def print_dict_tree(dict_tree, level=0):
    for key in dict_tree.keys():
        print(f'{"  "*level}{level}-{key}')
        if isinstance(dict_tree[key], dict):
            print_dict_tree(dict_tree[key], level+1)


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


def dataset_from_keychain(datamodel, keychain):
    data_dict = datamodel.data_dict
    data = reduce(lambda x, y: x[y], keychain.split('/'), data_dict)
    return data


def to_list(var):
    """
    Helper function to convert singleton input parameters into list-expecting parameters into singleton lists.
    """
    if isinstance(var, (list, tuple)):
        return var
    else:
        return [var]


class InputParamLogger:
    def __new__(cls, *args, **kwargs):
        input_param_dict = {'args': args, 'kwargs': kwargs}
        obj = super(InputParamLogger, cls).__new__(cls)
        obj.input_param_dict = input_param_dict
        return obj


class DataModel:
    def __init__(self, daily_path, run_name, num_points=1, datastream_list=None, shot_processor_list=None,
                 point_processor_list=None, reporter_list=None, reset_hard=False, quiet=False):
        self.daily_path = daily_path
        self.run_name = run_name
        self.num_points = num_points
        self.datastream_list = to_list(datastream_list)
        self.shot_processor_list = to_list(shot_processor_list)
        self.point_processor_list = point_processor_list
        self.reporter_list = to_list(reporter_list)
        self.quiet = quiet

        self.datastream_dict = dict()
        self.initialize_datastreams()

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

    def initialize_datastreams(self):
        for datastream in self.datastream_list:
            datastream.set_run(self.daily_path, self.run_name)
            self.datastream_dict[datastream.datastream_name] = datastream

    def set_num_shots(self):
        self.num_shots = self.datastream_list[0].num_shots
        if not all([datastream.num_shots == self.num_shots for datastream in self.datastream_list]):
            print('Warning, data streams' +
                  ', '.join([datastream.datastream_name for datastream in self.datastream_list]) +
                  f' have incommensurate numbers of files. num_shots set to: {self.num_shots}')

    def process(self):
        qprint(f'***Processing run: {self.run_name}***', quiet=self.quiet)
        if 'shot_processors' not in self.data_dict:
            self.data_dict['shot_processors'] = dict()
        for shot_processor in self.shot_processor_list:
            shot_processor.process(self, quiet=self.quiet)

        if 'point_processors' not in self.data_dict:
            self.data_dict['point_processors'] = dict()
        for point_processor in self.point_processor_list:
            point_processor.process(self, quiet=self.quiet)

        self.data_dict.save_dict()

    def run_reporters(self):
        for reporter in self.reporter_list:
            reporter.report(self)

    def print_dict_tree(self):
        print_dict_tree(self.data_dict, level=0)

    def set_shot_lists(self):
        if 'num_points' in self.data_dict:
            if self.data_dict['num_points'] != self.num_points:
                self.data_dict['point_processors'] = dict()
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


class DataModelDict:
    def __init__(self, daily_path, run_name, reset_hard=False, quiet=False):
        self.daily_path = daily_path
        self.run_name = run_name
        self.quiet=quiet
        self.dir_path = Path(self.daily_path, 'analysis', self.run_name)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.filename = f'{run_name}-datamodel.p'
        self.file_path = Path(self.dir_path, self.filename)
        if not reset_hard:
            self.data_dict = self.load_dict(quiet=self.quiet)
        else:
            self.create_dict()

    def create_dict(self):
        self.data_dict = dict()
        self.data_dict['daily_path'] = self.daily_path
        self.data_dict['run_name'] = self.run_name
        self.save_dict(quiet=self.quiet)

    def load_dict(self, quiet=False):
        try:
            qprint(f'Loading data_dict from {self.file_path}', quiet)
            self.data_dict = pickle.load(open(self.file_path, 'rb'))
        except (FileNotFoundError, EOFError) as e:
            qprint(e, quiet=quiet)
            qprint(f'Creating {self.filename} in {self.dir_path}', quiet=quiet)
            self.create_dict()
        return self.data_dict

    def save_dict(self, quiet=False):
        qprint(f'Saving data_dict to {self.file_path}', quiet=quiet)
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

    def __iter__(self):
        return iter(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def keys(self):
        return self.data_dict.keys()

    def items(self):
        return self.data_dict.items()
