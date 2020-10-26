from pathlib import Path
import pickle
from .datatools import to_list, qprint, get_shot_list_from_point, InputParamLogger


def print_dict_tree(dict_tree, level=0):
    for key in dict_tree.keys():
        print(f'{"  "*level}{level}-{key}')
        if isinstance(dict_tree[key], dict):
            print_dict_tree(dict_tree[key], level+1)


class DataModel(InputParamLogger):
    def __init__(self, daily_path, run_name, num_points=1, datastream_list=None, shot_processor_list=None,
                 point_processor_list=None, reporter_list=None, reset_hard=False, quiet=False):
        self.daily_path = daily_path
        self.run_name = run_name
        self.num_points = num_points
        self.datastream_list = to_list(datastream_list)
        self.shot_processor_list = to_list(shot_processor_list)
        self.point_processor_list = to_list(point_processor_list)
        self.reporter_list = to_list(reporter_list)
        self.quiet = quiet

        self.data_dict = DataModelDict(self.daily_path, self.run_name, reset_hard=reset_hard)
        self.initialize_data_dict()

        self.datastream_dict = dict()
        self.shot_processor_dict = dict()
        self.point_processor_dict = dict()
        self.reporter_dict = dict()
        self.datafield_dict = dict()

        self.load_datamodel(reset_hard)

        self.num_shots = None
        self.datastream_dict = dict()
        self.initialize_datastreams()

        self.set_shot_lists()

    @staticmethod
    def add_subdict(parent_dict, child_dict_name, overwrite=False):
        if child_dict_name not in parent_dict or overwrite:
            parent_dict[child_dict_name] = dict()

    def initialize_data_dict(self):
        sub_dict_list = ['datastreams', 'shot_processors', 'point_processors', 'reporters', 'datafields']
        for dict_name in sub_dict_list:
            self.add_subdict(self.data_dict, dict_name, overwrite=False)

    def add_from_input_params(self, datastream_list, shot_processor_list, point_processor_list, reporter_list):
        for datastream in datastream_list:
            self.data_dict['datastreams'][datastream.datastream_name] = datastream.input_param_dict
        for shot_processor in shot_processor_list:
            self.data_dict['shot_processors'][shot_processor.processor_name] = shot_processor.input_param_dict
        for point_processor in point_processor_list:
            self.data_dict['point_processors'][point_processor.processor_name] = point_processor.input_param_dict
        for reporter in reporter_list:
            self.data_dict['reporters'][reporter.reporter_name] = reporter.input_param_dict

    def add_datastream(self, datastream):
        name = datastream.datastream_name
        self.datastream_dict[name] = datastream
        self.data_dict['datastreams'][name] = datastream.input_param_dict

        datastream.set_run(self.daily_path, self.run_name)
        if self.num_shots is None:
            self.num_shots = datastream.num_shots
        elif datastream.num_shots != self.num_shots:
            print(f'Warning, num_shots for datastream: "{name}" incommensurate with datamodel num_shots!')

    def add_shot_processor(self, shot_processor):
        name = shot_processor.processor_name
        self.shot_processor_dict[name] = shot_processor
        self.data_dict['shot_processors'][name] = shot_processor.input_param_dict

    def add_point_processor(self, point_processor):
        name = point_processor.processor_name
        self.point_processor_dict[name] = point_processor
        self.data_dict['point_processors'][name] = point_processor.input_param_dict

    def add_reporter(self, reporter):
        name = reporter.reporter_name
        self.reporter_dict[name] = reporter
        self.data_dict['reporters'][name] = reporter.input_param_dict

    def add_datafield(self, datafield):
        name = datafield.field_name
        self.datafield_dict[name] = datafield
        self.data_dict['datafields'][name] = datafield.input_param_dict

    def load_datamodel(self, reset_hard):
        self.data_dict = DataModelDict(self.daily_path, self.run_name, reset_hard=reset_hard)
        self.load_datastreams()
        self.load_shot_processors()
        self.load_point_processors()
        self.load_reporters()
        self.load_datafields()

    def load_datastreams(self):
        datastream_dict = self.data_dict['datastreams']
        for input_param_dict in datastream_dict.values():
            datastream = InputParamLogger.rebuild(input_param_dict)
            self.add_datastream(datastream)

    def load_shot_processors(self):
        shot_processor_dict = self.data_dict['shot_processors']
        for input_param_dict in shot_processor_dict.values():
            shot_processor = InputParamLogger.rebuild(input_param_dict)
            self.add_shot_processor(shot_processor)

    def load_point_processors(self):
        point_processor_dict = self.data_dict['point_processors']
        for input_param_dict in point_processor_dict.values():
            point_processor = InputParamLogger.rebuild(input_param_dict)
            self.add_point_processor(point_processor)

    def load_reporters(self):
        reporter_dict = self.data_dict['reporters']
        for input_param_dict in reporter_dict.values():
            reporter = InputParamLogger.rebuild(input_param_dict)
            self.add_reporter(reporter)

    def load_datafields(self):
        datafield_dict = self.data_dict['datafields']
        for input_param_dict in datafield_dict.values():
            datafield = InputParamLogger.rebuild(input_param_dict)
            self.add_datafield(datafield)
            self.datafield_dict[datafield.field_name] = datafield

    def get_data(self, datafield_name, shot_num):
        data = self.datafield_dict[datafield_name].get_data(shot_num)
        return data

    def set_data(self, datafield_name, shot_num, data):
        self.datafield_dict[datafield_name].set_data(shot_num, data)

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
        self.quiet = quiet
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
