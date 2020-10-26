from pathlib import Path
import pickle
from .datatools import to_list, qprint, get_shot_list_from_point, InputParamLogger


def print_dict_tree(dict_tree, level=0):
    for key in dict_tree.keys():
        print(f'{"  "*level}{level}-{key}')
        if isinstance(dict_tree[key], dict):
            print_dict_tree(dict_tree[key], level+1)


class DataModel:
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

        self.datastream_dict = dict()
        self.shot_processor_dict = dict()
        self.point_processor_dict = dict()
        self.reporter_dict = dict()
        self.datafield_dict = dict()

        self.load_datamodel(reset_hard)

        self.num_shots = 0
        self.datastream_dict = dict()
        self.initialize_datastreams()

        self.set_shot_lists()

    def load_datamodel(self, reset_hard):
        self.data_dict = DataModelDict(self.daily_path, self.run_name, reset_hard=reset_hard)
        self.load_datastream()
        self.load_shot_processors()
        self.load_point_processors()
        self.load_reporters()
        self.load_datafields()

    def load_datastream(self):
        if 'datastreams' in self.data_dict:
            datastream_dict = self.data_dict['datastreams']
            for input_param_dict in datastream_dict:
                datastream = InputParamLogger.rebuild(input_param_dict)
                datastream.set_run(self.daily_path, self.run_name)
                self.datastream_dict[datastream.datastream_name] = datastream
                datastream.make_data_fields(datamodel=self)
        else:
            self.data_dict['datastreams'] = dict()

    def load_shot_processors(self):
        if 'shot_processors' in self.data_dict:
            shot_processor_dict = self.data_dict['shot_processors']
            for input_param_dict in shot_processor_dict:
                shot_processor = InputParamLogger.rebuild(input_param_dict)
                self.shot_processor_dict[shot_processor.processor_name] = shot_processor
        else:
            self.data_dict['shot_processors'] = dict()

    def load_point_processors(self):
        if 'point_processors' in self.data_dict:
            point_processor_dict = self.data_dict['point_processors']
            for input_param_dict in point_processor_dict:
                point_processor = InputParamLogger.rebuild(input_param_dict)
                self.point_processor_dict[point_processor.processor_name] = point_processor
        else:
            self.data_dict['point_processors'] = dict()

    def load_reporters(self):
        if 'reporters' in self.data_dict:
            reporter_dict = self.data_dict['reporters']
            for input_param_dict in reporter_dict:
                reporter = InputParamLogger.rebuild(input_param_dict)
                self.reporter_dict[reporter.reporter_name] = reporter
        else:
            self.data_dict['reporters'] = dict()

    def load_datafields(self):
        if 'datafields' in self.data_dict:
            datafield_dict = self.data_dict['datafields']
            for input_param_dict in datafield_dict:
                datafield = InputParamLogger.rebuild(input_param_dict)
                self.datafield_dict[datafield.field_name] = datafield
        else:
            self.data_dict['datafields'] = dict()

    def initialize_datastreams(self):
        for datastream in self.datastream_list:
            datastream.set_run(self.daily_path, self.run_name)
            self.datastream_dict[datastream.datastream_name] = datastream
            new_data_field_list = datastream.make_data_fields(datamodel=self)
            for data_field in new_data_field_list:
                self.add_data_field(data_field)

        self.num_shots = self.datastream_list[0].num_shots
        if not all([datastream.num_shots == self.num_shots for datastream in self.datastream_list]):
            print('Warning, data streams' +
                  ', '.join([datastream.datastream_name for datastream in self.datastream_list]) +
                  f' have incommensurate numbers of files. num_shots set to: {self.num_shots}')

    def add_data_field(self, data_field):
        self.datafield_dict[data_field.field_name] = data_field

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
