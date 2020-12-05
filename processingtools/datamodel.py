from pathlib import Path
import pickle
from .datatools import to_list, qprint, get_shot_list_from_point, InputParamLogger


def print_dict_tree(dict_tree, level=0):
    for key in dict_tree.keys():
        print(f'{"  "*level}{level}-{key}')
        if isinstance(dict_tree[key], dict):
            print_dict_tree(dict_tree[key], level+1)


def reload_data_model(*, daily_path, run_name, run_doc_string, num_points,
                      datastream_list, shot_processor_list,
                      point_processor_list, reporter_list,
                      reset_hard, quiet):
    data_model_dir = Path(daily_path, 'analysis', run_name)
    data_model_filename = f'{run_name}-datamodel.p'
    data_model_path = Path(data_model_dir, data_model_filename)

    if reset_hard:
        qprint(f'Creating {data_model_filename} in {data_model_path}', quiet=quiet)
        datamodel = create_data_model(daily_path=daily_path, run_name=run_name, run_doc_string=run_doc_string,
                                      num_points=num_points, quiet=quiet)
    else:
        if data_model_path.exists():
            datamodel = load_data_model(daily_path, run_name, quiet=quiet)
        else:
            qprint(f'Creating {data_model_filename} in {data_model_path}', quiet=quiet)
            datamodel = create_data_model(daily_path=daily_path, run_name=run_name, run_doc_string=run_doc_string,
                                          num_points=num_points, quiet=quiet)

    add_to_datamodel(datamodel=datamodel, datastream_list=datastream_list, shot_processor_list=shot_processor_list,
                     point_processor_list=point_processor_list, reporter_list=reporter_list)

    return datamodel


def load_data_model(daily_path, run_name, quiet=False):
    data_model_dir = Path(daily_path, 'analysis', run_name)
    data_model_filename = f'{run_name}-datamodel.p'
    data_model_path = Path(data_model_dir, data_model_filename)

    qprint(f'Loading data_dict from {data_model_path}', quiet=quiet)
    loaded_dict = pickle.load(open(data_model_path, 'rb'))
    run_doc_string = loaded_dict['run_doc_string']
    num_points = loaded_dict['num_points']
    data_dict = DataModelDict(daily_path, run_name, run_doc_string, num_points, quiet)
    data_dict.data_dict = loaded_dict
    datamodel = DataModel(data_dict=data_dict, quiet=quiet)
    return datamodel


def create_data_model(*, daily_path, run_name, run_doc_string, num_points, quiet):
    data_dict = DataModelDict(daily_path=daily_path, run_name=run_name,
                              run_doc_string=run_doc_string, num_points=num_points)
    datamodel = DataModel(data_dict, quiet=quiet)
    return datamodel


def add_to_datamodel(*, datamodel, datastream_list, shot_processor_list, point_processor_list, reporter_list):
    datastream_list = to_list(datastream_list)
    shot_processor_list = to_list(shot_processor_list)
    point_processor_list = to_list(point_processor_list)
    reporter_list = to_list(reporter_list)

    for datastream in datastream_list:
        datamodel.add_datastream(datastream)
    datamodel.set_shot_lists()
    for shot_processor in shot_processor_list:
        datamodel.add_shot_processor(shot_processor)
    for point_processor in point_processor_list:
        datamodel.add_point_processor(point_processor)
    for reporter in reporter_list:
        datamodel.add_reporter(reporter)


class DataModel:
    def __init__(self, data_dict, quiet):
        self.data_dict = data_dict
        self.daily_path = self.data_dict['daily_path']
        self.run_name = self.data_dict['run_name']
        self.num_points = self.data_dict['num_points']
        self.run_doc_string = self.data_dict['run_doc_string']
        self.quiet = quiet

        self.datastream_dict = dict()
        self.shot_processor_dict = dict()
        self.point_processor_dict = dict()
        self.reporter_dict = dict()
        self.datafield_dict = dict()

        self.load_datamodel()
        self.data_dict.save_dict(quiet=self.quiet)

    @staticmethod
    def add_subdict(parent_dict, child_dict_name, overwrite=False):
        if child_dict_name not in parent_dict or overwrite:
            parent_dict[child_dict_name] = dict()

    def add_datastream(self, datastream):
        name = datastream.datastream_name
        datastream.set_run(self.daily_path, self.run_name)
        datastream.make_data_fields(datamodel=self)

        self.datastream_dict[name] = datastream
        self.data_dict['datastreams'][name] = datastream.input_param_dict
        self.data_dict['num_shots'] = datastream.num_shots
        self.data_dict.save_dict(quiet=True)

    def add_shot_processor(self, shot_processor):
        name = shot_processor.processor_name
        if name in self.data_dict['shot_processors']:
            old_input_params = self.data_dict['shot_processors'][name]
            new_input_params = shot_processor.input_param_dict
            if new_input_params != old_input_params:
                print(f'overwriting and resetting shot processor: {name}')
                shot_processor.reset = True
        self.shot_processor_dict[name] = shot_processor
        self.data_dict['shot_processors'][name] = shot_processor.input_param_dict
        self.data_dict.save_dict(quiet=self.quiet)
    def add_point_processor(self, point_processor):
        name = point_processor.processor_name
        if name in self.data_dict['point_processors']:
            old_input_params = self.data_dict['point_processors'][name]
            new_input_params = point_processor.input_param_dict
            if new_input_params != old_input_params:
                print(f'overwriting and resetting point processor: {name}')
                point_processor.reset = True
        self.point_processor_dict[name] = point_processor
        self.data_dict['point_processors'][name] = point_processor.input_param_dict

    def add_reporter(self, reporter):
        name = reporter.reporter_name
        self.reporter_dict[name] = reporter
        self.data_dict['reporters'][name] = reporter.input_param_dict
        self.data_dict.save_dict(quiet=True)

    def add_datafield(self, datafield):
        name = datafield.field_name
        self.datafield_dict[name] = datafield
        self.data_dict['datafields'][name] = datafield.input_param_dict
        self.data_dict.save_dict(quiet=True)

    def load_datamodel(self):
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
        for shot_processor in self.shot_processor_dict.values():
            shot_processor.process(self, quiet=self.quiet)

        for point_processor in self.point_processor_dict.values():
            point_processor.process(self, quiet=self.quiet)
        self.data_dict.save_dict()

    def run_reporters(self):
        for reporter in self.reporter_dict.values():
            reporter.report(self)

    def print_dict_tree(self):
        print_dict_tree(self.data_dict, level=0)

    def set_shot_lists(self):
        num_shots = self.data_dict['num_shots']
        num_points = self.data_dict['num_points']
        if 'num_points' in self.data_dict:
            if self.data_dict['num_points'] != self.num_points:
                self.data_dict['point_processors'] = dict()
        self.data_dict['shot_list'] = dict()
        self.data_dict['loop_nums'] = dict()
        for point in range(self.num_points):
            key = f'point-{point:d}'
            point_shots, point_loops = get_shot_list_from_point(point, num_points, num_shots)
            self.data_dict['shot_list'][key] = point_shots
            self.data_dict['loop_nums'][key] = point_loops
        self.data_dict.save_dict(quiet=self.quiet)


class DataModelDict:
    def __init__(self, daily_path, run_name, run_doc_string, num_points, quiet=False):
        self.daily_path = daily_path
        self.run_name = run_name
        self.run_doc_string = run_doc_string
        self.num_points = num_points
        self.quiet = quiet

        dir_path = Path(self.daily_path, 'analysis', self.run_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        filename = f'{run_name}-datamodel.p'
        self.file_path = Path(dir_path, filename)

        self.data_dict = dict()
        self.data_dict['daily_path'] = self.daily_path
        self.data_dict['run_name'] = self.run_name
        self.data_dict['run_doc_string'] = self.run_doc_string
        self.data_dict['num_points'] = num_points
        sub_dict_list = ['datastreams', 'shot_processors', 'point_processors', 'reporters', 'datafields',
                         'shot_data', 'point_data']
        for sub_dict in sub_dict_list:
            self.data_dict[sub_dict] = dict()
        self.save_dict(quiet=self.quiet)

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
