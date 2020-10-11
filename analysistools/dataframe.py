from pathlib import Path
import pickle


class DataFrame:
    def __init__(self, daily_path, run_name):
        self.daily_path = daily_path
        self.run_name = run_name

    def setup_analyis_dict(self):
        self.analysis_dict = AnalysisDict(self.daily_path, self.run_name)

    def run_analyzer(self, analyzer_key):
        pass

class Analyzer:
    def __init__(self):
        pass

    def analyze_shot(self):
        raise NotImplementedError

class RawDataStream:
    def __init__(self, daily_path, run_name, datastream_name, file_prefix):
        self.daily_path = daily_path
        self.run_name = run_name
        self.datastream_name = datastream_name
        self.file_prefix = file_prefix

class AnalysisDict:
    def __init__(self, daily_path, run_name):
        self.daily_path = daily_path
        self.run_name = run_name
        self.analysis_path = Path(self.daily_path, 'analysis', self.run_name)
        self.analysis_path.mkdir(parents=True, exist_ok=True)
        filename = f'{run_name}-analysis.p'
        self.filepath = Path(self.analysis_path, filename)
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

    def __delitem__(self, key):
        del self.analysis_dict[key]

    def keys(self):
        return self.analysis_dict.keys()
