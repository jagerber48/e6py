import numpy as np
from enum import Enum
from .imagetools import get_image


class Analyzer:
    class OutputKey(Enum):
        pass

    @property
    def analyzer_type(self):
        raise NotImplementedError

    def __init__(self, analyzer_name='analyzer'):
        self.analyzer_name = analyzer_name

        self.input_param_dict = dict()
        self.analyzer_dict = None

    def setup_input_param_dict(self):
        self.input_param_dict = dict()
        self.input_param_dict['analyzer_name'] = self.analyzer_name
        self.input_param_dict['analyzer_type'] = self.analyzer_type

    def setup_analyzer_dict(self):
        analyzer_dict = dict()
        self.setup_input_param_dict()
        analyzer_dict['input_params'] = self.input_param_dict
        for enum in self.OutputKey:
            key = enum.value
            analyzer_dict[key] = dict()
        return analyzer_dict

    def analyze_run(self, datamodel):
        data_dict = datamodel.data_dict
        analyzer_dict = self.setup_analyzer_dict()
        num_shots = data_dict['num_shots']

        for shot_num in range(num_shots):
            shot_key = f'shot-{shot_num:d}'
            results_dict = self.analyze_shot(shot_num, datamodel)
            for key, value in results_dict.items():
                analyzer_dict[key][shot_key] = value

        data_dict['analyzers'][self.analyzer_name] = analyzer_dict

    def analyze_shot(self, shot_num, datamodel):
        raise NotImplementedError


class CountsAnalyzer(Analyzer):
    class OutputKey(Enum):
        COUNTS = 'counts'
    analyzer_type = 'CountsAnalyzer'

    def __init__(self, datastream_name, frame_name, roi_slice=None, analyzer_name='counts_analyzer'):
        super(CountsAnalyzer, self).__init__(analyzer_name)
        self.datastream_name = datastream_name
        self.frame_name = frame_name
        self.roi_slice = roi_slice

    def setup_input_param_dict(self):
        super(CountsAnalyzer, self).setup_input_param_dict()
        self.input_param_dict['datastream_name'] = self.datastream_name
        self.input_param_dict['frame_name'] = self.frame_name
        self.input_param_dict['roi_slice'] = self.roi_slice

    def analyze_shot(self, shot_num, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        file_path = datastream.get_file_path(shot_num)
        frame = get_image(file_path, self.frame_name, roi_slice=self.roi_slice)
        counts = np.nansum(frame)
        results_dict = {self.OutputKey.COUNTS.value: counts}
        return results_dict
