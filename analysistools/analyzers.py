import numpy as np
from enum import Enum
from .datamodel import Analyzer
from .imagetools import get_image


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
