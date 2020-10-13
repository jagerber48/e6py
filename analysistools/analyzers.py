import numpy as np
from .datamodel import RawShotAnalyzer
from .imagetools import get_image


class CountsAnalyzer(RawShotAnalyzer):
    def __init__(self, output_field_list, frame_name, datastream_name, roi_slice, analyzer_name='counts_analyzer'):
        super(CountsAnalyzer, self).__init__(output_field_list, analyzer_name, datastream_name)
        self.frame_name = frame_name
        self.roi_slice = roi_slice
        self.analyzer_type = 'CountsAnalyzer'

        self.counts_output = self.output_field_list[0]

    def setup_input_param_dict(self):
        super(CountsAnalyzer, self).setup_input_param_dict()
        self.input_param_dict['frame_name'] = self.frame_name
        self.input_param_dict['roi_slice'] = self.roi_slice
        self.input_param_dict['analyzer_type'] = self.analyzer_type

    def analyze_shot(self, datastream, shot_num=0):
        file_path = datastream.get_file_path(shot_num)
        frame = get_image(file_path, self.frame_name, roi_slice=self.roi_slice)
        counts = np.nansum(frame)
        results_dict = dict()
        results_dict[self.counts_output] = counts
        return results_dict
