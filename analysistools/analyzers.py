import numpy as np
from .datamodel import Analyzer
from .imagetools import get_image


class CountsAnalyzer(Analyzer):
    def __init__(self, output_field_list, frame_name, datastream, roi_slice, analyzer_name='counts_analyzer'):
        super(CountsAnalyzer, self).__init__(output_field_list, analyzer_name, datastream)
        self.frame_name = frame_name
        self.roi_slice = roi_slice
        self.counts_output = self.output_field_list[0]

    def analyze_shot(self, shot_num=0):
        file_path = self.datastream.get_file_path(shot_num)
        frame = get_image(file_path, self.frame_name, roi_slice=self.roi_slice)
        counts = np.nansum(frame)
        results_dict = dict()
        results_dict[self.counts_output] = counts
        return results_dict
