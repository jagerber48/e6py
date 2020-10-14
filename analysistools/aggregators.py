import numpy as np
from .datamodel import Aggregator
from .imagetools import get_image


class AvgImageAggregator(Aggregator):
    output_key_list = ['avg_img']
    aggregator_type = 'AvgImageAggregator'

    def __init__(self, datastream_name, frame_name, roi_slice, aggregator_name='avg_img_aggregator'):
        super(AvgImageAggregator, self).__init__(aggregator_name)
        self.datastream_name = datastream_name
        self.frame_name = frame_name
        self.roi_slice = roi_slice
        self.avg_img_output = self.output_key_list[0]

    def setup_analyzer_dict(self):
        analyzer_dict = super(AvgImageAggregator, self).setup_aggregator_dict()
        analyzer_dict['frame_name'] = self.frame_name
        analyzer_dict['roi_slice'] = self.roi_slice

    def aggregate_point(self, shot_list, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        avg_frame = None
        for shot_num in shot_list:
            file_path = datastream.get_file_path(shot_num)
            frame = get_image(file_path, self.frame_name, roi_slice=self.roi_slice)
            if avg_frame is None:
                avg_frame = frame
            else:
                avg_frame += frame
        results_dict = {self.avg_img_output: avg_frame}
        return results_dict


class RandomImageAggregator(Aggregator):
    output_key_list = ['random_img']
    aggregator_type = 'RandomImageAggregator'

    def __init__(self, datastream_name, frame_name, roi_slice, aggregator_name='random_img_aggregator'):
        super(RandomImageAggregator, self).__init__(aggregator_name)
        self.datastream_name = datastream_name
        self.frame_name = frame_name
        self.roi_slice = roi_slice
        self.random_img_output = self.output_key_list[0]

    def setup_analyzer_dict(self):
        analyzer_dict = super(RandomImageAggregator, self).setup_aggregator_dict()
        analyzer_dict['frame_name'] = self.frame_name
        analyzer_dict['roi_slice'] = self.roi_slice

    def aggregate_point(self, shot_list, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        random_shot = np.random.choice(shot_list)
        random_frame = None
        for shot_num in shot_list:
            if shot_num == random_shot:
                file_path = datastream.get_file_path(shot_num)
                random_frame = get_image(file_path, self.frame_name, roi_slice=self.roi_slice)
        results_dict = {self.random_img_output: random_frame}
        return results_dict
