import numpy as np
from .datamodel import RawAggregator
from .imagetools import get_image


class AvgImageAggregator(RawAggregator):
    def __init__(self, output_field_list, frame_name, datastream_name, roi_slice,
                 aggregator_name='avg_img_aggregator'):
        super(AvgImageAggregator, self).__init__(output_field_list, aggregator_name, datastream_name)
        self.frame_name = frame_name
        self.roi_slice = roi_slice
        self.avg_img_output = self.output_field_list[0]

    def setup_analyzer_dict(self, num_points=1):
        analyzer_dict = super(AvgImageAggregator, self).setup_aggregator_dict(num_points=num_points)
        analyzer_dict['frame_name'] = self.frame_name
        analyzer_dict['roi_slice'] = self.roi_slice

    def aggregate_point(self, datastream, shot_list):
        avg_frame = None
        for shot_num in shot_list:
            file_path = datastream.get_file_path(shot_num)
            frame = get_image(file_path, self.frame_name, roi_slice=self.roi_slice)
            if avg_frame is None:
                avg_frame = frame
            else:
                avg_frame += frame
        results_dict = dict()
        results_dict[self.avg_img_output] = avg_frame
        return results_dict


class RandomImageAggregator(RawAggregator):
    def __init__(self, output_field_list, frame_name, datastream_name, roi_slice,
                 aggregator_name='avg_img_aggregator'):
        super(RandomImageAggregator, self).__init__(output_field_list, aggregator_name, datastream_name)
        self.frame_name = frame_name
        self.roi_slice = roi_slice
        self.avg_img_output = self.output_field_list[0]

    def setup_analyzer_dict(self, num_points=1):
        analyzer_dict = super(RandomImageAggregator, self).setup_aggregator_dict(num_points=num_points)
        analyzer_dict['frame_name'] = self.frame_name
        analyzer_dict['roi_slice'] = self.roi_slice

    def aggregate_point(self, datastream, shot_list):
        random_shot = np.random.choice(shot_list)
        random_frame = None
        for shot_num in shot_list:
            if shot_num == random_shot:
                file_path = datastream.get_file_path(shot_num)
                random_frame = get_image(file_path, self.frame_name, roi_slice=self.roi_slice)
        results_dict = dict()
        results_dict[self.avg_img_output] = random_frame
        return results_dict
