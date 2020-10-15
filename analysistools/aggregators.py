import numpy as np
from enum import Enum
from .datamodel import Aggregator
from .imagetools import get_image


class AvgAtomRefImageAggregator(Aggregator):
    class OutputKey(Enum):
        AVERAGE_ATOM_IMG = 'avg_atom_img'
        AVERAGE_REF_IMG = 'avg_ref_img'
    aggregator_type = 'AvgAtomRefImageAggregator'

    def __init__(self, datastream_name, atom_frame_name, ref_frame_name, roi_slice, aggregator_name='avg_atom_ref_img_aggregator'):
        super(AvgAtomRefImageAggregator, self).__init__(aggregator_name)
        self.datastream_name = datastream_name
        self.atom_frame_name = atom_frame_name
        self.ref_frame_name = ref_frame_name
        self.roi_slice = roi_slice

    def setup_analyzer_dict(self):
        analyzer_dict = super(AvgAtomRefImageAggregator, self).setup_aggregator_dict()
        analyzer_dict['atom_frame_name'] = self.atom_frame_name
        analyzer_dict['ref_frame_name'] = self.ref_frame_name
        analyzer_dict['roi_slice'] = self.roi_slice

    def aggregate_point(self, shot_list, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        avg_atom_frame = None
        avg_ref_frame = None
        for shot_num in shot_list:
            file_path = datastream.get_file_path(shot_num)
            atom_frame = get_image(file_path, self.atom_frame_name, roi_slice=self.roi_slice)
            ref_frame = get_image(file_path, self.ref_frame_name, roi_slice=self.roi_slice)
            if avg_atom_frame is None:
                avg_atom_frame = atom_frame
                avg_ref_frame = ref_frame
            else:
                avg_atom_frame += atom_frame
                avg_ref_frame += ref_frame
        results_dict = dict()
        results_dict[self.OutputKey.AVERAGE_ATOM_IMG.value] = avg_atom_frame
        results_dict[self.OutputKey.AVERAGE_REF_IMG.value] = avg_atom_frame
        return results_dict


class RandomAtomRefImageAggregator(Aggregator):
    class OutputKey(Enum):
        RANDOM_ATOM_IMG = 'random_atom_img'
        RANDOM_REF_IMG = 'random_ref_img'
        RANDOM_SHOT_NUM = 'random_shot_num'

    aggregator_type = 'RandomAtomRefImageAggregator'

    def __init__(self, datastream_name, atom_frame_name, ref_frame_name, roi_slice, aggregator_name='random_atom_ref_img_aggregator'):
        super(RandomAtomRefImageAggregator, self).__init__(aggregator_name)
        self.datastream_name = datastream_name
        self.atom_frame_name = atom_frame_name
        self.ref_frame_name = ref_frame_name
        self.roi_slice = roi_slice

    def setup_analyzer_dict(self):
        analyzer_dict = super(RandomAtomRefImageAggregator, self).setup_aggregator_dict()
        analyzer_dict['atom_frame_name'] = self.atom_frame_name
        analyzer_dict['ref_frame_name'] = self.ref_frame_name
        analyzer_dict['roi_slice'] = self.roi_slice

    def aggregate_point(self, shot_list, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        random_shot = np.random.choice(shot_list)
        random_atom_frame = None
        random_ref_frame = None
        for shot_num in shot_list:
            if shot_num == random_shot:
                file_path = datastream.get_file_path(shot_num)
                random_atom_frame = get_image(file_path, self.atom_frame_name, roi_slice=self.roi_slice)
                random_ref_frame = get_image(file_path, self.ref_frame_name, roi_slice=self.roi_slice)
        results_dict = dict()
        results_dict[self.OutputKey.RANDOM_ATOM_IMG.value] = random_atom_frame
        results_dict[self.OutputKey.RANDOM_REF_IMG.value] = random_ref_frame
        results_dict[self.OutputKey.RANDOM_SHOT_NUM.value] = random_shot
        return results_dict
