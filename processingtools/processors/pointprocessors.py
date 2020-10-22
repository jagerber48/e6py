import numpy as np
from enum import Enum
from ..imagetools import get_image
from ..datamodel import qprint
from .processor import Processor, ProcessorScale, ProcessorWeight


class PointProcessor(Processor):
    class ResultKey(Enum):
        pass

    def __init__(self, *, name, weight):
        super(PointProcessor, self).__init__(name=name, weight=weight, scale=ProcessorScale.POINT)

    def scaled_process(self, datamodel, processor_dict, quiet=False):
        data_dict = datamodel.data_dict
        num_points = data_dict['num_points']

        for point in range(num_points):
            point_key = f'point-{point:d}'
            try:
                old_processed_shots = list(processor_dict['results'][point_key]['processed_shots'])
            except KeyError:
                old_processed_shots = []
            shots_to_be_processed = list(data_dict['shot_list'][point_key])
            if shots_to_be_processed != old_processed_shots:
                qprint(f'processing {point_key}', quiet=quiet)
                results_dict = self.process_point(point, datamodel)
                processor_dict['results'][point_key] = results_dict
                processor_dict['results'][point_key]['processed_shots'] = shots_to_be_processed
                data_dict.save_dict(quiet=True)
            else:
                qprint(f'skipping processing {point_key}', quiet=quiet)

    def process_point(self, point, datamodel):
        raise NotImplementedError


# class PointProcessor(InputParamLogger):
#     class OutputKey(Enum):
#         pass
#
#     @property
#     def aggregator_type(self):
#         raise NotImplementedError
#
#     def __init__(self, *, aggregator_name):
#         self.aggregator_name = aggregator_name
#
#     def create_aggregator_dict(self, data_dict):
#         aggregator_dict = dict()
#         aggregator_dict['input_param_dict'] = self.input_param_dict
#         aggregator_dict['results'] = dict()
#         data_dict['aggregators'][self.aggregator_name] = aggregator_dict
#         return aggregator_dict
#
#     def check_data_dict(self, data_dict):
#         if self.aggregator_name in data_dict['aggregators']:
#             aggregator_dict = data_dict['aggregators'][self.aggregator_name]
#             old_input_param_dict = aggregator_dict['input_param_dict']
#             if self.input_param_dict != old_input_param_dict:
#                 aggregator_dict = self.create_aggregator_dict(data_dict)
#         else:
#             aggregator_dict = self.create_aggregator_dict(data_dict)
#         return aggregator_dict
#
#     def aggregate_run(self, datamodel, quiet=False):
#         qprint(f'Running {self.aggregator_name} aggregation...', quiet=quiet)
#         data_dict = datamodel.data_dict
#         aggregator_dict = self.check_data_dict(data_dict)
#
#         num_points = data_dict['num_points']
#         for point in range(num_points):
#             point_key = f'point-{point:d}'
#             try:
#                 old_aggregated_shots = list(aggregator_dict['results'][point_key]['aggregated_shots'])
#             except KeyError:
#                 old_aggregated_shots = []
#             shots_to_be_aggregated = list(data_dict['shot_list'][point_key])
#             if shots_to_be_aggregated != old_aggregated_shots:
#                 qprint(f'aggregating {point_key}', quiet=quiet)
#                 results_dict = self.aggregate_point(point, datamodel)
#                 aggregator_dict['results'][point_key] = results_dict
#                 aggregator_dict['results'][point_key]['aggregated_shots'] = shots_to_be_aggregated
#                 data_dict.save_dict(quiet=True)
#             else:
#                 qprint(f'skipping {point_key} aggregation', quiet=quiet)
#
#     def aggregate_point(self, point, datamodel):
#         raise NotImplementedError


class MeanStdPointProcessor(PointProcessor):
    class ResultKey(Enum):
        pass

    def __init__(self, *, name, source_processor_name):
        super(MeanStdPointProcessor, self).__init__(name=name, weight=ProcessorWeight.LIGHT)
        self.source_processor_name = source_processor_name

    def process_point(self, point, datamodel):
        data_dict = datamodel.data_dict
        point_key = f'point-{point:d}'
        shot_list = data_dict['shot_list'][point_key]
        num_loops = data_dict['loop_nums'][point_key]

        source_processor_dict = data_dict['shot_processors'][self.source_processor_name]
        source_processor_name = source_processor_dict['input_param_dict']['kwargs']['name']
        source_processor_results_dict = source_processor_dict['results']

        results_dict = dict()
        results_dict[source_processor_name] = dict()

        # get the keys for the results corresponding to the first shot
        first_shot_dict = next(iter(source_processor_results_dict.values()))

        for key in first_shot_dict.keys():
            avg_value = None
            for shot in shot_list:
                shot_key = f'shot-{shot}'
                value = source_processor_results_dict[shot_key][key]
                if avg_value is None:
                    avg_value = value / num_loops
                else:
                    avg_value += value / num_loops

            std_value = None
            for shot in shot_list:
                shot_key = f'shot-{shot}'
                value = source_processor_results_dict[shot_key][key]
                if std_value is None:
                    std_value = (value - avg_value)**2 / (num_loops - 1)
                else:
                    std_value += (value - avg_value)**2 / (num_loops - 1)

            results_dict[source_processor_name][key] = dict()
            results_dict[source_processor_name][key]['mean'] = avg_value
            results_dict[source_processor_name][key]['std'] = std_value

        return results_dict


class AvgAtomRefImagePointProcessor(PointProcessor):
    class ResultKey(Enum):
        AVERAGE_ATOM_IMG = 'avg_atom_img'
        AVERAGE_REF_IMG = 'avg_ref_img'

    def __init__(self, *, datastream_name, atom_frame_name, ref_frame_name, roi_slice, name):
        super(AvgAtomRefImagePointProcessor, self).__init__(name=name, weight=ProcessorWeight.LIGHT)
        self.datastream_name = datastream_name
        self.atom_frame_name = atom_frame_name
        self.ref_frame_name = ref_frame_name
        self.roi_slice = roi_slice

    def process_point(self, point, datamodel):
        data_dict = datamodel.data_dict
        point_key = f'point-{point:d}'
        datastream = datamodel.datastream_dict[self.datastream_name]
        avg_atom_frame = None
        avg_ref_frame = None
        shot_list = data_dict['shot_list'][point_key]
        num_loops = data_dict['loop_nums'][point_key]
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
        avg_atom_frame = avg_atom_frame / num_loops
        avg_ref_frame = avg_ref_frame / num_loops
        results_dict = dict()
        results_dict[self.ResultKey.AVERAGE_ATOM_IMG.value] = avg_atom_frame
        results_dict[self.ResultKey.AVERAGE_REF_IMG.value] = avg_ref_frame
        return results_dict


class RandomAtomRefImagePointProcessor(PointProcessor):
    class ResultKey(Enum):
        RANDOM_ATOM_IMG = 'random_atom_img'
        RANDOM_REF_IMG = 'random_ref_img'
        RANDOM_SHOT_NUM = 'random_shot_num'

    def __init__(self, *, datastream_name, atom_frame_name, ref_frame_name, roi_slice,
                 name):
        super(RandomAtomRefImagePointProcessor, self).__init__(name=name, weight=ProcessorWeight.LIGHT)
        self.datastream_name = datastream_name
        self.atom_frame_name = atom_frame_name
        self.ref_frame_name = ref_frame_name
        self.roi_slice = roi_slice

    def process_point(self, point, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        random_atom_frame = None
        random_ref_frame = None
        shot_list = datamodel.data_dict['shot_list'][f'point-{point:d}']
        random_shot = np.random.choice(shot_list)
        for shot_num in shot_list:
            if shot_num == random_shot:
                file_path = datastream.get_file_path(shot_num)
                random_atom_frame = get_image(file_path, self.atom_frame_name, roi_slice=self.roi_slice)
                random_ref_frame = get_image(file_path, self.ref_frame_name, roi_slice=self.roi_slice)
        results_dict = dict()
        results_dict[self.ResultKey.RANDOM_ATOM_IMG.value] = random_atom_frame
        results_dict[self.ResultKey.RANDOM_REF_IMG.value] = random_ref_frame
        results_dict[self.ResultKey.RANDOM_SHOT_NUM.value] = random_shot
        return results_dict
