import numpy as np
from enum import Enum
from ..datatools import qprint
from ..datafield import DataDictField
from .processor import Processor, ProcessorScale


class PointProcessor(Processor):
    class ResultKey(Enum):
        pass

    def __init__(self, *, processor_name, reset):
        super(PointProcessor, self).__init__(processor_name=processor_name, scale=ProcessorScale.POINT)
        self.reset = reset

    def scaled_process(self, datamodel, quiet=False):
        data_dict = datamodel.data_dict
        num_points = data_dict['num_points']
        processor_output_dict = data_dict['point_data'][self.processor_name]
        if 'processed_shots' not in processor_output_dict:
            processor_output_dict['processed_shots'] = dict()
            for point_num in range(num_points):
                point_key = f'point-{point_num:d}'
                processor_output_dict['processed_shots'][point_key] = []

        for point in range(num_points):
            point_key = f'point-{point:d}'
            try:
                old_processed_shots = list(processor_output_dict['processed_shots'][point_key])
            except KeyError:
                old_processed_shots = []
            shots_to_be_processed = list(data_dict['shot_list'][point_key])
            if shots_to_be_processed != old_processed_shots:
                qprint(f'processing {point_key}', quiet=quiet)
                self.process_point(point, datamodel)
                processor_output_dict['processed_shots'][point_key] = shots_to_be_processed
                data_dict.save_dict(quiet=True)
            else:
                qprint(f'skipping processing {point_key}', quiet=quiet)

    def process_point(self, point, datamodel):
        raise NotImplementedError


class MeanStdPointProcessor(PointProcessor):
    class ResultKey(Enum):
        pass

    def __init__(self, *, processor_name, source_processor_name, reset):
        super(MeanStdPointProcessor, self).__init__(processor_name=processor_name, reset=reset)
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
    def __init__(self, *, processor_name, atom_frame_field_name, ref_frame_field_name,
                 avg_atom_field_name, avg_ref_field_name, roi_slice, reset):
        super(AvgAtomRefImagePointProcessor, self).__init__(processor_name=processor_name, reset=reset)
        self.atom_frame_field_name = atom_frame_field_name
        self.ref_frame_field_name = ref_frame_field_name
        self.avg_atom_field_name = avg_atom_field_name
        self.avg_ref_field_name = avg_ref_field_name
        self.roi_slice = roi_slice

    def process_point(self, point, datamodel):
        data_dict = datamodel.data_dict
        point_key = f'point-{point:d}'
        avg_atom_frame = None
        avg_ref_frame = None
        shot_list = data_dict['shot_list'][point_key]
        num_loops = data_dict['loop_nums'][point_key]
        for shot_num in shot_list:
            atom_frame = datamodel.get_data(self.atom_frame_field_name, shot_num)[self.roi_slice].astype(float)
            ref_frame = datamodel.get_data(self.ref_frame_field_name, shot_num)[self.roi_slice].astype(float)
            if avg_atom_frame is None:
                avg_atom_frame = atom_frame
                avg_ref_frame = ref_frame
            else:
                avg_atom_frame += atom_frame
                avg_ref_frame += ref_frame
        avg_atom_frame = avg_atom_frame / num_loops
        avg_ref_frame = avg_ref_frame / num_loops

        avg_atom_field = DataDictField(datamodel=datamodel,
                                       field_name=self.avg_atom_field_name,
                                       data_source_name=self.processor_name,
                                       scale='point')
        avg_ref_field = DataDictField(datamodel=datamodel,
                                      field_name=self.avg_ref_field_name,
                                      data_source_name=self.processor_name,
                                      scale='point')

        avg_atom_field.set_data(point, avg_atom_frame)
        avg_ref_field.set_data(point, avg_ref_frame)


class RandomAtomRefImagePointProcessor(PointProcessor):
    def __init__(self, *, processor_name,
                 atom_frame_field_name, ref_frame_field_name,
                 rndm_atom_field_name, rndm_ref_field_name, rndm_shot_field_name,
                 roi_slice, reset):
        super(RandomAtomRefImagePointProcessor, self).__init__(processor_name=processor_name, reset=reset)
        self.atom_frame_field_name = atom_frame_field_name
        self.ref_frame_field_name = ref_frame_field_name
        self.rndm_atom_field_name = rndm_atom_field_name
        self.rndm_ref_field_name = rndm_ref_field_name
        self.rndm_shot_field_name = rndm_shot_field_name
        self.roi_slice = roi_slice

    def process_point(self, point, datamodel):
        data_dict = datamodel.data_dict
        point_key = f'point-{point:d}'
        shot_list = data_dict['shot_list'][point_key]
        random_shot = np.random.choice(shot_list)
        rndm_atom_frame = datamodel.get_data(self.atom_frame_field_name, random_shot)[self.roi_slice]
        rndm_ref_frame = datamodel.get_data(self.ref_frame_field_name, random_shot)[self.roi_slice]

        rndm_atom_field = DataDictField(datamodel=datamodel,
                                        field_name=self.rndm_atom_field_name,
                                        data_source_name=self.processor_name,
                                        scale='point')
        rndm_ref_field = DataDictField(datamodel=datamodel,
                                       field_name=self.rndm_ref_field_name,
                                       data_source_name=self.processor_name,
                                       scale='point')
        rndm_shot_field = DataDictField(datamodel=datamodel,
                                        field_name=self.rndm_shot_field_name,
                                        data_source_name=self.processor_name,
                                        scale='point')

        rndm_atom_field.set_data(point, rndm_atom_frame)
        rndm_ref_field.set_data(point, rndm_ref_frame)
        rndm_shot_field.set_data(point, random_shot)


class CountsThresholdPointProcessor(PointProcessor):
    class ResultKey(Enum):
        THRESHOLD = 'threshold'
        SHOTS_ABOVE = 'shots_above'
        NUM_ABOVE = 'num_above'
        FRAC_ABOVE = 'frac_above'

        SHOTS_BELOW = 'shots_below'
        NUM_BELOW = 'num_below'
        FRAC_BELOW = 'frac_below'

    def __init__(self, *, processor_name, counts_field_name, output_field_name, threshold, reset):
        super(CountsThresholdPointProcessor, self).__init__(processor_name=processor_name, reset=reset)
        self.counts_field_name = counts_field_name
        self.output_field_name = output_field_name
        self.threshold = threshold

    def process_point(self, point, datamodel):
        data_dict = datamodel.data_dict
        shot_list = data_dict['shot_list'][f'point-{point:d}']

        shots_above_list = []
        shots_below_list = []

        for shot_num in shot_list:
            counts = datamodel.get_data(self.counts_field_name, shot_num)
            if counts > self.threshold:
                shots_above_list.append(shot_num)
            elif counts <= self.threshold:
                shots_below_list.append(shot_num)

        num_total = len(shot_list)

        num_above = len(shots_above_list)
        frac_above = num_above / num_total

        num_below = len(shots_below_list)
        frac_below = num_below / num_total

        results_dict = dict()
        results_dict[self.ResultKey.THRESHOLD.value] = self.threshold
        results_dict[self.ResultKey.SHOTS_ABOVE.value] = shots_above_list
        results_dict[self.ResultKey.NUM_ABOVE.value] = num_above
        results_dict[self.ResultKey.FRAC_ABOVE.value] = frac_above
        results_dict[self.ResultKey.SHOTS_BELOW.value] = shots_below_list
        results_dict[self.ResultKey.NUM_BELOW.value] = num_below
        results_dict[self.ResultKey.FRAC_BELOW.value] = frac_below

        output_field = DataDictField(datamodel=datamodel,
                                     field_name=self.output_field_name,
                                     data_source_name=self.processor_name,
                                     scale='point')
        output_field.set_data(point, results_dict)

        return results_dict
