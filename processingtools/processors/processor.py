from enum import Enum
from ..datamodel import InputParamLogger, qprint


class ProcessorScale(Enum):
    SHOT = 'shot'
    POINT = 'point'
    RUN = 'run'


class Processor(InputParamLogger):
    class ResultKey(Enum):
        pass

    def __init__(self, *, processor_name, data_target_name, scale):
        self.processor_name = processor_name
        self.data_target_name = data_target_name
        self.scale = scale

    def create_target_dict(self, data_dict):
        target_dict = dict()
        target_dict['processor_param_dict'] = self.input_param_dict
        target_dict['results'] = dict()
        data_dict[f'processed_{self.scale.value}_data'][self.data_target_name] = target_dict
        return target_dict

    def load_target_dict(self, data_dict):
        if self.data_target_name in data_dict[f'processed_{self.scale.value}_data']:
            target_dict = data_dict[f'processed_{self.scale.value}_data'][self.data_target_name]
            old_processor_param_dict = target_dict['processor_param_dict']
            if self.input_param_dict != old_processor_param_dict:
                target_dict = self.create_target_dict(data_dict)
        else:
            target_dict = self.create_target_dict(data_dict)
        return target_dict

    def process(self, datamodel, quiet=False):
        qprint(f'**Running {self.scale.value}_processor: {self.processor_name}**', quiet=quiet)
        data_dict = datamodel.data_dict
        target_dict = self.load_target_dict(data_dict)
        self.scaled_process(datamodel, target_dict, quiet=quiet)

    def scaled_process(self, datamodel, target_dict, quiet=False):
        """
        Subclasses will implement generic processing procedures depending on the scale of the Processor. For example
        a ShotProcessor will loop through all shots while a PointProcessor will loop through all points.
        """
        raise NotImplementedError
