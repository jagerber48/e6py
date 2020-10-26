from enum import Enum
from ..datatools import InputParamLogger, qprint


class ProcessorScale(Enum):
    SHOT = 'shot'
    POINT = 'point'
    RUN = 'run'


class Processor(InputParamLogger):
    class ResultKey(Enum):
        pass

    def __init__(self, *, processor_name, scale):
        self.processor_name = processor_name
        self.scale = scale

    # def create_processor_dict(self, data_dict):
    #     processor_dict = dict()
    #     data_dict[f'{self.scale.value}_processors'][self.processor_name] = processor_dict
    #     return processor_dict

    # def load_processor_dict(self, data_dict):
    #     if self.processor_name in data_dict[f'{self.scale.value}_processors']:
    #         processor_dict = data_dict[f'{self.scale.value}_processors'][self.processor_name]
    #         old_input_param_dict = processor_dict['input_param_dict']
    #         if self.input_param_dict != old_input_param_dict:
    #             processor_dict = self.create_processor_dict(data_dict)
    #     else:
    #         processor_dict = self.create_processor_dict(data_dict)
    #     return processor_dict

    def process(self, datamodel, quiet=False):
        qprint(f'**Running {self.scale.value}_processor: {self.processor_name}**', quiet=quiet)
        # data_dict = datamodel.data_dict
        # processor_dict = self.load_processor_dict(data_dict)
        self.scaled_process(datamodel, quiet=quiet)

    def scaled_process(self, datamodel, quiet=False):
        """
        Subclasses will implement generic processing procedures depending on the scale of the Processor. For example
        a ShotProcessor will loop through all shots while a PointProcessor will loop through all points.
        """
        raise NotImplementedError
