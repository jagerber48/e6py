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
