from functools import reduce
import numpy as np


def qprint(text, quiet=False):
    if not quiet:
        print(text)


def get_shot_list_from_point(point, num_points, num_shots, start_shot=0, stop_shot=None):
    # TODO: Implement different conventions for shot and point start indices
    shots = np.arange(point, num_shots, num_points)
    start_mask = start_shot <= shots
    shots = shots[start_mask]
    if stop_shot is not None:
        stop_mask = shots <= stop_shot
        shots = shots[stop_mask]
    num_loops = len(shots)
    return shots, num_loops


def shot_to_loop_and_point(shot, num_points=1, shot_index_convention=0,
                           loop_index_convention=0, point_index_convention=0):
    """
    Convert shot number to loop and point using the number of points. Default assumption is indexing for
    shot, loop, and point all starts from zero with options for other conventions.
    """
    shot_ind = shot - shot_index_convention
    loop_ind = shot_ind // num_points
    point_ind = shot_ind % num_points
    loop = loop_ind + loop_index_convention
    point = point_ind + point_index_convention
    return loop, point


def loop_and_point_to_shot(loop, point, num_points=1, shot_index_convention=0,
                           loop_index_convention=0, point_index_convention=0):
    """
    Convert loop and point to shot number using the number of points. Default assumption is indexing for
    shot, loop, and point all starts from zero with options for other conventions.
    """
    loop_ind = loop - loop_index_convention
    point_ind = point - point_index_convention
    shot_ind = loop_ind * num_points + point_ind
    shot = shot_ind + shot_index_convention
    return shot


def dataset_from_keychain(datamodel, keychain):
    data_dict = datamodel.data_dict
    data = reduce(lambda x, y: x[y], keychain.split('/'), data_dict)
    return data


def to_list(var):
    """
    Helper function to convert singleton input parameters into list-expecting parameters into singleton lists.
    """
    if isinstance(var, (list, tuple)):
        return var
    else:
        return [var]


class InputParamLogger:
    def __new__(cls, *args, **kwargs):
        input_param_dict = {'args': args, 'kwargs': kwargs, 'class': cls}
        obj = super(InputParamLogger, cls).__new__(cls)
        obj.input_param_dict = input_param_dict
        return obj

    @staticmethod
    def rebuild(input_param_dict):
        rebuild_class = input_param_dict['class']
        args = input_param_dict['args']
        kwargs = input_param_dict['kwargs']
        new_obj = rebuild_class(*args, **kwargs)
        return new_obj