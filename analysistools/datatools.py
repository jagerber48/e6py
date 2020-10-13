import numpy as np
from pathlib import Path
import pickle


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


def get_num_files(data_path):
    file_list = list(data_path.glob('*.h5'))
    num_files = len(file_list)
    return num_files


def get_datastream_path(daily_path, run_name, datastream_name):
    datastream_path = Path(daily_path, 'data', run_name, datastream_name)
    analysis_path = Path(daily_path, 'analysis', run_name)
    analysis_path.mkdir(parents=True, exist_ok=True)
    return datastream_path
