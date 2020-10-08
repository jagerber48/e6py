import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from . import datatools
from .datatools import AnalysisDict


def get_image(file_path, image_key, roi_slice=slice(None, None)):
    h5_file = h5py.File(file_path, 'r')
    image = h5_file[image_key][roi_slice].astype(float)
    # if roi_slice is not None:
    #     image = image[roi_slice]
    return image


def roi_from_center_pixel(center_pixel, pixel_half_ranges):
    x_center = center_pixel[1]
    x_half_range = pixel_half_ranges[1]
    x_lower = x_center - x_half_range
    x_upper = x_center + x_half_range + 1
    x_slice = slice(x_lower, x_upper, 1)

    y_center = center_pixel[0]
    y_half_range = pixel_half_ranges[0]
    y_lower = y_center - y_half_range
    y_upper = y_center + y_half_range + 1
    y_slice = slice(y_lower, y_upper, 1)

    return tuple((y_slice, x_slice))


def display_images(daily_path, run_name, imaging_system_name,
                   roi_slice=None, num_points=1, start_shot=0, transpose=False):
    datastream_path = datatools.get_datastream_path(daily_path, run_name, imaging_system_name)
    num_shots = datatools.get_num_files(datastream_path)

    analysis_dict = AnalysisDict(daily_path, run_name)
    analysis_dict.set_shot_lists(num_points, num_shots)

    point_shots_dict = dict()
    point_loop_num_dict = dict()
    show_shot_dict = dict()
    atom_frame_avg_dict = dict()
    ref_frame_avg_dict = dict()
    fig_dict = dict()

    for point in range(num_points):
        point_shots, point_loops = datatools.get_shot_list_from_point(point, num_points, num_shots)
        point_shots_dict[point] = point_shots
        point_loop_num_dict[point] = point_loops

        show_shot = np.random.choice(point_shots)
        show_shot_dict[point] = show_shot

        atom_frame_avg_dict[point] = None
        ref_frame_avg_dict[point] = None

        fig_dict[point] = plt.figure(figsize=(12, 12))
        fig_dict[point].suptitle(f'{run_name} - Point {point} - {point_loop_num_dict[point]} Loops')

    if transpose and roi_slice is not None:
        roi_slice = tuple(reversed(roi_slice))

    cmap = 'binary_r'

    for shot_num in range(start_shot, num_shots):
        loop, point = e6utils.shot_to_loop_and_point(shot_num, num_points=num_points)
        file_name = f'{file_prefix}_{shot_num:05d}{file_suffix}'
        file_path = Path(data_path, file_name)

        atom_frame = get_image(file_path, 'atom_frame', roi_slice=roi_slice)
        atom_frame = atom_frame / conversion_gain
        ref_frame = get_image(file_path, 'reference_frame', roi_slice=roi_slice)
        ref_frame = ref_frame / conversion_gain

        if atom_frame_avg_dict[point] is None:
            atom_frame_avg_dict[point] = atom_frame
            ref_frame_avg_dict[point] = ref_frame
        else:
            atom_frame_avg_dict[point] += atom_frame
            ref_frame_avg_dict[point] += ref_frame

        if shot_num == show_shot_dict[point]:
            min_val = np.min(atom_frame)
            max_val = np.max(atom_frame)

            fig = fig_dict[point]
            ax_atom_single = fig.add_subplot(2, 2, 1)
            im = ax_atom_single.imshow(atom_frame, vmin=min_val, vmax=max_val, cmap=cmap)
            plt.colorbar(im, ax=ax_atom_single)
            ax_atom_single.set_title(f'{run_name}:  Single Atom Shot - Point {point} - Shot #{shot_num}')

            ax_ref_single = fig.add_subplot(2, 2, 2)
            im = ax_ref_single.imshow(ref_frame, vmin=min_val, vmax=max_val, cmap=cmap)
            plt.colorbar(im, ax=ax_ref_single)
            ax_ref_single.set_title(f'{run_name}: Single Ref Shot - Point {point} -  Shot #{shot_num}')

    for point in range(num_points):
        fig = fig_dict[point]

        atom_frame_avg = atom_frame_avg_dict[point] / point_loop_num_dict[point]
        ref_frame_avg = ref_frame_avg_dict[point] / point_loop_num_dict[point]

        min_val = np.min(atom_frame_avg)
        max_val = np.max(atom_frame_avg)

        ax_atom_avg = fig.add_subplot(2, 2, 3)
        im = ax_atom_avg.imshow(atom_frame_avg, vmin=min_val, vmax=max_val, cmap=cmap)
        plt.colorbar(im, ax=ax_atom_avg)
        ax_atom_avg.set_title(f'{run_name}: Avg Atom Shot - Point {point} - {point_loop_num_dict[point]} Loops')

        ax_ref_avg = fig.add_subplot(2, 2, 4)
        im = ax_ref_avg.imshow(ref_frame_avg, vmin=min_val, vmax=max_val, cmap=cmap)
        plt.colorbar(im, ax=ax_ref_avg)
        ax_ref_avg.set_title(f'{run_name}: Avg Ref Shot - Point {point} - {point_loop_num_dict[point]} Loops')

        fig.savefig(Path(analysis_path, f'images - Point {point:d}.png'))

    plt.show()