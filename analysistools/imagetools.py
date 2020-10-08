import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from . import datatools
from .datatools import AnalysisDict, shot_to_loop_and_point


def get_image(file_path, image_key, roi_slice=None):
    if roi_slice is None:
        roi_slice = slice(None, None)
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


def display_images(daily_path, run_name, imaging_system_name, file_prefix='jkam_capture', conversion_gain=1,
                   roi_slice=None, num_points=1, start_shot=0, stop_shot=None, transpose=False):
    datastream_path = datatools.get_datastream_path(daily_path, run_name, imaging_system_name)
    num_shots = datatools.get_num_files(datastream_path)
    final_shot = stop_shot
    if final_shot is None:
        final_shot = num_shots - 1

    analysis_dict = AnalysisDict(daily_path, run_name)
    analysis_dict.set_shot_lists(num_shots, num_points=num_points, start_shot=start_shot, stop_shot=stop_shot)
    analysis_path = analysis_dict.analysis_path

    shot_list_by_point = analysis_dict['shot_list']
    loop_num_by_point = analysis_dict['loop_nums']

    display_images_analysis_dict = dict()
    display_images_analysis_dict['conversion_gain'] = conversion_gain
    show_shot_dict = dict()
    atom_frame_avg_dict = dict()
    ref_frame_avg_dict = dict()
    display_images_analysis_dict['show_shot_dict'] = show_shot_dict
    display_images_analysis_dict['atom_frame_avg_dict'] = atom_frame_avg_dict
    display_images_analysis_dict['ref_frame_avg_dict'] = ref_frame_avg_dict
    fig_dict = dict()

    for point in range(num_points):
        point_key = f'point-{point:d}'
        show_shot = np.random.choice(shot_list_by_point[point_key])
        show_shot_dict[point_key] = show_shot

        atom_frame_avg_dict[point_key] = None
        ref_frame_avg_dict[point_key] = None

        fig_dict[point_key] = plt.figure(figsize=(12, 12))
        fig_dict[point_key].suptitle(f'{run_name} - Point {point} - {loop_num_by_point[point_key]} Loops')

    if transpose and roi_slice is not None:
        roi_slice = tuple(reversed(roi_slice))

    cmap = 'binary_r'

    for shot_num in range(start_shot, final_shot + 1):
        loop, point = shot_to_loop_and_point(shot_num, num_points=num_points)
        point_key = f'point-{point:d}'
        file_name = f'{file_prefix}_{shot_num:05d}.h5'
        file_path = Path(datastream_path, file_name)

        atom_frame = get_image(file_path, 'atom_frame', roi_slice=roi_slice)
        atom_frame = atom_frame / conversion_gain
        ref_frame = get_image(file_path, 'reference_frame', roi_slice=roi_slice)
        ref_frame = ref_frame / conversion_gain

        if atom_frame_avg_dict[point_key] is None:
            atom_frame_avg_dict[point_key] = atom_frame
            ref_frame_avg_dict[point_key] = ref_frame
        else:
            atom_frame_avg_dict[point_key] += atom_frame
            ref_frame_avg_dict[point_key] += ref_frame

        if shot_num == show_shot_dict[point_key]:
            min_val = np.min(atom_frame)
            max_val = np.max(atom_frame)

            fig = fig_dict[point_key]
            ax_atom_single = fig.add_subplot(2, 2, 1)
            im = ax_atom_single.imshow(atom_frame, vmin=min_val, vmax=max_val, cmap=cmap)
            fig.colorbar(im, ax=ax_atom_single)
            ax_atom_single.set_title(f'{run_name}:  Single Atom Shot - Point {point} - Shot #{shot_num}')

            ax_ref_single = fig.add_subplot(2, 2, 2)
            im = ax_ref_single.imshow(ref_frame, vmin=min_val, vmax=max_val, cmap=cmap)
            fig.colorbar(im, ax=ax_ref_single)
            ax_ref_single.set_title(f'{run_name}: Single Ref Shot - Point {point} -  Shot #{shot_num}')

    for point in range(num_points):
        point_key = f'point-{point:d}'
        fig = fig_dict[point_key]

        atom_frame_avg = atom_frame_avg_dict[point_key] / loop_num_by_point[point_key]
        ref_frame_avg = ref_frame_avg_dict[point_key] / loop_num_by_point[point_key]
        atom_frame_avg_dict[point_key] = atom_frame_avg
        ref_frame_avg_dict[point_key] = ref_frame_avg

        min_val = np.min(atom_frame_avg)
        max_val = np.max(atom_frame_avg)

        ax_atom_avg = fig.add_subplot(2, 2, 3)
        im = ax_atom_avg.imshow(atom_frame_avg, vmin=min_val, vmax=max_val, cmap=cmap)
        fig.colorbar(im, ax=ax_atom_avg)
        ax_atom_avg.set_title(f'{run_name}: Avg Atom Shot - Point {point} - {loop_num_by_point[point_key]} Loops')

        ax_ref_avg = fig.add_subplot(2, 2, 4)
        im = ax_ref_avg.imshow(ref_frame_avg, vmin=min_val, vmax=max_val, cmap=cmap)
        fig.colorbar(im, ax=ax_ref_avg)
        ax_ref_avg.set_title(f'{run_name}: Avg Ref Shot - Point {point} - {loop_num_by_point[point_key]} Loops')

        fig.savefig(Path(analysis_path, f'images - Point {point:d}.png'))
    plt.show()

    analysis_dict['display_images_analysis'] = display_images_analysis_dict
    analysis_dict.save_dict()


def count_analysis(daily_path, run_name, imaging_system_name, file_prefix='jkam_capture', conversion_gain=1,
                   roi_slice=None, num_points=1, start_shot=0, stop_shot=None, transpose=False,
                   **hist_kwargs):
    datastream_path = datatools.get_datastream_path(daily_path, run_name, imaging_system_name)
    num_shots = datatools.get_num_files(datastream_path)
    final_shot = stop_shot
    if final_shot is None:
        final_shot = num_shots - 1

    analysis_dict = AnalysisDict(daily_path, run_name)
    analysis_dict.set_shot_lists(num_shots, num_points=num_points, start_shot=start_shot, stop_shot=stop_shot)
    analysis_path = analysis_dict.analysis_path

    loop_num_by_point = analysis_dict['loop_nums']

    counts_analysis_dict = dict()
    counts_dict = dict()
    ref_counts_dict = dict()
    counts_analysis_dict['counts'] = counts_dict
    counts_analysis_dict['ref_counts'] = ref_counts_dict

    fig_dict = dict()

    for point in range(num_points):
        point_key = f'point-{point:d}'
        counts_dict[point_key] = []
        ref_counts_dict[point_key] = []
        fig_dict[point_key] = plt.figure(figsize=(12, 12))

    if transpose and roi_slice is not None:
        roi_slice = tuple(reversed(roi_slice))

    for shot_num in range(start_shot, final_shot + 1):
        loop, point = shot_to_loop_and_point(shot_num, num_points=num_points)
        point_key = f'point-{point:d}'
        file_name = f'{file_prefix}_{shot_num:05d}.h5'
        file_path = Path(datastream_path, file_name)

        atom_frame = get_image(file_path, 'atom_frame', roi_slice=roi_slice)
        atom_frame = atom_frame / conversion_gain
        ref_frame = get_image(file_path, 'reference_frame', roi_slice=roi_slice)
        ref_frame = ref_frame / conversion_gain

        counts = np.nansum(atom_frame)
        counts_dict[point_key].append(counts)

        ref_counts = np.nansum(ref_frame)
        ref_counts_dict[point_key].append(ref_counts)

    for point in range(num_points):
        point_key = f'point-{point:d}'
        fig = fig_dict[point_key]
        ax_plot = fig.add_subplot(2, 1, 1)
        ax_plot.plot(counts_dict[point_key], '.', markersize=10)
        ax_plot.plot(ref_counts_dict[point_key], '.', markersize=10)
        ax_hist = fig.add_subplot(2, 1, 2)
        ax_hist.hist(counts_dict[point_key], alpha=0.5, **hist_kwargs)
        ax_hist.hist(ref_counts_dict[point_key], alpha=0.5, **hist_kwargs)
        fig.suptitle(f'{run_name} - Point {point} - {loop_num_by_point[point_key]} Loops', fontsize=16)
        fig.savefig(Path(analysis_path, f'counts - Point {point}.png'))
    plt.show()

    analysis_dict['counts_analysis'] = counts_analysis_dict
    analysis_dict.save_dict()
