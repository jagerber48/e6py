import numpy as np
import h5py


def get_image(file_path, image_key, roi_slice=None):
    if roi_slice is None:
        roi_slice = slice(None, None)
    h5_file = h5py.File(file_path, 'r')
    image = h5_file[image_key][roi_slice].astype(float)
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


def threshold_discrimination_analysis(analysis_dict, threshold):
    # TODO: Run on reference data
    counts_analysis_dict = analysis_dict['counts_analysis']
    counts_dict = counts_analysis_dict['counts']
    shot_list = analysis_dict['shot_list']
    loop_nums = analysis_dict['loop_nums']
    num_points = analysis_dict['num_points']
    
    td_dict = dict()
    analysis_dict['threshold_analysis'] = td_dict
    td_dict['threshold'] = threshold
    td_dict['loops_above'] = dict()
    td_dict['shots_above'] = dict()
    td_dict['num_above'] = dict()
    td_dict['fraction_above'] = dict()
    td_dict['loops_below'] = dict()
    td_dict['shots_below'] = dict()
    td_dict['num_below'] = dict()
    td_dict['fraction_below'] = dict()

    for point in range(num_points):
        point_key = f'point-{point:d}'
        point_shot_list = shot_list[point_key]
        num_loops = loop_nums[point_key]
        counts = np.array(counts_dict[point_key])

        loops_above = np.where(counts > threshold)[0]
        shots_above = point_shot_list[loops_above]
        num_above = len(loops_above)
        fraction_above = num_above / num_loops
        td_dict['loops_above'][point_key] = loops_above
        td_dict['shots_above'][point_key] = shots_above
        td_dict['num_above'][point_key] = num_above
        td_dict['fraction_above'][point_key] = fraction_above

        loops_below = np.where(counts <= threshold)[0]
        shots_below = point_shot_list[loops_below]
        num_below = len(loops_below)
        fraction_below = num_below / num_loops
        td_dict['loops_below'][point_key] = loops_below
        td_dict['shots_below'][point_key] = shots_below
        td_dict['num_below'][point_key] = num_below
        td_dict['fraction_below'][point_key] = fraction_below
    analysis_dict.save_dict()
