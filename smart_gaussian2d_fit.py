import numpy as np
import scipy.ndimage
import scipy.stats
from scipy.optimize import least_squares
from scipy.special import erf
import time
import matplotlib.pyplot as plt


def gaussian_2d(x, y, x0=0, y0=0, sx=1, sy=1, A=1, offset=0, theta=0, x_slope=0, y_slope=0):
    rx = np.cos(theta) * (x-x0) - np.sin(theta) * (y-y0)
    ry = np.sin(theta) * (x-x0) + np.cos(theta) * (y-y0)
    return A * np.exp(-(1/2) * ((rx/sx)**2 + ((ry/sy)**2))) + offset + x_slope * (x-x0) + y_slope * (y-y0)


def img_moments(img):
    rvec = np.indices(img.shape)
    tot = img.sum()
    if tot <= 0:
        raise ValueError('Integrated image intensity is negative, image may be too noisy. '
                         'Image statistics cannot be calculated.')
    x0 = np.sum(img*rvec[0])/tot
    y0 = np.sum(img*rvec[1])/tot
    varx = np.sum(img * (rvec[0] - x0)**2) / tot
    vary = np.sum(img * (rvec[1] - y0)**2) / tot
    if varx <= 0 or vary <= 0:
        raise ValueError('varx or vary is negative, image may be too noisy. '
                         'Image statistics cannot be calculated.')
    sx = np.sqrt(varx)
    sy = np.sqrt(vary)
    return x0, y0, sx, sy


# noinspection PyPep8Naming
def get_guess_values(img, quiet):
    # Get fit guess values
    x_range = img.shape[0]
    y_range = img.shape[1]
    A_guess = img.max()-img.min()
    B_guess = img.min()
    try:
        x0_guess, y0_guess, sx_guess, sy_guess = img_moments(img)
    except ValueError as e:
        if not quiet:
            print(e)
            print('Using default guess values.')
        x0_guess, y0_guess, sx_guess, sy_guess = [x_range/2, y_range/2, x_range/2, y_range/2]
    p_guess = np.array([x0_guess, y0_guess, sx_guess, sy_guess, A_guess, B_guess])
    if not quiet:
        print(f'x0_guess = {x0_guess:.1f}')
        print(f'y0_guess = {y0_guess:.1f}')
        print(f'sx_guess = {sx_guess:.1f}')
        print(f'sy_guess = {sy_guess:.1f}')
    return p_guess


def get_dict_param_keys():
    dict_param_keys = ['x0', 'y0', 'sx', 'sy', 'A', 'offset', 'theta']
    return dict_param_keys


def make_param_dict(name, val, std, conf_level=erf(1 / np.sqrt(2)), dof=None):
    pdict = {'name': name, 'val': val, 'std': std, 'conf_level': conf_level}
    if dof is None:  # Assume normal distribution if dof not specified
        tcrit = scipy.stats.norm.ppf((1 + conf_level) / 2)
    else:
        tcrit = scipy.stats.t.ppf((1 + conf_level) / 2, dof)
    pdict['err_half_range'] = tcrit * std
    pdict['err_full_range'] = 2 * pdict['err_half_range']
    pdict['val_lb'] = val - pdict['err_half_range']
    pdict['val_ub'] = val + pdict['err_half_range']
    return pdict


def create_fit_struct(img, popt, pcov, conf_level, dof, param_keys):
    coords_arrays = np.indices(img.shape)
    model_img = gaussian_2d(coords_arrays[0], coords_arrays[1], *popt)
    dict_param_keys = param_keys
    fit_struct = dict()
    for i in range(popt.size):
        if dict_param_keys[i] in {'sx', 'sy'}:
            popt[i] = np.abs(popt[i])
        key = dict_param_keys[i]
        pdict = make_param_dict(key, popt[i], np.sqrt(pcov[i, i]), conf_level, dof)
        fit_struct[key] = pdict
    fit_struct['cov'] = pcov
    fit_struct['data_img'] = img
    fit_struct['model_img'] = model_img
    fit_struct['NGauss'] = fit_struct['A']['val'] * 2 * np.pi * fit_struct['sx']['val'] * fit_struct['sy']['val']
    fit_struct['NSum'] = np.sum(img)
    # TODO: NSum_BGsubtract should subtract linear background as well if it was fitted for
    fit_struct['NSum_BGsubtract'] = np.sum(img - fit_struct['offset']['val'])
    return fit_struct


def make_visualization_figure(fit_struct, param_keys, show_plot=True, save_name=None):
    # TODO: Catch error if center of fit is outside plot range
    img = fit_struct['data_img']
    model_img = fit_struct['model_img']
    x_range = img.shape[0]
    y_range = img.shape[1]
    x0 = int(round(fit_struct['x0']['val']))
    y0 = int(round(fit_struct['y0']['val']))
    sx = fit_struct['sx']['val']
    sy = fit_struct['sy']['val']
    img_min = np.min([img.min(), model_img.min()])
    img_max = np.max([img.max(), model_img.max()])

    # Plotting
    fig = plt.figure(figsize=(8, 8))

    # Data 2D Plot
    ax_data = fig.add_subplot(2, 2, 1, position=[0.1, 0.5, 0.25, 0.35])
    ax_data.imshow(img, vmin=img_min, vmax=img_max, cmap='binary')
    ax_data.set_aspect(x_range / y_range)
    ax_data.xaxis.tick_top()
    ax_data.set_xlabel('Horizontal Position')
    ax_data.xaxis.set_label_position('top')
    ax_data.set_ylabel('Vertical Position')
    ax_data.set_ylim(0, x_range)
    ax_data.set_xlim(0, y_range)

    # Fit 2D Plot
    ax_fit = fig.add_subplot(2, 2, 4, position=[0.4, 0.1, 0.25, 0.35])
    ax_fit.imshow(model_img, vmin=img_min, vmax=img_max, cmap='binary')
    ax_fit.set_aspect(x_range / y_range)
    ax_fit.yaxis.tick_right()
    ax_fit.set_xlabel('Horizontal Position')
    ax_fit.set_ylabel('Vertical Position')
    ax_fit.yaxis.set_label_position('right')
    ax_fit.set_ylim(0, x_range)
    ax_fit.set_xlim(0, y_range)

    # X Linecut Plot
    ax_x_line = fig.add_subplot(2, 2, 2, position=[0.4, 0.5, 0.25, 0.35])
    x_int_cut_dat = np.sum(img, axis=1) / np.sqrt(2 * np.pi * sy**2)
    x_int_cut_model = np.sum(model_img, axis=1) / np.sqrt(2 * np.pi * sy**2)
    ax_x_line.plot(x_int_cut_dat, range(x_range), 'o', zorder=1)
    ax_x_line.plot(x_int_cut_model, range(x_range), zorder=2)
    ax_x_line.invert_yaxis()
    ax_x_line.yaxis.tick_right()
    ax_x_line.xaxis.tick_top()
    ax_x_line.set_xlabel('Integrated Intensity')
    ax_x_line.xaxis.set_label_position('top')
    try:
        # x_line_cut_dat = img[:, y0]
        ax_data.axvline(y0, linestyle='--')
        ax_fit.axvline(y0, linestyle='--')
        # ax_x_line.plot(x_line_cut_dat, range(x_range), 'o', zorder=0)
    except IndexError as e:
        print(e)

    # Y Linecut Plot
    ax_y_line = fig.add_subplot(2, 2, 3, position=[0.1, 0.1, 0.25, 0.35])
    y_int_cut_dat = np.sum(img, axis=0) / np.sqrt(2 * np.pi * sx**2)
    y_int_cut_model = np.sum(model_img, axis=0) / np.sqrt(2 * np.pi * sx**2)
    ax_y_line.plot(range(y_range), y_int_cut_dat, 'o', zorder=1)
    ax_y_line.plot(range(y_range), y_int_cut_model, zorder=2)
    ax_y_line.invert_yaxis()
    ax_y_line.set_ylabel('Integrated Intensity')

    try:
        # y_line_cut_dat = img[x0, :]
        ax_data.axhline(x0, linestyle='--')
        ax_fit.axhline(x0, linestyle='--')
        # ax_y_line.plot(range(y_range), y_line_cut_dat, 'o', zorder=0)
    except IndexError as e:
        print(e)

    # Write parameter values
    # popt = fit_struct['popt']
    dict_param_keys = param_keys
    print_str = ''
    for key in dict_param_keys:

        param = fit_struct[key]
        print_str += f"{key} = {param['val']:.1f} +- {param['err_half_range']:.3f}\n"
    fig.text(.8, .5, print_str)

    fit_struct['fit_fig'] = fig

    if save_name is not None:
        plt.savefig(save_name)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return


# noinspection PyTypeChecker
def fit_gaussian2d(img, zoom=1.0, theta_offset=0, fit_lin_slope=True, quiet=True, show_plot=True, save_name=None,
                   conf_level=erf(1 / np.sqrt(2))):
    """
    :param img: Image to fit
    :param zoom: Decimate rate to speed up fitting if downsample is selected
    :param theta_offset: Central value about which theta is expected to scatter. Allowed values of theta will be
    theta_offset +- 45 deg. Fits with theta near the edge of this range may swap sx and sy for similar images
    :param fit_lin_slope: Flag to indicate if a fit should be done for a linear background
    :param quiet: Squelch variable
    :param show_plot: Whether to show the plot or not
    :param save_name: File name for saved figure, None means don't save
    :param conf_level: Confidence level for confidence region
    :return: Returns a struct containing relevant data output of the fit routine
    Take an image as input and fits the image amplitude with a 2D gaussian.
    Attempts to guess fit values by extracting 2D mean and variance of the image. this only
    makes sense if the image intensity is mostly positive.
    """

    p_guess = get_guess_values(img, quiet)
    p_guess = np.append(p_guess, 0)  # Append extra parameter for theta
    if fit_lin_slope:
        p_guess = np.append(p_guess, [0, 0])  # Append two extra parameters for x_slope, y_slope
    # Downsample image to speed up fit
    img_downsampled = scipy.ndimage.interpolation.zoom(img, 1 / zoom)
    if not quiet:
        print(f'Image downsampled by factor: {zoom:.1f}')

    coords_arrays = np.indices(img_downsampled.shape)  # (2, x_range, y_range) array of coordinate labels
    # Perform fit
    # p_bounds = ([-np.inf, -np.inf, 0, 0, -np.inf, -np.inf, 0],
    #             [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 360])

    if fit_lin_slope:
        def img_cost_func(x):
            return np.ravel(gaussian_2d(coords_arrays[0] * zoom, coords_arrays[1] * zoom, *x) - img_downsampled)

        param_keys = ['x0', 'y0', 'sx', 'sy', 'A', 'offset', 'theta', 'x_slope', 'y_slope']
    else:
        def img_cost_func(x):
            return np.ravel(gaussian_2d(coords_arrays[0] * zoom, coords_arrays[1] * zoom,
                                        x_slope=0, y_slope=0, *x)
                            - img_downsampled)
        param_keys = ['x0', 'y0', 'sx', 'sy', 'A', 'offset', 'theta']

    t_fit_start = time.time()

    # noinspection PyTypeChecker
    lsq_struct: dict = least_squares(img_cost_func, p_guess, verbose=0)

    t_fit_stop = time.time()
    if not quiet:
        print(f'fit time = {t_fit_stop - t_fit_start:.2f} s')

    popt = lsq_struct['x']
    jac = lsq_struct['jac']
    cost = lsq_struct['cost']

    theta = popt[6]
    if theta >= np.pi / 2 or theta < 0:
        theta = theta % (2 * np.pi)
    if 0 + theta_offset <= theta < np.pi / 4 + theta_offset:
        pass
    elif np.pi / 4 + theta_offset <= theta < 3 * np.pi / 4 + theta_offset:
        theta = theta - np.pi / 2
        popt[2], popt[3] = popt[3], popt[2]
        jac[:, [2, 3]] = jac[:, [3, 2]]
    elif 3 * np.pi / 4 + theta_offset <= theta < 5 * np.pi / 4 + theta_offset:
        theta = theta - np.pi
    elif 5 * np.pi / 4 + theta_offset <= theta < 7 * np.pi / 4 + theta_offset:
        theta = theta - 3 * np.pi / 2
        popt[2], popt[3] = popt[3], popt[2]
        jac[:, [2, 3]] = jac[:, [3, 2]]
    elif 7 * np.pi / 4 + theta_offset <= theta < 2 * np.pi + theta_offset:
        theta = theta - 2 * np.pi
    popt[6] = theta * 180 / np.pi
    jac[6, :] = jac[6, :] * 180 / np.pi
    jac[:, 6] = jac[:, 6] * 180 / np.pi
    jac[6, 6] = jac[6, 6] * np.pi / 180

    n = img_downsampled.shape[0]*img_downsampled.shape[1]  # Number of data points
    p = popt.size  # Number of fit parameters
    dof = n - p

    s2 = 2 * cost / dof
    try:
        cov = s2 * np.linalg.inv(np.matmul(jac.T, jac))
    except np.linalg.LinAlgError as e:
        print(e)
        cov = 0 * jac
    fit_struct = create_fit_struct(img, popt, cov, conf_level, dof, param_keys)
    if show_plot or (save_name is not None):
        make_visualization_figure(fit_struct, param_keys, show_plot, save_name)

    return fit_struct
