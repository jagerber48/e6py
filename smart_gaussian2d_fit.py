import numpy as np
import scipy.ndimage
import scipy.stats
from scipy.optimize import least_squares
from scipy.special import erf
import time
import matplotlib.pyplot as plt


def gaussian_2d(x, y, x0=0, y0=0, sx=1, sy=1, A=1, offset=0, theta=0):
    rx = np.cos(np.radians(theta))*(x-x0) - np.sin(np.radians(theta))*(y-y0)
    ry = np.sin(np.radians(theta))*(x-x0) + np.cos(np.radians(theta))*(y-y0)
    return A * np.exp(-(1/2)*((rx/sx)**2 + ((ry/sy)**2))) + offset


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


def get_guess_values(img, quiet):
    # Get fit guess values
    x_range = img.shape[0]
    y_range = img.shape[1]
    A_guess = img.max()-img.min()
    B_guess = img.min()
    try:
        x0_guess, y0_guess, sx_guess, sy_guess = img_moments(img)
    except ValueError as e:
        print(e)
        print('Using default guess values.')
        x0_guess, y0_guess, sx_guess, sy_guess = [x_range/2, y_range/2, x_range/2, y_range/2]
    p_guess = np.array([x0_guess, y0_guess, sx_guess, sy_guess, A_guess, B_guess, 0])
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
    if dof is None:  # Assume normal distribution is dof not specified
        tcrit = scipy.stats.norm.ppf((1 + conf_level) / 2)
    else:
        tcrit = scipy.stats.t.ppf((1 + conf_level) / 2, dof)
    pdict['err_half_range'] = tcrit * std
    pdict['err_full_range'] = 2 * pdict['err_half_range']
    pdict['val_lb'] = val - pdict['err_half_range']
    pdict['val_ub'] = val + pdict['err_half_range']
    return pdict


def create_fit_struct(img, popt, pcov, conf_level, dof):
    coords_arrays = np.indices(img.shape)
    model_img = gaussian_2d(coords_arrays[0], coords_arrays[1], *popt)
    dict_param_keys = get_dict_param_keys()
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
    fit_struct['NSum_BGsubtract'] = np.sum(img - fit_struct['offset']['val'])
    return fit_struct


def make_visualization_figure(fit_struct, show_plot=True, save_name=None):
    img = fit_struct['data_img']
    model_img = fit_struct['model_img']
    x_range = img.shape[0]
    y_range = img.shape[1]
    x0 = int(round(fit_struct['x0']['val']))
    y0 = int(round(fit_struct['y0']['val']))
    img_min = np.min([img.min(), model_img.min()])
    img_max = np.max([img.max(), model_img.max()])

    x_line_cut_dat = img[:, y0]
    x_line_cut_model = model_img[:, y0]
    y_line_cut_dat = img[x0, :]
    y_line_cut_model = model_img[x0, :]

    x_int_cut_dat = np.sum(img, axis=1)/y_range
    x_int_cut_model = np.sum(model_img, axis=1)/y_range
    y_int_cut_dat = np.sum(img, axis=0)/x_range
    y_int_cut_model = np.sum(model_img, axis=0)/x_range

    # Plotting
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(2, 2, 1, position=[0.1, 0.5, 0.25, 0.35])
    ax.imshow(img, vmin=img_min, vmax=img_max)
    ax.axvline(y0, linestyle='--')
    ax.axhline(x0, linestyle='--')
    ax.set_aspect(x_range / y_range)
    ax.xaxis.tick_top()
    ax.set_xlabel('Horizontal Position')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Vertical Position')

    ax = fig.add_subplot(2, 2, 4, position=[0.4, 0.1, 0.25, 0.35])
    ax.imshow(model_img, vmin=img_min, vmax=img_max)
    ax.axvline(y0, linestyle='--')
    ax.axhline(x0, linestyle='--')
    ax.set_aspect(x_range / y_range)
    ax.yaxis.tick_right()
    ax.set_xlabel('Horizontal Position')
    ax.set_ylabel('Vertical Position')
    ax.yaxis.set_label_position('right')

    ax = fig.add_subplot(2, 2, 2, position=[0.4, 0.5, 0.25, 0.35])
    ax.plot(x_int_cut_dat, range(x_range))
    ax.plot(x_int_cut_model, range(x_range))
    ax.plot(x_line_cut_dat, range(x_range))
    ax.plot(x_line_cut_model, range(x_range))
    ax.invert_yaxis()
    ax.yaxis.tick_right()
    ax.xaxis.tick_top()
    ax.set_xlabel('Integrated Intensity')
    ax.xaxis.set_label_position('top')

    ax = fig.add_subplot(2, 2, 3, position=[0.1, 0.1, 0.25, 0.35])
    ax.plot(range(y_range), y_int_cut_dat)
    ax.plot(range(y_range), y_int_cut_model)
    ax.plot(range(y_range), y_line_cut_dat)
    ax.plot(range(y_range), y_line_cut_model)
    ax.invert_yaxis()
    ax.set_ylabel('Integrated Intensity')

    # Write parameter values
    # popt = fit_struct['popt']
    dict_param_keys = get_dict_param_keys()
    print_str = ''
    for i in range(7):
        key = dict_param_keys[i]
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


def fit_gaussian2d(img, zoom=1.0, quiet=True, show_plot=True, save_name=None,
                   conf_level=erf(1 / np.sqrt(2))):
    """
    :param img: Image to fit
    :param zoom: Decimate rate to speed up fitting if downsample is selected
    :param zoom: Boolean indicating whether or not to downsample
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
    # Downsample image to speed up fit
    img_downsampled = scipy.ndimage.interpolation.zoom(img, 1 / zoom)
    if not quiet:
        print(f'Image downsampled by factor: {zoom:.1f}')

    coords_arrays = np.indices(img_downsampled.shape)  # (2, x_range, y_range) array of coordinate labels
    # Perform fit
    # p_bounds = ([-np.inf, -np.inf, 0, 0, -np.inf, -np.inf, 0],
    #             [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 360])

    def img_cost_func(x):
        return np.ravel(gaussian_2d(coords_arrays[0] * zoom, coords_arrays[1] * zoom, *x) - img_downsampled)
    t_fit_start = time.time()
    lsq_struct = least_squares(img_cost_func, p_guess, verbose=0)
    t_fit_stop = time.time()
    if not quiet:
        print(f'fit time = {t_fit_stop - t_fit_start:.2f} s')

    popt = lsq_struct['x']
    jac = lsq_struct['jac']

    n = img_downsampled.shape[0]*img_downsampled.shape[1]  # Number of data points
    p = popt.size  # Number of fit parameters
    dof = n - p
    s2 = 2*lsq_struct['cost']/dof
    cov = s2 * np.linalg.inv(np.matmul(jac.T, jac))

    fit_struct = create_fit_struct(img, popt, cov, conf_level, dof)
    if show_plot or (save_name is not None):
        make_visualization_figure(fit_struct, show_plot, save_name)

    return fit_struct
