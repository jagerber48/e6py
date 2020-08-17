import numpy as np
import scipy.ndimage
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt
# from .E6utils import gaussian_2d_s as gaussian_2d
from .e6utils import bimodal_2d, img_moments, gaussian_2d, parabolic_2d


def smart_bimodal2d_fit(img, dec_rate=None, downsample=False, quiet=True, save=False, filename='test.png'):
    """
    :param img: Image to fit
    :param dec_rate: Decimate rate to speed up fitting if downsample is selected
    :param downsample: Boolean indicating whether or not to downsample
    :param quiet: Squelch variable
    :param save: Save fit visualization
    :param filename: file to save fit visualization to.
    :return: Returns a struct containing relevant data output of the fit routine
    Take an image as input and fits the image amplitude with a 2D gaussian.
    Attempts to guess fit values by extracting 2D mean and variance of the image. this only
    makes sense if the image intensity is mostly positive.
    """
    x_range = img.shape[0]
    y_range = img.shape[1]

    # Get fit guess values
    peakP_guess = img.max()-img.min()
    peakG_guess = peakP_guess/2
    B_guess = img.min()
    try:
        x0_guess, y0_guess, sx_guess, sy_guess = img_moments(img)
    except ValueError as e:
        # print(e)
        # print('Using default guess values.')
        x0_guess, y0_guess, sx_guess, sy_guess = [x_range/2, y_range/2, x_range/2, y_range/2]
    parx_guess=sx_guess
    pary_guess=sy_guess
    p_guess = np.array([x0_guess, y0_guess, sx_guess, sy_guess,
                        parx_guess, pary_guess, peakG_guess, peakP_guess, B_guess, 0])
    if not quiet:
        print(f'x0_guess = {x0_guess:.1f}')
        print(f'y0_guess = {y0_guess:.1f}')
        print(f'sx_guess = {sx_guess:.1f}')
        print(f'sy_guess = {sy_guess:.1f}')
        print(f'parx_guess = {parx_guess:.1f}')
        print(f'pary_guess = {pary_guess:.1f}')

    # Downsample image to speed up fit
    if downsample is True and dec_rate is None:
            # Resolution should be high enough so there are
            # at least dec_threshold pixels across one standard deviation.
            dec_threshold = 5
            dec_rate = np.max([np.min([sx_guess, sy_guess]) / dec_threshold, 1])
    elif downsample is False:
        dec_rate = 1
    img_down = scipy.ndimage.interpolation.zoom(img, 1 / dec_rate)
    rvec = np.indices(img_down.shape)
    if not quiet:
        print(f'Image downsampled by factor: {dec_rate:.1f}')

    # Perform fit
    # p_bounds = ([-np.inf, -np.inf, 0, 0, -np.inf, -np.inf, 0],
    #             [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 360])

    def img_cost_func(x):
        return np.ravel(bimodal_2d(rvec[0] * dec_rate, rvec[1] * dec_rate, *x) - img_down)

    t_fit_start = time.time()
    lsq_struct = least_squares(img_cost_func, p_guess, verbose=0)
    t_fit_stop = time.time()
    popt = lsq_struct.x
    pcov = lsq_struct.jac

    # Visualize fit outputs
    rvec = np.indices(img.shape)
    model_img = bimodal_2d(rvec[0], rvec[1], *popt)
    model_nobg = bimodal_2d(rvec[0], rvec[1], *popt[0:-2], 0, popt[-1])
    gauss_nobg = gaussian_2d(rvec[0], rvec[1], *popt[0:4], popt[6], 0, popt[-1])
    par_nobg = parabolic_2d(rvec[0], rvec[1], popt[0],popt[1],*popt[4:6], popt[7], 0, popt[-1])

    dict_param_keys = ['x0', 'y0', 'sx', 'sy', 'parx', 'pary', 'peakG','peakP', 'B', 'theta']

    fit_struct = dict()
    fit_struct['popt'] = popt
    # fit_struct['pcov'] = pcov
    for i in range(10):
        key = dict_param_keys[i]
        fit_struct[key] = popt[i]
    fit_struct['NSum'] = np.sum(img)
    fit_struct['NFit'] = np.sum(model_img)
    fit_struct['N_nobg'] = np.sum(model_nobg)
    fit_struct['NGauss'] = fit_struct['peakG']*2*np.pi*fit_struct['sx']*fit_struct['sy']
    fit_struct['NGauss_nobg'] = np.sum(gauss_nobg)
    fit_struct['NPar_nobg'] = np.sum(par_nobg)

    # fit_struct['data_img'] = img
    # fit_struct['model_img'] = model_img
    # fit_struct['model_nobg'] = model_nobg
    # fit_struct['gauss_nobg'] = gauss_nobg
    # fit_struct['par_nobg'] = par_nobg

    if not False:
        if not quiet:
            print(f'fit time = {t_fit_stop - t_fit_start:.2f} s')
        img_min = np.min([img.min(), model_img.min()])
        img_max = np.max([img.max(), model_img.max()])

        x_int_cut_dat = np.sum(img, axis=1)
        x_int_cut_model = np.sum(model_img, axis=1)
        y_int_cut_dat = np.sum(img, axis=0)
        y_int_cut_model = np.sum(model_img, axis=0)

        # Plotting
        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(2, 2, 1, position=[0, .5, .4, .4])
        ax.imshow(img, vmin=img_min, vmax=img_max)
        ax.set_aspect(x_range / y_range)
        ax.xaxis.tick_top()
        ax.set_xlabel('Horizontal Position')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Vertical Position')

        ax = fig.add_subplot(2, 2, 4, position=[.5, 0, .4, .4])
        ax.imshow(model_img, vmin=img_min, vmax=img_max)
        ax.set_aspect(x_range / y_range)
        ax.yaxis.tick_right()
        ax.set_xlabel('Horizontal Position')
        ax.set_ylabel('Vertical Position')
        ax.yaxis.set_label_position('right')

        ax = fig.add_subplot(2, 2, 2, position=[.5, .5, .4, .4])
        ax.plot(x_int_cut_dat, range(x_range))
        ax.plot(x_int_cut_model, range(x_range))
        ax.invert_yaxis()
        ax.yaxis.tick_right()
        ax.xaxis.tick_top()
        ax.set_xlabel('Integrated Intensity')
        ax.xaxis.set_label_position('top')

        ax = fig.add_subplot(2, 2, 3, position=[0, 0, .4, .4])
        ax.plot(range(y_range), y_int_cut_dat)
        ax.plot(range(y_range), y_int_cut_model)
        ax.invert_yaxis()
        ax.set_ylabel('Integrated Intensity')

        print_str = ''
        for i in range(10):
            print_str += f'{dict_param_keys[i]} = {popt[i]:.1f}\n'
        # fig.text(.52, .50, print_str)
        fig.text(1, .50, print_str)
        if quiet:
            plt.close(fig)
        if not quiet:
            plt.show()
        if save:
            plt.savefig(filename)
    t_tot = time.time()
    if not quiet:
        print(f'total time = {t_tot - t_fit_start:.2f} s')

    return fit_struct