import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
import scipy.stats
import matplotlib.pyplot as plt


def make_fit_param_dict(name, val, std, conf_level=erf(1 / np.sqrt(2)), dof=None):
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


def create_fit_struct(fit_func, input_data, output_data, popt_dict, pcov, conf_level, dof, lightweight=False):
    popt = list(popt_dict.values())
    model_data = fit_func(input_data, *popt)
    fit_struct = dict()
    fit_struct_param_keys = []
    for i, key in enumerate(popt_dict.keys()):
        fit_param_dict = make_fit_param_dict(key, popt_dict[key], np.sqrt(pcov[i, i]), conf_level, dof)
        fit_struct[key] = fit_param_dict
        fit_struct_param_keys.append(key)
    fit_struct['param_keys'] = fit_struct_param_keys
    kwargs = dict()
    for key in fit_struct_param_keys:
        val = fit_struct[key]['val']
        kwargs[key] = val
    fit_struct['kwargs'] = kwargs
    fit_struct['cov'] = pcov
    if not lightweight:
        fit_struct['input_data'] = input_data
        fit_struct['output_data'] = output_data
        fit_struct['model'] = model_data
    return fit_struct


def e6_fit(output_data, fit_func, param_guess, input_data=None, param_keys=None, conf_level=erf(1 / np.sqrt(2)),
           lightweight=False, *args, **kwargs):
    if param_keys is None:
        param_keys = []
        for idx, param in enumerate(param_guess):
            param_keys.append(f'fit_param_{idx}')

    if input_data is None:
        input_data = np.indices(output_data.shape)
    input_data = input_data
    output_data = np.nan_to_num(output_data)

    def img_cost_func(fit_params):
        return np.nan_to_num(np.ravel(fit_func(input_data, *fit_params) - output_data))

    lsq_struct = least_squares(img_cost_func, param_guess, verbose=0, *args, **kwargs)

    popt = lsq_struct['x']
    popt_dict = dict(zip(param_keys, popt))
    jac = lsq_struct['jac']
    cost = lsq_struct['cost']

    n_data_points = np.prod(output_data.shape)
    n_fit_parameters = len(popt_dict)
    dof = n_data_points - n_fit_parameters
    sigma_squared = 2 * cost / dof
    try:
        cov = sigma_squared * np.linalg.inv(np.matmul(jac.T, jac))
    except np.linalg.LinAlgError as e:
        print(e)
        cov = 0 * jac

    fit_struct = create_fit_struct(fit_func, input_data, output_data, popt_dict, cov, conf_level, dof,
                                   lightweight=lightweight)
    return fit_struct


def lorentzian_fit_function(x, delta_x, x0, amplitude, offset):
    delta_x_hwhm = delta_x / 2
    detuning = x - x0
    return amplitude * (delta_x_hwhm**2) / (delta_x_hwhm**2 + detuning**2) + offset


def lor_fit(x_data, y_data, param_guess=None, x_label='', y_label='Signal', x_units='', y_units='a.u.',
            quiet=False, lightweight=False):
    # TODO: Return error bars
    param_keys = ['delta_x', 'x0', 'amplitude', 'offset']

    if param_guess is None:
        delta_x_guess = (np.nanmax(x_data) - np.nanmin(x_data)) / 2
        x0_guess = (np.nanmax(x_data) + np.nanmin(x_data)) / 2
        amplitude_guess = np.nanmax(y_data) - np.nanmin(y_data)
        offset_guess = np.nanmin(y_data)
        param_guess=(delta_x_guess, x0_guess, amplitude_guess, offset_guess)

    fit_struct = e6_fit(y_data, lorentzian_fit_function, input_data=x_data,
                        param_guess=param_guess, param_keys=param_keys, lightweight=lightweight)
    popt = [fit_struct[key]['val'] for key in fit_struct['param_keys']]

    if not quiet:
        print(f'Linewidth (FWHM) = {popt[0]:.2f} {x_units}')
        print(f'Center Frequency = {popt[1]:.2f} {x_units}')
        print(f'Amplitude = {popt[2]:.2f} {y_units}')
        print(f'Offset = {popt[3]:.2f} {y_units}')

        x_min = np.min(x_data)
        x_max = np.max(x_data)
        x_range = (1 / 2) * (x_max - x_min)
        x_center = (1 / 2) * (x_max + x_min)
        plot_min = x_center - 1.1 * x_range
        plot_max = x_center + 1.1 * x_range
        plot_x_list = np.linspace(plot_min, plot_max, 100)
        plt.plot(x_data, y_data, '.', markersize=10)
        plt.plot(plot_x_list, lorentzian_fit_function(plot_x_list, *popt))
        plt.xlabel(f'{x_label} ({x_units})')
        plt.ylabel(f'{y_label} ({y_units})')
        plt.show()
    return fit_struct