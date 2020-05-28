from IPython.display import display, Math
import numpy as np
import pandas as pd
import csv
import scipy.constants as constants
from scipy.special import dawsn, erfc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import OrderedDict
from pathlib import Path
import os


def transmitted_intensity_deprecated(beta, nutilde, tau_s, t):
    # Equivalent to function transmitted_intensity but slower to calculate and more complicated
    tprime = t/tau_s
    arg1 = -tprime + 1j*(nutilde*tprime**2/2 - 1/(2*nutilde))
    arg2 = (1j+tprime*nutilde)/np.sqrt(2*1j*nutilde)
    return (beta**2/nutilde)*np.abs(np.sqrt(np.pi/2)*np.exp(arg1) + 1j*np.sqrt(2)*dawsn(arg2))**2


def transmitted_intensity(beta, nutilde, tau_s, t):
    tprime = t/tau_s
    arg1 = -tprime - (1j/2)*tprime**2*nutilde + 1j/(2*nutilde)
    arg2 = (1j-tprime*nutilde)/(np.sqrt(2*1j)*np.sqrt(nutilde))
    A = np.pi*beta**2/(2*np.abs(nutilde))
    return A*np.abs(np.exp(arg1)*erfc(arg2))**2


def fit_func(t, t0, tau_s, beta, offset, nutilde):
    return transmitted_intensity(beta, nutilde, tau_s, t-t0) + offset


def calc_params(popt, pcov, L):
    t0 = popt[0]
    tau_s = popt[1]
    beta = popt[2]
    offset = popt[3]
    nutilde = popt[4]

    fFSR = constants.c / (2 * L)

    kappa = 2 / tau_s
    finesse = 2 * np.pi * fFSR / kappa
    omegadot = nutilde / (tau_s ** 2)

    params = (t0, tau_s, beta, offset, nutilde, kappa, finesse, omegadot, L, fFSR)

    perr = np.sqrt(np.diagonal(pcov))
    t0_err = perr[0]
    tau_s_err = perr[1]
    beta_err = perr[2]
    offset_err = perr[3]
    nutilde_err = perr[4]
    cross_err = pcov[1, 4]

    kappa_err = 2 * tau_s_err / tau_s ** 2
    finesse_err = 2 * np.pi * fFSR * kappa_err / kappa ** 2
    omegadot_err = np.sqrt((nutilde_err / tau_s ** 2) ** 2 + (2 * tau_s_err * nutilde / tau_s ** 3) ** 2 + 2 * (
                -2 * nutilde / tau_s ** 5) * cross_err)

    params_err = (t0_err, tau_s_err, beta_err, offset_err, nutilde_err, kappa_err, finesse_err, omegadot_err)

    return params, params_err


def disp_params(params, params_err):
    t0, tau_s, beta, offset, nutilde, kappa, finesse, omegadot, L, fFSR = params
    t0_err, tau_s_err, beta_err, offset_err, nutilde_err, kappa_err, finesse_err, omegadot_err = params_err

    display(Math(rf'\kappa = 2\pi \times ({kappa/(2*np.pi)*1e-6:.3f} '
                 rf'\pm {kappa_err/(2*np.pi)*1e-6:.3f} \text{{MHz}})'))
    display(Math(rf'\mathcal{{F}} = {finesse:.0f} \pm {finesse_err:.0f}'))
    display(Math(
        rf'\dot{{\omega}} = 2\pi \times ({omegadot/(2*np.pi)*1e-9*1e-3:.3f} '
        rf'\pm {omegadot_err/(2*np.pi)*1e-9*1e-3:.3f} \frac{{\text{{GHz}}}}{{\text{{ms}}}})'))
    print('\n')
    display(Math(rf'L = {L*1e3:.1f} \text{{ mm}}'))
    display(Math(rf'\nu_{{FSR}} = {fFSR*1e-9:.3f} \text{{GHz}}'))
    print('\n')
    display(Math(rf't_0 = {t0*1e6:.3f} \pm {t0_err*1e6:.3f} \text{{ $\mu$s}}'))
    display(Math(rf'\tau_s = {tau_s*1e6:.3f} \pm {tau_s_err*1e6:.3f} \text{{ $\mu$s}}'))
    display(Math(rf'\beta = {beta:.3f} \pm {beta_err:.3f}'))
    display(Math(rf'\text{{offset}} = {offset:.3f} \pm {offset_err:.3f}'))
    display(Math(rf'\tilde{{\nu}} = {nutilde:.3f} \pm {nutilde_err:.3f}'))
    return


def extract_data(file_name):
    data = pd.read_csv(file_name, usecols=(3, 4), skiprows=18,
                       delimiter=',', names=["Time (s)", "Voltage (V)"])
    fit_t = data['Time (s)'].values
    fit_data = data['Voltage (V)'].values
    return fit_t, fit_data


def csv_write_overwrite(save_file, data_dict, is_duplicate=lambda x, y: x['uid'] == y['uid']):
    temp_file = Path(str(save_file).strip('.csv') + '_temp.csv')
    # new_data_uid = data_dict[uid_field]
    data_saved = False
    with open(save_file, 'r', newline='') as read_file:
        csv_reader = csv.DictReader(read_file)
        with open(temp_file, 'w', newline='') as write_file:
            csv_writer = csv.DictWriter(write_file, fieldnames=data_dict.keys())
            csv_writer.writeheader()
            for line in csv_reader:
                # if line['uid'] != new_data_uid:
                if not is_duplicate(line, data_dict):
                    csv_writer.writerow(line)
                # elif line['uid'] == new_data_uid:
                elif is_duplicate(line, data_dict):
                    csv_writer.writerow(data_dict)
                    data_saved = True
            if not data_saved:
                csv_writer.writerow(data_dict)
    os.remove(save_file)
    os.rename(temp_file, save_file)


def save_data(save_file, mirror_1, mirror_2, finesse_mean, finesse_err, cav_length, date_str, filename, notes):
    mirror_1 = f'{mirror_1:d}'
    mirror_2 = f'{mirror_2:d}'
    finesse_mean = f'{finesse_mean:.0f}'
    finesse_err = f'{finesse_err:.0f}'
    cav_length = f'{cav_length*1e3:.2f}'  # cav_length expressed in mm
    uid = date_str+filename
    new_data = [mirror_1, mirror_2, finesse_mean, finesse_err, cav_length, date_str, filename, notes, uid]
    fieldnames = ['mirror_1', 'mirror_2', 'finesse_mean', 'finesse_err', 'cav_length (mm)',
                  'date', 'filename', 'notes', 'uid']
    data_dict = OrderedDict(zip(fieldnames, new_data))
    csv_write_overwrite(save_file, data_dict)
    print(f'Data saved in {save_file}')


def ringdown_fit(file_name, L=5e-3, p_guess=(0, 1.5e-7, 0.5, 0, 1)):
    fit_t, fit_data = extract_data(file_name)
    popt, pcov = curve_fit(fit_func, fit_t, fit_data, p_guess)

    plt.plot(fit_t*1e6, fit_data, label='Data')
    plt.plot(fit_t*1e6, fit_func(fit_t, *p_guess), label='Guess Curve')
    plt.plot(fit_t*1e6, fit_func(fit_t, *popt), label='Fit Curve', color='r')
    plt.xlabel('Time ($\mu$s)')
    plt.ylabel('Signal (a.u.)')
    plt.legend()
    plt.title('Fitted Data')
    plt.show()

    params, params_err = calc_params(popt, pcov, L)
    disp_params(params, params_err)
    return list(zip(params, params_err))
