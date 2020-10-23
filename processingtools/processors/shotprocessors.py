from enum import Enum
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.constants import hbar
from scipy.interpolate import interp1d
import h5py
import xarray as xr
from ..imagetools import get_image
from ..datamodel import qprint
from ..datastreamtools import get_gagescope_trace
from .processor import Processor, ProcessorWeight, ProcessorScale
from ...smart_gaussian2d_fit import fit_gaussian2d
from ..fittools import lor_fit


class ShotProcessor(Processor):
    class ResultKey(Enum):
        pass

    def __init__(self, *, name, weight, reset):
        super(ShotProcessor, self).__init__(name=name, weight=weight, scale=ProcessorScale.SHOT)
        self.reset = reset

    def scaled_process(self, datamodel, processor_dict, quiet=False):
        data_dict = datamodel.data_dict

        num_shots = data_dict['num_shots']
        for shot_num in range(num_shots):
            shot_key = f'shot-{shot_num:d}'
            if shot_key not in processor_dict['results'] or self.reset:
                qprint(f'processing {shot_key}', quiet=quiet)
                results_dict = self.process_shot(shot_num, datamodel)
                processor_dict['results'][shot_key] = results_dict
                data_dict.save_dict(quiet=True)
            else:
                qprint(f'skipping processing {shot_key}', quiet=quiet)

    def process_shot(self, shot_num, datamodel):
        raise NotImplementedError


class CountsShotProcessor(ShotProcessor):
    class ResultKey(Enum):
        COUNTS = 'counts'

    def __init__(self, *, datastream_name, frame_name, roi_slice, name, reset):
        super(CountsShotProcessor, self).__init__(name=name, weight=ProcessorWeight.LIGHT, reset=reset)
        self.datastream_name = datastream_name
        self.frame_name = frame_name
        self.roi_slice = roi_slice

    def process_shot(self, shot_num, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        file_path = datastream.get_file_path(shot_num)
        frame = get_image(file_path, self.frame_name, roi_slice=self.roi_slice)
        counts = np.nansum(frame)
        results_dict = {self.ResultKey.COUNTS.value: counts}
        return results_dict


rb_atom_dict = dict()
rb_atom_dict['atom_name'] = 'Rb87'
rb_atom_dict['cross_section'] = 2.907e-13  # m^2 - Steck Rubidium 87 D Line Data
rb_atom_dict['linewidth'] = 2 * np.pi * 6.07e6  # Hz -  Steck Rubidium 87 D Line Data

# W/m^2 - Steck Rubidium 87 D Line Data, convert mW/cm^2 to W/m^2
rb_atom_dict['saturation_intensity'] = 1.67 * 1e4 / 1e3

rb_atom_dict['transition_frequency'] = 2 * np.pi * 384.230e12  # Hz - D2 Transition


quantum_efficiency = 0.38  # Number of electrons per photon
adu_conversion = 1 / 0.37  # Number of digital counts per electron, ADU/e-
bit_conversion = 2 ** -8  # 16 bit camera output is converted to 8 bits before data is captured
total_gain = bit_conversion * adu_conversion * quantum_efficiency  # Number of recoreded digital counts per photon

side_imaging_dict = dict()
side_imaging_dict['magnification'] = 0.77
side_imaging_dict['pixel_area'] = 6.45e-6 ** 2
side_imaging_dict['count_conversion'] = total_gain


class AbsorptionShotProcessor(ShotProcessor):
    class ResultKey(Enum):
        ABSORPTION_IMAGE = 'absorption_image'
        OD_IMAGE = 'od_image'

    def __init__(self, *, datastream_name, atom_frame_name, bright_frame_name, dark_frame_name,
                 atom_dict, imaging_system_dict, roi_slice, calc_high_sat, name, reset):
        super(AbsorptionShotProcessor, self).__init__(name=name, weight=ProcessorWeight.HEAVY, reset=reset)
        if imaging_system_dict is None:
            imaging_system_dict = side_imaging_dict
        if atom_dict is None:
            atom_dict = rb_atom_dict
        self.datastream_name = datastream_name
        self.atom_frame_name = atom_frame_name
        self.bright_frame_name = bright_frame_name
        self.dark_frame_name = dark_frame_name
        self.atom_dict = atom_dict
        self.imaging_system_dict = imaging_system_dict
        self.roi_slice = roi_slice
        self.calc_high_sat = calc_high_sat

        self.linewidth = self.atom_dict['linewidth']
        self.cross_section = self.atom_dict['cross_section']
        self.transition_frequency = self.atom_dict['transition_frequency']
        self.saturation_intensity = self.atom_dict['saturation_intensity']

        self.pixel_area = self.imaging_system_dict['pixel_area']
        self.magnification = self.imaging_system_dict['magnification']
        self.count_conversion = self.imaging_system_dict['count_conversion']

    def process_shot(self, shot_num, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        file_path = datastream.get_file_path(shot_num)
        atom_frame = get_image(file_path, self.atom_frame_name, roi_slice=self.roi_slice)
        bright_frame = get_image(file_path, self.bright_frame_name, roi_slice=self.roi_slice)
        dark_frame = get_image(file_path, self.dark_frame_name, roi_slice=self.roi_slice)
        od_frame, atom_frame = self.absorption_od_and_number(atom_frame, bright_frame, dark_frame)
        results_dict = dict()
        results_dict[self.ResultKey.ABSORPTION_IMAGE.value] = atom_frame
        results_dict[self.ResultKey.OD_IMAGE.value] = od_frame
        return results_dict

    def absorption_od_and_number(self, atom_frame, bright_frame, dark_frame):
        atom_counts, bright_counts = self.absorption_bg_subtract(atom_frame, bright_frame, dark_frame)
        optical_density = self.optical_density_analysis(atom_counts, bright_counts)
        atom_number = self.atom_count_analysis(atom_counts, bright_counts, optical_density, calc_high_sat=True)
        return optical_density, atom_number

    @staticmethod
    def absorption_bg_subtract(atom_frame, bright_frame, dark_frame):
        atom_counts = atom_frame - dark_frame
        bright_counts = bright_frame - dark_frame
        return atom_counts, bright_counts

    @staticmethod
    def optical_density_analysis(atom_counts, bright_counts):
        """
        Calculate transmissivity and optical density. Note that data is rejected if atom_counts > bright counts or
        if either one is negative. These conditions can arise due noise in the beams including shot noise or
        temporal fluctuations in beam powers. This seems like the easiest way to handle these edge cases but it could
        lead to biases in atom number estimations.
        """
        transmissivity = np.true_divide(atom_counts, bright_counts,
                                        out=np.full_like(atom_counts, 0, dtype=float),
                                        where=np.logical_and(atom_counts > 0, bright_counts > 0))
        optical_density = -1 * np.log(transmissivity, out=np.full_like(atom_counts, np.nan, dtype=float),
                                      where=np.logical_and(0 < transmissivity, transmissivity <= 1))
        return optical_density

    def atom_count_analysis(self, atom_counts, bright_counts, optical_density=None, calc_high_sat=True):
        if optical_density is None:
            optical_density = self.optical_density_analysis(atom_counts, bright_counts)
        low_sat_atom_number = self.atom_count_analysis_below_sat(optical_density)
        if calc_high_sat:
            high_sat_atom_number = self.atom_count_analysis_above_sat(atom_counts, bright_counts)
        else:
            high_sat_atom_number = 0
        atom_number = low_sat_atom_number + high_sat_atom_number
        return atom_number

    def atom_count_analysis_below_sat(self, optical_density,
                                      detuning=0):
        detuning_factor = 1 + (2 * detuning / self.linewidth) ** 2
        column_density_below_sat = (detuning_factor / self.cross_section) * optical_density
        column_area = self.pixel_area / self.magnification  # size of a pixel in object plane
        column_number = column_area * column_density_below_sat
        return column_number

    def atom_count_analysis_above_sat(self, atom_counts, bright_counts, image_pulse_time=100e-6,
                                      efficiency_path=1.0):
        # convert counts to detected photons
        atom_photons_det = atom_counts / self.count_conversion
        bright_photons_det = bright_counts / self.count_conversion

        # convert detected photons to detected intensity
        atom_intensity_det = (atom_photons_det *
                              (hbar * self.transition_frequency) / (self.pixel_area * image_pulse_time))
        bright_intensity_det = (bright_photons_det *
                                (hbar * self.transition_frequency) / (self.pixel_area * image_pulse_time))

        # convert detected intensity to intensity before and after atoms
        intensity_out = atom_intensity_det / efficiency_path / self.magnification
        intensity_in = bright_intensity_det / efficiency_path / self.magnification

        # convert intensity in and out to resonant saturation parameter in and out
        s0_out = intensity_out / self.saturation_intensity
        s0_in = intensity_in / self.saturation_intensity

        # calculate column density from s0_in and s0_out
        column_density = (s0_in - s0_out) / self.cross_section

        # calculate column atom number from column_density and column_area
        column_area = self.pixel_area / self.magnification  # size of a pixel in the object plane
        column_number = column_density * column_area
        return column_number


# noinspection PyPep8Naming
class HetDemodulationShotProcessor(ShotProcessor):
    class ResultKey(Enum):
        RESULT_FILE_PATH = 'result_file_path'

    def __init__(self, *, datastream_name, channel_name, segment_name, carrier_frequency,
                 bandwidth, downsample_rate, name, reset):
        super(HetDemodulationShotProcessor, self).__init__(name=name, weight=ProcessorWeight.HEAVY, reset=reset)
        self.datastream_name = datastream_name
        self.channel_name = channel_name
        self.segment_name = segment_name
        self.carrier_frequency = carrier_frequency
        self.bandwidth = bandwidth
        self.downsample_rate = downsample_rate

    def process_shot(self, shot_num, datamodel):
        datastream = datamodel.datastream_dict[self.datastream_name]
        file_path = datastream.get_file_path(shot_num)
        output_data_path = Path(datamodel.daily_path, 'analysis',
                                datamodel.run_name, self.name)
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file_path = Path(output_data_path, f'het_analysis_{shot_num:05d}.h5')

        raw_het, dt = get_gagescope_trace(file_path, self.channel_name, self.segment_name)
        num_samples = len(raw_het)
        t_coord = np.arange(0, num_samples) * dt

        raw_het_xr = xr.DataArray(raw_het, coords={'time': t_coord}, dims=['time'])
        I_het, Q_het, A_het, phi_het, time_series = self.demodulate(raw_het_xr, self.downsample_rate, dt)

        with h5py.File(str(output_file_path), 'w') as hf:
            hf.create_dataset('I_het', data=I_het.astype('float'))
            hf.create_dataset('Q_het', data=Q_het.astype('float'))
            hf.create_dataset('A_het', data=A_het.astype('float'))
            hf.create_dataset('phi_het', data=phi_het.astype('float'))
            hf.create_dataset('time_series', data=time_series.astype('float'))

        results_dict = dict()
        results_dict[self.ResultKey.RESULT_FILE_PATH.value] = output_file_path
        return results_dict

    def butter_lowpass_filter(self, data, order, dt):
        nyquist_freq = (1 / 2) * (1 / dt)
        normalized_cutoff = self.bandwidth / nyquist_freq
        # Get the filter coefficients
        # noinspection PyTupleAssignmentBalance
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def demodulate(self, het_xr, downsample_rate, dt):
        time_data = het_xr.coords['time'].values
        lo_signal = np.exp(1j * 2 * np.pi * self.carrier_frequency * time_data)
        demod_sig = het_xr * lo_signal
        demod_sig = self.butter_lowpass_filter(demod_sig, 3, dt)
        demod_sig = demod_sig[::downsample_rate]
        I_sig = np.real(demod_sig)
        Q_sig = np.imag(demod_sig)
        A_sig = np.sqrt(I_sig ** 2 + Q_sig ** 2)
        phi_sig = np.arctan2(Q_sig, I_sig)
        time_data = time_data[::downsample_rate]
        I_xr = xr.DataArray(I_sig, dims=['time'], coords={'time': time_data})
        Q_xr = xr.DataArray(Q_sig, dims=['time'], coords={'time': time_data})
        A_xr = xr.DataArray(A_sig, dims=['time'], coords={'time': time_data})
        phi_xr = xr.DataArray(phi_sig, dims=['time'], coords={'time': time_data})
        return I_xr, Q_xr, A_xr, phi_xr, time_data


class AbsorptionGaussianFitShotProcessor(ShotProcessor):
    class ResultKey(Enum):
        GAUSSIAN_FIT_STRUCT = 'gaussian_fit_struct'

    def __init__(self, *, source_processor_name, name, reset):
        super(AbsorptionGaussianFitShotProcessor, self).__init__(name=name, weight=ProcessorWeight.LIGHT, reset=reset)
        self.source_processor_name = source_processor_name

    def process_shot(self, shot_num, datamodel):
        data_dict = datamodel.data_dict
        shot_key = f'shot-{shot_num:d}'
        frame = data_dict['shot_processors'][self.source_processor_name]['results'][shot_key]['absorption_image']
        fit_struct = fit_gaussian2d(frame, show_plot=False, save_name=None, quiet=True)
        results_dict = {self.ResultKey.GAUSSIAN_FIT_STRUCT.value: fit_struct}
        return results_dict


# noinspection PyPep8Naming
class CavSweepFitShotProcessor(ShotProcessor):
    class ResultKey(Enum):
        LOR_FIT_STRUCT = 'lor_fit_struct'

    def __init__(self, *, het_demod_processor_name, vco_channel_name, name, reset):
        super(CavSweepFitShotProcessor, self).__init__(name=name, weight=ProcessorWeight.LIGHT, reset=reset)
        self.het_demod_processor_name = het_demod_processor_name
        self.vco_channel_name = vco_channel_name

    def process_shot(self, shot_num, datamodel):
        shot_key = f'shot-{shot_num:d}'
        data_dict = datamodel.data_dict
        het_demod_processor_dict = data_dict['shot_processors'][self.het_demod_processor_name]

        A_het, het_time_trace = self.get_A_het_trace(het_demod_processor_dict, shot_key)
        vco_trace, vco_time_trace = self.get_vco_trace(datamodel, het_demod_processor_dict, shot_num)

        interp_func = interp1d(vco_time_trace, vco_trace)
        vco_trace_interp = interp_func(het_time_trace)

        fit_struct = lor_fit(vco_trace_interp, A_het, quiet=True)
        results_dict = {self.ResultKey.LOR_FIT_STRUCT.value: fit_struct}
        return results_dict

    @staticmethod
    def get_A_het_trace(het_demod_processor_dict, shot_key):
        het_demod_file_path = het_demod_processor_dict['results'][shot_key]['result_file_path']
        time_trace = h5py.File(het_demod_file_path, 'r')['time_series']
        A_het = h5py.File(het_demod_file_path, 'r')['A_het']
        return A_het, time_trace

    def get_vco_trace(self, datamodel, het_demod_processor_dict, shot_num):
        datastream_name = (het_demod_processor_dict['input_param_dict']['kwargs']['datastream_name'])
        datastream = datamodel.datastream_dict[datastream_name]

        segment_name = het_demod_processor_dict['input_param_dict']['kwargs']['segment_name']

        vco_file_path = datastream.get_file_path(shot_num)
        vco_trace, dt = get_gagescope_trace(vco_file_path, self.vco_channel_name, segment_name)
        num_samples = len(vco_trace)
        time_trace = np.arange(0, num_samples) * dt
        vco_trace = self.vco_v_to_f(vco_trace)
        return vco_trace, time_trace

    @staticmethod
    def vco_v_to_f(volt):
        # Calibration taken from
        freq = 2 * (62.970 * volt + 42.014)
        return freq
