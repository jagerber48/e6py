import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from uncertainties import ufloat
from .datamodel import dataset_from_keychain, shot_to_loop_and_point
from .fittools import lorentzian_fit_function


class Reporter:
    def __init__(self, *, reporter_name):
        self.reporter_name = reporter_name

    def report(self, datamodel):
        raise NotImplementedError


class AtomRefCountsReporter(Reporter):
    def __init__(self, *, atom_counts_processor_name, ref_counts_processor_name, reporter_name='counts_reporter'):
        super(AtomRefCountsReporter, self).__init__(reporter_name=reporter_name)
        self.atom_counts_processor_name = atom_counts_processor_name
        self.ref_counts_processor_name = ref_counts_processor_name

    def report(self, datamodel):
        data_dict = datamodel.data_dict
        run_name = data_dict['run_name']
        num_points = data_dict['num_points']

        for point in range(num_points):
            point_key = f'point-{point:d}'
            shot_list = data_dict['shot_list'][point_key]
            atom_data = []
            ref_data = []
            for shot in shot_list:
                atom_counts = (data_dict['shot_processors'][self.atom_counts_processor_name]
                                        ['results'][f'shot-{shot}']['counts'])
                ref_counts = (data_dict['shot_processors'][self.ref_counts_processor_name]
                                       ['results'][f'shot-{shot}']['counts'])
                atom_data.append(atom_counts)
                ref_data.append(ref_counts)
            fig = plt.figure(figsize=(12, 12))

            ax_loop = fig.add_subplot(2, 1, 1)
            ax_loop.set_xlabel('Loop Number')
            ax_loop.set_ylabel('Counts')

            ax_hist = fig.add_subplot(2, 1, 2)
            ax_hist.set_xlabel('Counts')
            ax_hist.set_ylabel('Frequency')

            for y_data in [atom_data, ref_data]:
                ax_loop.plot(y_data, '.', markersize=10)
                ax_hist.hist(y_data, alpha=0.5)

            figure_title = f'{self.reporter_name} - {run_name} - {point_key}'
            fig.suptitle(figure_title, fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            daily_path = data_dict['daily_path']
            save_path = Path(daily_path, 'analysis', run_name)
            save_file_path = Path(save_path, f'{figure_title}.png')
            fig.savefig(save_file_path)

        plt.show()


class AvgRndmImgReporter(Reporter):
    def __init__(self, avg_processor_name, rndm_processor_name,
                 reporter_name='avg_rndm_img_reporter'):
        super(AvgRndmImgReporter, self).__init__(reporter_name=reporter_name)
        self.avg_processor_name = avg_processor_name
        self.rndm_processor_name = rndm_processor_name

    def report(self, datamodel):
        data_dict = datamodel.data_dict
        run_name = data_dict['run_name']
        num_points = data_dict['num_points']

        cmap = 'binary_r'

        for point in range(num_points):
            point_key = f'point-{point:d}'
            num_loops = data_dict['loop_nums'][point_key]

            atom_rndm_img = dataset_from_keychain(datamodel,
                                                  f'point_processors/{self.rndm_processor_name}/results'
                                                  f'/{point_key}/random_atom_img')
            ref_rndm_img = dataset_from_keychain(datamodel,
                                                 f'point_processors/{self.rndm_processor_name}/results'
                                                 f'/{point_key}/random_atom_img')
            atom_avg_img = dataset_from_keychain(datamodel,
                                                 f'point_processors/{self.avg_processor_name}/results'
                                                 f'/{point_key}/avg_atom_img')
            ref_avg_img = dataset_from_keychain(datamodel,
                                                f'point_processors/{self.avg_processor_name}/results'
                                                f'/{point_key}/avg_atom_img')

            single_min_list = [np.nanmin(img) for img in [atom_rndm_img, ref_rndm_img]]
            single_min_val = np.nanmin(single_min_list)
            single_max_list = [np.nanmax(img) for img in [atom_rndm_img, ref_rndm_img]]
            single_max_val = np.nanmax(single_max_list)

            avg_min_list = [np.nanmin(img) for img in [atom_avg_img, ref_avg_img]]
            avg_min_val = np.nanmin(avg_min_list)
            avg_max_list = [np.nanmax(img) for img in [atom_avg_img, ref_avg_img]]
            avg_max_val = np.nanmax(avg_max_list)

            random_shot_num = dataset_from_keychain(datamodel,
                                                    f'point_processors/{self.rndm_processor_name}/results'
                                                    f'/{point_key}/random_shot_num')

            fig = plt.figure(figsize=(12, 12))

            ax_atom_rndm = fig.add_subplot(2, 2, 1)
            im = ax_atom_rndm.imshow(atom_rndm_img, vmin=single_min_val, vmax=single_max_val, cmap=cmap)
            fig.colorbar(im, ax=ax_atom_rndm)
            ax_atom_rndm.set_title(f'{run_name}:  Single Atom Frame - Point {point} - Shot #{random_shot_num}')

            ax_ref_rndm = fig.add_subplot(2, 2, 2)
            im = ax_ref_rndm.imshow(ref_rndm_img, vmin=single_min_val, vmax=single_max_val, cmap=cmap)
            fig.colorbar(im, ax=ax_ref_rndm)
            ax_ref_rndm.set_title(f'{run_name}:  Single Reference Frame - Point {point} - Shot #{random_shot_num}')

            ax_atom_avg = fig.add_subplot(2, 2, 3)
            im = ax_atom_avg.imshow(atom_avg_img, vmin=avg_min_val, vmax=avg_max_val, cmap=cmap)
            fig.colorbar(im, ax=ax_atom_avg)
            ax_atom_avg.set_title(f'{run_name}:  Average Atom Frame - Point {point} - {num_loops} Loops')

            ax_ref_avg = fig.add_subplot(2, 2, 4)
            im = ax_ref_avg.imshow(ref_avg_img, vmin=avg_min_val, vmax=avg_max_val, cmap=cmap)
            fig.colorbar(im, ax=ax_ref_avg)
            ax_ref_avg.set_title(f'{run_name}:  Average Reference Frame - Point {point} - {num_loops} Loops')

            figure_title = f'{self.reporter_name} - {run_name} - {point_key}'
            fig.suptitle(figure_title, fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            daily_path = data_dict['daily_path']
            save_path = Path(daily_path, 'analysis', run_name)
            save_file_path = Path(save_path, f'{figure_title}.png')
            fig.savefig(save_file_path)

        plt.show()


class AllShotsReporter(Reporter):
    def __init__(self, *, reporter_name, reset):
        super(AllShotsReporter, self).__init__(reporter_name=reporter_name)
        self.reset = reset

    def report(self, datamodel):
        data_dict = datamodel.data_dict
        run_name = data_dict['run_name']
        num_points = data_dict['num_points']

        daily_path = data_dict['daily_path']
        save_dir = Path(daily_path, 'analysis', run_name, 'reporters', self.reporter_name)

        for point in range(num_points):
            point_key = f'point-{point:d}'
            shot_list = data_dict['shot_list'][point_key]
            for shot_num in shot_list:
                loop_num, _ = shot_to_loop_and_point(shot=shot_num, num_points=num_points)
                plot_title = f'{self.reporter_name} - {run_name} - {point_key} - loop-{loop_num} - shot-{shot_num}'
                save_file_path = Path(save_dir, point_key, f'{self.reporter_name}_{shot_num:05d}.png')
                if not save_file_path.exists() or self.reset:
                    fig = self.report_shot(shot_num, datamodel)
                    fig.suptitle(plot_title, fontsize=16)
                    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    save_file_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_file_path)
                    plt.close(fig)

    def report_shot(self, shot_num, datamodel):
        raise NotImplementedError


class ImageAllShotsReporter(AllShotsReporter):
    def __init__(self, *, reporter_name, image_dir_path, file_prefix, image_name, reset):
        super(ImageAllShotsReporter, self).__init__(reporter_name=reporter_name, reset=reset)
        self.image_dir_path = image_dir_path
        self.file_prefix = file_prefix


class GaussianFitAllShotsReporter(AllShotsReporter):
    def __init__(self, *, reporter_name, gaussian_fit_processor, reset):
        super(GaussianFitAllShotsReporter, self).__init__(reporter_name=reporter_name, reset=reset)
        self.gaussian_fit_processor = gaussian_fit_processor

    # noinspection PyPep8Naming
    def report_shot(self, shot_num, datamodel):
        data_dict = datamodel.data_dict
        shot_key = f'shot-{shot_num:d}'
        fit_struct = (data_dict['shot_processors'][self.gaussian_fit_processor]
                      ['results'][shot_key]['gaussian_fit_struct'])

        img = fit_struct['data_img']
        model_img = fit_struct['model_img']
        x_range = img.shape[1]
        y_range = img.shape[0]
        x0 = int(round(fit_struct['x0']['val']))
        y0 = int(round(fit_struct['y0']['val']))
        sx = fit_struct['sx']['val']
        sy = fit_struct['sy']['val']
        img_min = np.min([img.min(), model_img.min()])
        img_max = np.max([img.max(), model_img.max()])

        # Plotting
        fig = plt.figure(figsize=(12, 12))

        # Data 2D Plot
        ax_data = fig.add_subplot(2, 2, 1, position=[0.1, 0.5, 0.25, 0.35])
        ax_data.imshow(img, vmin=img_min, vmax=img_max, cmap='binary_r')
        # TODO: check aspect ratio
        ax_data.set_aspect(y_range / x_range)
        ax_data.xaxis.tick_top()
        ax_data.set_xlabel('Horizontal Position')
        ax_data.xaxis.set_label_position('top')
        ax_data.set_ylabel('Vertical Position')
        ax_data.set_ylim(0, y_range)
        ax_data.set_xlim(0, x_range)

        # Fit 2D Plot
        ax_fit = fig.add_subplot(2, 2, 4, position=[0.4, 0.1, 0.25, 0.35])
        ax_fit.imshow(model_img, vmin=img_min, vmax=img_max, cmap='binary_r')
        ax_fit.set_aspect(y_range / x_range)
        ax_fit.yaxis.tick_right()
        ax_fit.set_xlabel('Horizontal Position')
        ax_fit.set_ylabel('Vertical Position')
        ax_fit.yaxis.set_label_position('right')
        ax_fit.set_ylim(0, y_range)
        ax_fit.set_xlim(0, x_range)

        # Y Linecut Plot
        ax_yline = fig.add_subplot(2, 2, 2, position=[0.4, 0.5, 0.25, 0.35])
        y_int_cut_dat = np.sum(img, axis=1) / np.sqrt(2 * np.pi * sx ** 2)
        y_int_cut_model = np.sum(model_img, axis=1) / np.sqrt(2 * np.pi * sx ** 2)
        ax_yline.plot(y_int_cut_dat, range(y_range), 'o', zorder=1)
        ax_yline.plot(y_int_cut_model, range(y_range), zorder=2)
        ax_yline.yaxis.tick_right()
        ax_yline.xaxis.tick_top()
        ax_yline.set_xlabel('Integrated Intensity')
        ax_yline.xaxis.set_label_position('top')
        ax_data.axvline(x0, linestyle='--')
        ax_fit.axvline(x0, linestyle='--')

        # X Linecut Plot
        ax_xline = fig.add_subplot(2, 2, 3, position=[0.1, 0.1, 0.25, 0.35])
        x_int_cut_dat = np.sum(img, axis=0) / np.sqrt(2 * np.pi * sy ** 2)
        x_int_cut_model = np.sum(model_img, axis=0) / np.sqrt(2 * np.pi * sy ** 2)
        ax_xline.plot(range(x_range), x_int_cut_dat, 'o', zorder=1)
        ax_xline.plot(range(x_range), x_int_cut_model, zorder=2)
        ax_xline.invert_yaxis()
        ax_xline.set_ylabel('Integrated Intensity')
        ax_data.axhline(y0, linestyle='--')
        ax_fit.axhline(y0, linestyle='--')

        print_str = ''
        for key in fit_struct['param_keys']:
            param = fit_struct[key]
            val = round(param['val'], 3)
            std = round(param['std'], 3)
            print_str += f"{key} = {ufloat(val, std)}\n"
        fig.text(.8, .5, print_str)

        return fig


class LorFitAllShotsReporter(AllShotsReporter):
    def __init__(self, *, reporter_name, lor_fit_processor, x_label, x_units, y_label, y_units, reset):
        super(LorFitAllShotsReporter, self).__init__(reporter_name=reporter_name, reset=reset)
        self.lor_fit_processor = lor_fit_processor
        self.x_label = x_label
        self.x_units = x_units
        self.y_label = y_label
        self.y_units = y_units

    # noinspection PyPep8Naming
    def report_shot(self, shot_num, datamodel):
        data_dict = datamodel.data_dict
        shot_key = f'shot-{shot_num:d}'
        fit_struct = (data_dict['shot_processors'][self.lor_fit_processor]
                      ['results'][shot_key]['lor_fit_struct'])

        x_data = fit_struct['input_data']
        y_data = fit_struct['output_data']
        popt = [fit_struct[key]['val'] for key in fit_struct['param_keys']]

        x_min = np.min(x_data)
        x_max = np.max(x_data)
        x_range = (1 / 2) * (x_max - x_min)
        x_center = (1 / 2) * (x_max + x_min)
        plot_min = x_center - 1.1 * x_range
        plot_max = x_center + 1.1 * x_range
        plot_x_list = np.linspace(plot_min, plot_max, 100)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_data[5:], y_data[5:], '.', markersize=10)
        ax.plot(plot_x_list, lorentzian_fit_function(plot_x_list, *popt))
        ax.set_xlabel(f'{self.x_label} ({self.x_units})')
        ax.set_ylabel(f'{self.y_label} ({self.y_units})')
        return fig


class HetDemodAllShotsReporter(AllShotsReporter):
    def __init__(self, *, reporter_name, atom_het_datafields, ref_het_datafields,
                 t_start=None, t_stop=None, reset=False):
        super(HetDemodAllShotsReporter, self).__init__(reporter_name=reporter_name, reset=reset)
        self.atom_het_datafields = atom_het_datafields
        self.ref_het_datafields = ref_het_datafields
        self.t_start = t_start
        self.t_stop = t_stop

    # noinspection PyPep8Naming
    def report_shot(self, shot_num, datamodel):
        atom_A_het = datamodel.get_data(self.atom_het_datafields[0], shot_num)
        atom_phi_het = datamodel.get_data(self.atom_het_datafields[1], shot_num)
        atom_I_het = datamodel.get_data(self.atom_het_datafields[2], shot_num)
        atom_Q_het = datamodel.get_data(self.atom_het_datafields[3], shot_num)

        ref_A_het = datamodel.get_data(self.ref_het_datafields[0], shot_num)
        ref_phi_het = datamodel.get_data(self.ref_het_datafields[1], shot_num)
        ref_I_het = datamodel.get_data(self.ref_het_datafields[2], shot_num)
        ref_Q_het = datamodel.get_data(self.ref_het_datafields[3], shot_num)

        dt = atom_A_het.attrs['dx']
        t = np.arange(0, len(atom_A_het) * dt, dt)
        if self.t_start is None:
            self.t_start = t[0]
        if self.t_stop is None:
            self.t_stop = t[-1]
        mask = np.logical_and(self.t_start < t, t < self.t_stop)
        t = t[mask] * 1e3

        fig = plt.figure(figsize=(12, 12))

        ax_A_het = fig.add_subplot(2, 2, 1)
        atom_A_het = atom_A_het[mask]
        ref_A_het = ref_A_het[mask]
        ax_A_het.plot(t, atom_A_het)
        ax_A_het.plot(t, ref_A_het)
        ax_A_het.set_title('Amplitude')

        ax_phi_het = fig.add_subplot(2, 2, 3)
        atom_phi_het = atom_phi_het[mask]
        ref_phi_het = ref_phi_het[mask]
        # Center central data point of phi_het between 0 and 2 * np.pi
        atom_phi_het = np.unwrap(atom_phi_het)
        ref_phi_het = np.unwrap(ref_phi_het)
        num_samples = len(atom_phi_het)
        atom_phi_het = atom_phi_het - ((atom_phi_het[int(num_samples / 2)] // (2 * np.pi)) * 2 * np.pi)
        ref_phi_het = ref_phi_het - ((ref_phi_het[int(num_samples / 2)] // (2 * np.pi)) * 2 * np.pi)

        ax_phi_het.plot(t, np.unwrap(atom_phi_het) / np.pi)
        ax_phi_het.plot(t, np.unwrap(ref_phi_het) / np.pi)
        ax_phi_het.set_title(r'Phase ($\pi$)')
        ax_phi_het.set_xlabel('Time (ms)')

        ax_I_het = fig.add_subplot(2, 2, 4)
        atom_I_het = atom_I_het[mask]
        ref_I_het = ref_I_het[mask]
        ax_I_het.plot(t, atom_I_het)
        ax_I_het.plot(t, ref_I_het)
        ax_I_het.legend(['With Atoms', 'No Atoms'])
        ax_I_het.set_title('I-Quadrature')
        ax_I_het.set_xlabel('Time (ms)')

        ax_Q_het = fig.add_subplot(2, 2, 2)
        atom_Q_het = atom_Q_het[mask]
        ref_Q_het = ref_Q_het[mask]
        ax_Q_het.plot(t, atom_Q_het)
        ax_Q_het.plot(t, ref_Q_het)
        ax_Q_het.set_title('Q-Quadrature')

        return fig


class HetDemodSingleShotReporter(Reporter):
    def __init__(self, *, reporter_name, atom_het_demod_processor, ref_het_demod_processor, shot_num,
                 t_start=None, t_stop=None):
        super(HetDemodSingleShotReporter, self).__init__(reporter_name=reporter_name)
        self.atom_het_demod_processor = atom_het_demod_processor
        self.ref_het_demod_processor = ref_het_demod_processor
        self.shot_num = shot_num
        self.t_start = t_start
        self.t_stop = t_stop

    # noinspection PyPep8Naming
    def report(self, datamodel):
        data_dict = datamodel.data_dict
        run_name = data_dict['run_name']

        shot_key = f'shot-{self.shot_num:d}'
        atom_h5file = (data_dict['shot_processors'][self.atom_het_demod_processor]
                       ['results'][shot_key]['result_file_path'])
        ref_h5file = (data_dict['shot_processors'][self.ref_het_demod_processor]
                      ['results'][shot_key]['result_file_path'])
        with h5py.File(atom_h5file, 'r') as atom_file:
            with h5py.File(ref_h5file, 'r') as ref_file:
                t = atom_file['time_series']
                if self.t_start is None:
                    self.t_start = t[0]
                if self.t_stop is None:
                    self.t_stop = t[-1]
                mask = np.logical_and(self.t_start < t, t < self.t_stop)
                t = t[mask] * 1e3

                fig = plt.figure(figsize=(12, 12))

                ax_A_het = fig.add_subplot(2, 2, 1)
                atom_A_het = (atom_file['A_het'][mask])
                ref_A_het = (ref_file['A_het'][mask])
                ax_A_het.plot(t, atom_A_het)
                ax_A_het.plot(t, ref_A_het)
                ax_A_het.set_title('Amplitude')

                ax_phi_het = fig.add_subplot(2, 2, 3)
                atom_phi_het = (atom_file['phi_het'][mask])
                ref_phi_het = (ref_file['phi_het'][mask])
                ax_phi_het.plot(t, np.unwrap(atom_phi_het) / np.pi)
                ax_phi_het.plot(t, np.unwrap(ref_phi_het) / np.pi)
                ax_phi_het.set_title(r'Phase ($\pi$)')
                ax_phi_het.set_xlabel('Time (ms)')

                ax_Q_het = fig.add_subplot(2, 2, 2)
                atom_Q_het = (atom_file['Q_het'][mask])
                ref_Q_het = (ref_file['Q_het'][mask])
                ax_Q_het.plot(t, atom_Q_het)
                ax_Q_het.plot(t, ref_Q_het)
                ax_Q_het.set_title('Q-Quadrature')

                ax_I_het = fig.add_subplot(2, 2, 4)
                atom_I_het = (atom_file['I_het'][mask])
                ref_I_het = (ref_file['I_het'][mask])
                ax_I_het.plot(t, atom_I_het)
                ax_I_het.plot(t, ref_I_het)
                ax_I_het.legend(['With Atoms', 'No Atoms'])
                ax_I_het.set_title('I-Quadrature')
                ax_I_het.set_xlabel('Time (ms)')

                figure_title = f'{self.reporter_name} - {run_name} - {shot_key}'
                fig.suptitle(figure_title, fontsize=16)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
