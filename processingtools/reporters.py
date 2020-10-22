import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .datamodel import dataset_from_keychain


class Reporter:
    def __init__(self, reporter_name):
        self.reporter_name = reporter_name

    def report(self, datamodel):
        raise NotImplementedError


class AtomRefCountsReporter(Reporter):
    def __init__(self, atom_counts_processor_name, ref_counts_processor_name, reporter_name='counts_reporter'):
        super(AtomRefCountsReporter, self).__init__(reporter_name)
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
                atom_counts = data_dict['shot_processors'][self.atom_counts_processor_name]['results'][f'shot-{shot}']['counts']
                ref_counts = data_dict['shot_processors'][self.ref_counts_processor_name]['results'][f'shot-{shot}']['counts']
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
        super(AvgRndmImgReporter, self).__init__(reporter_name)
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
