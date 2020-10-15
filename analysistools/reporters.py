import numpy as np
import matplotlib.pyplot as plt
from .datamodel import Reporter, dataset_from_keychain


class CountsReporter(Reporter):
    def __init__(self, y_axis_keychains, reporter_name='counts_reporter'):
        super(CountsReporter, self).__init__(reporter_name)
        self.y_axis_keychains = y_axis_keychains

    def report(self, datamodel):
        data_dict = datamodel.data_dict
        run_name = data_dict['run_name']
        num_points = data_dict['num_points']

        y_data_list = [list(dataset_from_keychain(datamodel, keychain).values()) for keychain in self.y_axis_keychains]

        for point in range(num_points):
            point_key = f'point-{point:d}'
            fig = plt.figure(figsize=(8, 8))

            ax_loop = fig.add_subplot(2, 1, 1)
            ax_loop.set_xlabel('Loop Number')
            ax_loop.set_ylabel('Counts')

            ax_hist = fig.add_subplot(2, 1, 2)
            ax_hist.set_xlabel('Counts')
            ax_hist.set_ylabel('Frequency')

            for y_data in y_data_list:
                ax_loop.plot(y_data, '.', markersize=10)
                ax_hist.hist(y_data, alpha=0.5)

            figure_title = f'{self.reporter_name} - {run_name} - {point_key}'
            fig.suptitle(figure_title, fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])


class AvgRndmImgReporter(Reporter):
    def __init__(self, atom_rndm_keychain, atom_avg_keychain, ref_rndm_keychain, ref_avg_keychain,
                 reporter_name='avg_rndm_img_reporter'):
        super(AvgRndmImgReporter, self).__init__(reporter_name)
        self.atom_rndm_keychain = atom_rndm_keychain
        self.atom_avg_keychain = atom_avg_keychain
        self.ref_rndm_keychain = ref_rndm_keychain
        self.ref_avg_keychain = ref_avg_keychain

    def report(self, datamodel):
        data_dict = datamodel.data_dict
        run_name = data_dict['run_name']
        num_points = data_dict['num_points']

        atom_rndm_img = dataset_from_keychain(datamodel, self.atom_rndm_keychain)
        atom_avg_img = dataset_from_keychain(datamodel, self.atom_avg_keychain)
        ref_rndm_img = dataset_from_keychain(datamodel, self.ref_rndm_keychain)
        ref_avg_img = dataset_from_keychain(datamodel, self.ref_avg_keychain)
        img_list = [atom_rndm_img, atom_avg_img, ref_rndm_img, ref_avg_img]

        min_list = [np.nanmin(img) for img in img_list]
        min_val = np.nanmin(min_list)
        max_list = [np.nanmax(img) for img in img_list]
        max_val = np.nanmax(max_list)

        cmap = 'binary_r'

        for point in range(num_points):
            point_key = f'point-{point:d}'
            fig = plt.figure(figsize=(8, 8))

            ax_atom_rndm = fig.add_subplot(2, 2, 1)
            im = ax_atom_rndm.imshow(atom_rndm_img, vmin=min_val, vmax=max_val, cmap=cmap)
            fig.colorbar(im, ax=ax_atom_rndm)
            ax_atom_rndm.set_title(f'{run_name}:  Single Atom Shot - Point {point} - Shot #{shot_num}')

