import matplotlib.pyplot as plt
from .datamodel import Reporter, dataset_from_keychain


class CountsReporter(Reporter):
    def __init__(self, y_axis_keychains, reporter_name='counts_reporter'):
        super(CountsReporter, self).__init__(reporter_name)
        self.reporter_name = reporter_name
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
