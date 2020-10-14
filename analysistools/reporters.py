import matplotlib.pyplot as plt
from .datamodel import Reporter


class PlotReporter(Reporter):
    def __init__(self, reporter_name, x_axis_keychain, y_axis_keychains, x_label, y_label):
        super(PlotReporter, self).__init__(reporter_name)
        self.x_axis_keychain = x_axis_keychain
        if not isinstance(y_axis_keychains, (list, tuple)):
            y_axis_keychains = [y_axis_keychains]
        self.y_axis_keychain_list = y_axis_keychains
        self.reporter_name = reporter_name
        self.x_label = x_label
        self.y_label = y_label

    def report_run(self, data_dict, run_name=None):
        figure_title = self.reporter_name
        if run_name is not None:
            figure_title += f' - {run_name}'
        num_points = data_dict['num_points']
        for point in range(num_points):
            point_figure_title = f'{figure_title} - point {point:d}'
            x_data, y_data_list = self.get_xy_data(data_dict, point)
            self.report(x_data, y_data_list, point_figure_title)

    def report(self, x_data, y_data, point_figure_title):
        raise NotImplementedError

    def get_xy_data(self, data_dict, point):
        if self.x_axis_keychain is not None:
            x_data = dataset_from_keychain(data_dict, self.x_axis_keychain)
        else:
            x_data = None
        point_key = f'point-{point:d}'
        y_data_list = []
        for keychain in self.y_axis_keychain_list:
            point_keychain = f'{keychain}/{point_key}'
            y_data = dataset_from_keychain(data_dict, point_keychain)
            y_data_list.append(y_data)
        return x_data, y_data_list

class HistogramReporter(Reporter):
    def __init__(self, x_axis_keychain, y_axis_keychains, reporter_name, x_label, y_label):
        super(HistogramReporter, self).__init__(x_axis_keychain, y_axis_keychains, reporter_name, x_label, y_label)

    def report(self, x_data, y_data_list, figure_title=None):
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

        if figure_title is None:
            figure_title = self.reporter_name
        fig.suptitle(figure_title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])