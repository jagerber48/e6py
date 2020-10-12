import matplotlib.pyplot as plt
from .datamodel import Reporter


class HistogramReporter(Reporter):
    def __init__(self, x_axis_keychain, y_axis_keychains, reporter_name, x_label, y_label):
        super(HistogramReporter, self).__init__(x_axis_keychain, y_axis_keychains, reporter_name, x_label, y_label)

    def report(self, x_data, y_data_list):
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

        fig.suptitle(self.reporter_name, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])