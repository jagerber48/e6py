import numpy as np
from PyQt5.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel, QSpacerItem, QSizePolicy
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major')


class GaussianIntegrateAxisPlot(pg.PlotItem):
    def __init__(self, *args, **kwargs):
        super(GaussianIntegrateAxisPlot, self).__init__(*args, **kwargs)
        self.setMouseEnabled(x=False, y=False)

    def update(self, data_img, model_img, slice_axis, sx, sy):
        if slice_axis == 0:
            y_range = data_img.shape[1]
            integrate_axis = 1
            data_cut_data = np.sum(data_img, axis=integrate_axis) / np.sqrt(2 * np.pi * sy ** 2)
            model_cut_data = np.sum(model_img, axis=integrate_axis) / np.sqrt(2 * np.pi * sy ** 2)
            self.invertY(True)
            self.plot(range(y_range), data_cut_data,
                      pen=pg.mkPen(width=0.5), symbolBrush='w', symbolSize=5)
            self.plot(range(y_range), model_cut_data, pen=pg.mkPen('g'))

        else:
            x_range = data_img.shape[0]
            integrate_axis = 0
            data_cut_data = np.sum(data_img, axis=integrate_axis) / np.sqrt(2 * np.pi * sx ** 2)
            model_cut_data = np.sum(model_img, axis=integrate_axis) / np.sqrt(2 * np.pi * sx ** 2)
            self.plot(model_cut_data, range(x_range),
                      pen=pg.mkPen(width=0.5), symbolBrush='w', symbolSize=5)
            self.plot(data_cut_data, range(x_range), pen=pg.mkPen('y'))


class Gaussian2DPlot(pg.PlotItem):
    def __init__(self, *args, **kwargs):
        super(Gaussian2DPlot, self).__init__(*args, **kwargs)
        self.img = None
        self.x0 = 0
        self.y0 = 0
        self.angle = 0

        self.image_item = pg.ImageItem()
        self.addItem(self.image_item)

        self.setAspectLocked(True, 1)

        self.v_line = pg.InfiniteLine(0, angle=90, movable=False, pen='g')
        self.addItem(self.v_line)

        self.h_line = pg.InfiniteLine(0, angle=0, movable=False, pen='y')
        self.addItem(self.h_line)

        self.x_line = pg.InfiniteLine((0, 0), angle=0, movable=False, pen='r')
        self.addItem(self.x_line)

        self.y_line = pg.InfiniteLine((0, 0), angle=90, movable=False, pen='b')
        self.addItem(self.y_line)

    def update(self, img, x0, y0, angle):
        self.img = img
        self.x0 = x0
        self.y0 = y0
        self.angle = angle

        self.image_item.setImage(self.img)
        self.v_line.setPos((self.y0, self.x0))
        self.h_line.setPos((self.y0, self.x0))

        self.x_line.setPos((self.y0, self.x0))
        self.x_line.setAngle(self.angle)

        self.y_line.setPos((self.y0, self.x0))
        self.y_line.setAngle(self.angle + 90)


class FitVisualizationWindow(QWidget):
    def __init__(self, fit_struct, parent=None):
        super(FitVisualizationWindow, self).__init__(parent=parent)

        self.fit_struct = fit_struct
        self.data_img = self.fit_struct['data_img']
        self.model_img = self.fit_struct['model_img']
        self.x_range = self.data_img.shape[0]
        self.y_range = self.data_img.shape[1]
        self.x0 = self.fit_struct['x0']['val']
        self.y0 = self.fit_struct['y0']['val']
        self.sx = self.fit_struct['sx']['val']
        self.sy = self.fit_struct['sy']['val']
        self.angle = self.fit_struct['angle']['val']
        self.img_min = np.min([self.data_img.min(), self.model_img.min()])
        self.img_max = np.max([self.data_img.max(), self.model_img.max()])

        self.data_imageitem = pg.ImageItem(image=self.data_img)
        self.model_imageitem = pg.ImageItem(image=self.model_img)

        self.resize(800, 800)

        self.gridlayout = QGridLayout(self)
        self.graphicslayout = pg.GraphicsLayoutWidget(parent=self)
        self.graphicslayout.show()
        self.textlayout = QVBoxLayout(self)
        self.gridlayout.addWidget(self.graphicslayout, 0, 0, 1, 1)
        self.gridlayout.addLayout(self.textlayout, 0, 1, 1, 1)

        self.textlayout.addItem(QSpacerItem(14, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        for key in self.fit_struct['param_keys']:
            label = QLabel()
            val = fit_struct[key]['val']
            label.setText(f'{str(key)} = {val:.2f}')
            self.textlayout.addWidget(label)
        self.textlayout.addItem(QSpacerItem(14, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.data_plot = Gaussian2DPlot()
        self.graphicslayout.addItem(self.data_plot, row=0, col=0, rowspan=1, colspan=1)

        self.model_plot = Gaussian2DPlot()
        self.graphicslayout.addItem(self.model_plot, row=1, col=1, rowspan=1, colspan=1)
        self.model_plot.setXLink(self.data_plot.getViewBox())
        self.model_plot.setYLink(self.data_plot.getViewBox())

        self.horizontal_cut_plot = GaussianIntegrateAxisPlot()
        self.graphicslayout.addItem(self.horizontal_cut_plot, row=1, col=0, rowspan=1, colspan=1)


        self.vertical_cut_plot = GaussianIntegrateAxisPlot
        self.graphicslayout.addItem(self.vertical_cut_plot, row=0, col=1, rowspan=1, colspan=1)

        self.data_plot.update(self.data_img, self.x0, self.y0, self.angle)
        self.model_plot.update(self.model_img, self.x0, self.y0, self.angle)


        # data_plot = self.graphicslayout.addPlot(row=0, col=0, rowspan=1, colspan=1)
        # data_plot.setMouseEnabled(x=False, y=False)
        # data_plot.setAspectLocked(True, 1)
        # data_plot.addItem(self.data_imageitem)
        # data_v_line = pg.InfiniteLine(angle=90, movable=False)
        # data_h_line = pg.InfiniteLine(angle=0, movable=False)
        # data_plot.addItem(data_v_line, ignoreBounds=True)
        # data_plot.addItem(data_h_line, ignoreBounds=True)
        # data_v_line.setPos(self.y0)
        # data_h_line.setPos(self.x0)
        # data_x_line = pg.InfiniteLine((self.y0,self.x0), angle=self.fit_struct['angle']['val'], movable=False, pen='r')
        # data_y_line = pg.InfiniteLine((self.y0,self.x0), angle=self.fit_struct['angle']['val']+90, movable=False, pen='b')
        # data_plot.addItem(data_x_line, ignoreBounds=True)
        # data_plot.addItem(data_y_line, ignoreBounds=True)

        # model_plot = self.graphicslayout.addPlot(row=1, col=1, rowspan=1, colspan=1)
        # # model_plot.setMouseEnabled(x=False, y=False)
        # # model_plot.setAspectLocked(True)
        # model_plot.setXLink(data_plot.getViewBox())
        # model_plot.setYLink(data_plot.getViewBox())
        # model_plot.addItem(self.model_imageitem)
        # model_v_line = pg.InfiniteLine(angle=90, movable=False)
        # model_h_line = pg.InfiniteLine(angle=0, movable=False)
        #
        # model_plot.addItem(model_v_line, ignoreBounds=True)
        # model_plot.addItem(model_h_line, ignoreBounds=True)
        # model_v_line.setPos(self.y0)
        # model_h_line.setPos(self.x0)
        # model_x_line = pg.InfiniteLine((self.y0,self.x0), angle=self.fit_struct['angle']['val'], movable=False, pen='r')
        # model_y_line = pg.InfiniteLine((self.y0,self.x0), angle=self.fit_struct['angle']['val']+90, movable=False, pen='b')
        # model_plot.addItem(model_x_line, ignoreBounds=True)
        # model_plot.addItem(model_y_line, ignoreBounds=True)
        # self.ellipse = pg.EllipseROI((self.y0 - self.sy / 2, self.x0 - self.sx / 2), (self.sx, self.sy), movable=False, rotatable=False, resizable=False)
        # self.ellipse.setTransformOriginPoint(QPointF(self.sy / 2, self.sx / 2))
        # # self.ellipse.setRotation(self.fit_struct['angle']['val'])
        # model_plot.addItem(self.ellipse, ignoreBounds=True)
        # # model_plot.hideButtons()

        horizontal_plot = self.graphicslayout.addPlot(row=1, col=0, rowspan=1, colspan=1)
        horizontal_plot.setMouseEnabled(x=False, y=False)
        horizontal_plot.setXLink(self.data_plot.getViewBox())
        horizontal_plot.invertY(True)
        horizontal_plot.plot(range(self.y_range), self.horizontal_int_cut_data,
                             pen=pg.mkPen(width=0.5), symbolBrush='w', symbolSize=5)
        horizontal_plot.plot(range(self.y_range), self.horizontal_int_cut_model, pen=pg.mkPen('r'))
        # horizontal_plot.hideButtons()

        vertical_plot = self.graphicslayout.addPlot(row=0, col=1, rowspan=1, colspan=1)
        vertical_plot.setMouseEnabled(x=False, y=False)
        vertical_plot.setYLink(self.data_plot.getViewBox())
        vertical_plot.plot(self.vertical_int_cut_data, range(self.x_range),
                           pen=pg.mkPen(width=0.5), symbolBrush='w', symbolSize=5)
        vertical_plot.plot(self.vertical_int_cut_model, range(self.x_range), pen=pg.mkPen('b'))
        # vertical_plot.hideButtons()

        self.data_plot.autoRange()

        self.show()

