import numpy as np
import scipy.constants as const

c = const.c


class OpticalCavity:
    def __init__(self, roc_1, roc_2, cav_length, t1, t2, l1, l2):
        self.roc_1 = roc_1
        self.roc_2 = roc_2
        self.cav_length = cav_length
        self.fsr = c / (2 * self.cav_length)
        self.g1 = 1 - self.cav_length /self.roc_1
        self.g2 = 1 - self.cav_length /self.roc_2
        self.
        self.dist_to_concentric = self.roc_1 + self.roc_2 - self.cav_length
        self.dist_to_planar = self.cav_length
        self.z0 = self.calc_z0()
        self.zr = self.calc_zr()

    def calc_z0(self):
        numerator = self.cav_length * (self.cav_length - self.roc_2)
        denominator = (self.cav_length - self.roc_1) + (self.cav_length - self.roc_2)
        z0 = numerator /denominator
        return z0

    def calc_zr(self):
        zr = np.sqrt(self.z0 * (self.roc_1 - self.z0))
        return zr

    def calc_transverse_mode_spacing(self):



class CavityMode:
    def __init__(self, cavity, wavelength):
        self.cavity = cavity
        self.wavelength = wavelength
        self.z0 = self.cavity.z0
        self.zr = self.cavity.zr
        self.w0 = np.sqrt(self.zr * self.wavelength /np.pi)
        self.wmir_1 = self.calc_wz(0)
        self.wmir_2 = self.calc_wz(self.cavity.cav_lenth)
        self.mode_volume = np.pi/4 * self.w0**2 * self.cavity.cav_length

    def calc_w0(self, wavelength):
        w0 = np.sqrt(self.zr * wavelength /np.pi)
        return w0

    def calc_wz(self, z):
        wz = self.w0 * np.sqrt(1 + ((z - self.z0 ) /self.zr )**2)
        return wz
