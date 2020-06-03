import numpy as np
import scipy.constants as const
import E6OptTrap

c = const.c


class Mirror:
    def __init__(self, roc, transmission_ppm, losses_ppm):
        self.roc = roc
        self.transmission = transmission_ppm*1e-6
        self.loss = losses_ppm*1e-6


class OpticalCavity:
    def __init__(self, mirror_1, mirror_2, length):
        self.mirror_1 = mirror_1
        self.mirror_2 = mirror_2

        self.roc_1 = mirror_1.roc
        self.roc_2 = mirror_2.roc
        self.length = length
        self.g_1 = self.degeneracy_param(self.roc_1, length)
        self.g_2 = self.degeneracy_param(self.roc_2, length)
        self.dist_to_conc = self.roc_1 + self.roc_2 - self.length

        self.T_1 = mirror_1.transmission
        self.T_2 = mirror_2.transmission
        self.L_1 = mirror_1.loss
        self.L_2 = mirror_2.loss
        self.total_losses = self.T_1 + self.T_2 + self.L_1 + self.L_2
        self.finesse = self.calc_finesse(self.total_losses)
        self.input_efficiency = self.T_1 / self.total_losses
        self.output_efficiency = self.T_2 / self.total_losses

    # def input_power_to_peak_field(self, P):

    @staticmethod
    def degeneracy_param(R, L):
        return 1 - 2 * L / R

    @staticmethod
    def calc_finesse(total_losses):
        return 2 * np.pi / total_losses


class CavityMode:
    def __init__(self, cavity, wavelength):
        self.cavity = cavity
        self.wavelength = wavelength

        g1 = self.cavity.g_1
        g2 = self.cavity.g_2
        L = self.cavity.length
        self.z1 = L * (-g2 * (1 - g1)) / (g1 + g2 - 2 * g1 * g2)  # Position of first mirror relative to beam waist
        self.z2 = L * (g1 * (1 - g2)) / (g1 + g2 - 2 * g1 * g2)  # Position of second mirror relative to beam waist
        self.zr = L * np.sqrt((g1 * g2 * (1 - g1 * g2)) / (g1 + g2 - 2 * g1 * g2)**2)  # Rayleigh Range
        self.w0 = np.sqrt(self.wavelength * L / np.pi) \
            * ((g1 * g2 * (1 - g1 * g2)) / (g1 + g2 - 2 * g1 * g2)**2)**(1/4)  # Waist size
        self.w1 = np.sqrt(self.wavelength * L / np.pi) \
            * (g2 / (g1 * (1 - g1 * g2)))**(1/4)  # Waist on first mirror
        self.w2 = np.sqrt(self.wavelength * L / np.pi) \
            * (g1 / (g2 * (1 - g1 * g2)))**(1/4)  # Waist on second mirror
        self.theta = E6OptTrap.Beam.w0_to_theta(self.w0, self.wavelength)  # Beam divergence angle
        self.NA = np.sin(self.theta)  # Beam divergence numerical aperture

        self.mode_volume = np.pi/4 * self.w0**2 * self.cavity.cav_length
