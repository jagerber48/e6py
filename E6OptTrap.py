import numpy as np
import scipy.constants as const
from functools import reduce
from . import E6utils
from .Atoms import Rb87_Atom

hbar = const.hbar
c = const.c
ep0 = const.epsilon_0


class Beam:
    # Class to model an optical gaussian beam
    def __init__(self, waist_x, power, wavelength, waist_y=None, z0_x=0, z0_y=0):
        self.waist_x = waist_x
        if waist_y is None:
            waist_y = waist_x
        self.waist_y = waist_y
        self.power = power
        self.I0 = self.power_to_max_intensity(self.power, self.waist_x, self.waist_y)
        self.E0 = E6utils.intensity_to_e_field(self.I0)
        self.wavelength = wavelength
        self.z0_x = z0_x
        self.z0_y = z0_y
        self.zr_x = self.w0_to_zr(self.waist_x, self.wavelength)
        self.zr_y = self.w0_to_zr(self.waist_y, self.wavelength)
        self.trans_list = []
        # self.trans_list is a list of transformations (translations and rotations)
        # to apply to beam geometry. List of functions which takes 3d vectors as inputs and output 3d vectors
        self.field = None

    def beam_field_profile(self, x, y, z):
        # Electric field amplitude as a function of x, y, z
        # Transform [x, y, z] according to transformations listed in self.trans_list
        [x, y, z] = reduce(lambda vec, f: f(vec), self.trans_list[::-1], [x, y, z])
        return self.gaussian_field_profile(x, y, z,
                                           self.E0, wavelength=self.wavelength,
                                           w0_x=self.waist_x, w0_y=self.waist_y,
                                           z0_x=self.z0_x, z0_y=self.z0_y)

    def beam_intensity_profile(self, x, y, z):
        # intensity as a function of x, y, z
        # Transform [x, y, z] according to transformations listed in self.trans_list
        [x, y, z] = reduce(lambda vec, f: f(vec), self.trans_list[::-1], [x, y, z])
        return self.gaussian_intensity_profile(x, y, z,
                                               self.I0, wavelength=self.wavelength,
                                               w0_x=self.waist_x, w0_y=self.waist_y,
                                               z0_x=self.z0_x, z0_y=self.z0_y)

    def make_e_field(self, x0=(-1, -1, -1), xf=(1, 1, 1), n_steps=(10, 10, 10)):
        """ Make xarray of the electric field amplitude
        :param x0: Tuple of (x0, y0, z0) lower range values
        :param xf: Tuple of (xf, yf, zf) upper range values
        :param n_steps: Tuple of (nx, ny, nz) number of steps
        :return: xarray of spatial field profile
        """
        self.field = E6utils.func3d_xr(self.beam_field_profile, x0, xf, n_steps)
        return self.field

    def make_intensity_field(self, x0=(-1, -1, -1), xf=(1, 1, 1), n_steps=(10, 10, 10)):
        """ Make xarray of the optical intensity
        :param x0: Tuple of (x0, y0, z0) lower range values
        :param xf: Tuple of (xf, yf, zf) upper range values
        :param n_steps: Tuple of (nx, ny, nz) number of steps
        :return: xarray of spatial field profile
        """
        self.field = E6utils.func3d_xr(self.beam_intensity_profile, x0, xf, n_steps)
        return self.field

    def translate(self, trans_vec):
        # Add translation transformation to self.trans_list
        self.trans_list.append(lambda v: E6utils.translate_vec(v, -trans_vec))
        return self

    def transform(self, trans_mat):
        # Add matrix transformation to self.trans_list
        self.trans_list.append(lambda v: E6utils.transform_vec(v, trans_mat))
        return self

    def rotate(self, axis=(1, 0, 0), angle=0.0):
        # Add rotation transformation to self.trans_list
        rot_mat = E6utils.rot_mat(axis, -angle)
        self.transform(rot_mat)
        return self

    @staticmethod
    def waist_profile(waist, wavelength, z):
        zr = Beam.w0_to_zr(waist, wavelength)
        return waist * np.sqrt(1 + (z / zr) ** 2)

    @staticmethod
    def w0_to_zr(w0, wavelength):
        return np.pi * w0 ** 2 / wavelength

    @staticmethod
    def power_to_max_intensity(P, w0_x, w0_y=None):
        if w0_y is None:
            w0_y = w0_x
        return 2 * P / (np.pi * w0_x * w0_y)

    @staticmethod
    def gaussian_field_profile(x, y, z, E0, w0_x, z0_x, wavelength, w0_y=None, z0_y=None):
        if w0_y is None:
            w0_y = w0_x
        if z0_y is None:
            z0_y = z0_x
        # Note scaling on sx and sy to convert between Gaussian variance and Gaussian beam waist
        return E0 * (np.sqrt(w0_x / Beam.waist_profile(w0_x, wavelength, z - z0_x))
                     * np.sqrt(w0_y / Beam.waist_profile(w0_y, wavelength, z - z0_y))
                     * E6utils.gaussian_2d(x, y, x0=0, y0=0, sx=w0_x / np.sqrt(2), sy=w0_y / np.sqrt(2)))

    @staticmethod
    def gaussian_intensity_profile(x, y, z, I0, w0_x, z0_x, wavelength, w0_y=None, z0_y=None):
        if w0_y is None:
            w0_y = w0_x
        if z0_y is None:
            z0_y = z0_x
        # Note scaling on sx and sy to convert between Gaussian variance and Gaussian beam waist
        return I0 * ((w0_x / Beam.waist_profile(w0_x, wavelength, z - z0_x))
                     * (w0_y / Beam.waist_profile(w0_y, wavelength, z - z0_y))
                     * E6utils.gaussian_2d(x, y, x0=0, y0=0, sx=w0_x / 2, sy=w0_y / 2))


class OptTrap:
    def __init__(self, beams, atom=Rb87_Atom, quiet=False):
        try:
            len(beams)
        except TypeError:
            beams = (beams,)
        self.beams = beams
        self.atom = atom
        self.trap_freqs = [0, 0, 0]
        self.trap_freq_geom_mean = 0
        self.trap_depth = 0
        self.pot_field = None
        self.get_trap_params()
        self.quiet = quiet
        if not self.quiet:
            print('Created optical trap with the following properties:')
            self.print_properties()

    def get_trap_params(self):
        """ Calculate trap frequencies, geometric mean of trap frequencies, and trap depth

        Calculation accomplished by numerically simulating the optical potential in the vicinity of the origin.
        Trap depth is found by determining the minimum of the optical potential
        Trap frequency is found by numerically calculating the hessian matrix (array of 2nd derivatives) of the
        optical potential and diagonalizing to extract principle components.
        """
        tot_pot_field = self.make_pot_field(x0=(-1e-7,) * 3, xf=(1e-7,) * 3, n_steps=(10,) * 3)
        self.trap_depth = -tot_pot_field.values.min()
        hess = E6utils.hessian(tot_pot_field, x0=0, y0=0, z0=0)
        vals = np.linalg.eig(hess)[0]  # extract only eigenvalues, ignore eigenvectors
        self.trap_freqs = np.sqrt(vals / self.atom.mass)
        self.trap_freq_geom_mean = np.prod(self.trap_freqs) ** (1 / 3)
        return self.trap_depth, self.trap_freqs

    def make_pot_field(self, x0=(-1e-6,) * 3, xf=(1e-6,) * 3, n_steps=(10,) * 3):
        x0 = E6utils.single_to_triple(x0)
        xf = E6utils.single_to_triple(xf)
        n_steps = E6utils.single_to_triple(n_steps)
        tot_pot_field = E6utils.template_xr(0, x0, xf, n_steps)
        for i in range(0, len(self.beams)):
            beam = self.beams[i]
            beam_opt_field = beam.make_e_field(x0, xf, n_steps)
            beam_pot_field = beam_opt_field.pipe(lambda x: self.atom.optical_potential(x, beam.wavelength))
            tot_pot_field = tot_pot_field + beam_pot_field
        self.pot_field = tot_pot_field
        return self.pot_field

    def calc_psd(self, N, T, quiet=None):
        # Calculate phase space density given atom number and temperature using trap parameters
        if quiet is None:
            quiet = self.quiet
        psd = N * ((const.hbar * self.trap_freq_geom_mean) / (const.k * T)) ** 3
        if not quiet:
            print(f'PSD = {psd:.1e}')
        return psd

    def calc_peak_density(self, N, T, quiet=None):
        # Calculate peak density from phase space density
        if quiet is None:
            quiet = self.quiet
        lambda_db = const.h / np.sqrt(2 * np.pi * self.atom.mass * const.k * T)
        n0 = self.calc_psd(N, T, quiet=True) / (lambda_db ** 3)
        if not quiet:
            print(f'Peak Density = {n0*(1e-2 ** 3):.1e} cm^-3')
        return n0

    def calc_cloud_sizes(self, T, quiet=None):
        sigma_list = [None, None, None]
        for i in range(3):
            sigma_list[i] = np.sqrt(const.k*T / (self.atom.mass * self.trap_freqs[i]**2))
        if quiet is None:
            quiet = self.quiet
        if not quiet:
            print(f'Cloud size (sx, sy, sz) = (' + ', '.join([f'{sigma*1e6:.2f}' for sigma in sigma_list]) + ') \\mu m')
        return sigma_list

    def print_properties(self):
        print(f'Trap Depth = {(self.trap_depth/const.h)*1e-6:.2f} MHz = {(self.trap_depth/const.k)*1e6:.2f} \\mu K')
        print('Trap Frequencies (wx, wy, wz) = ('
              + ', '.join([f'{self.trap_freqs[i]/(2*np.pi):.2f}' for i in range(3)]) + ') Hz'
              )
        print(f'Geometric Mean = {self.trap_freq_geom_mean / (2 * np.pi):.2f} Hz')
        return


def make_grav_pot(mass=Rb87_Atom.mass, grav_vec=(0, 0, -1),
                  x0=(-1,)*3, xf=(1,)*3, n_steps=(10,)*3):
    def grav_func(x, y, z):
        grav_comp = np.dot([x, y, z], grav_vec)
        return -mass*const.g*grav_comp
    grav_pot = E6utils.func3d_xr(grav_func, x0, xf, n_steps)
    return grav_pot


def make_sphere_quad_pot(gf=-(1/2), mf=-1, B_grad=1, units='T/m', trans_list=None,
                         strong_axis=(0, 0, 1),
                         x0=(-1,)*3, xf=(1,)*3, n_steps=(10,)*3):
    if units == 'T/m':
        pass
    elif units == 'G/cm':
        B_grad = B_grad*1e-4*1e2  # Convert G/cm into T/m
    else:
        print('unrecognized units, only T/m and G/cm supported')
    gyromagnetic_ratio_classical = 2*np.pi*1.4*1e6*1e4
    # Convert 1.4 MHz/Gauss into 1.4e10 Hz/T
    mat = E6utils.matrix_rot_v_onto_w((0, 0, 1), strong_axis)
    if trans_list is None:
        trans_list = []
    if not (np.array([strong_axis]) == np.array([0, 0, 1])).all():
        trans_list = [lambda vec: E6utils.transform_vec(vec, mat)] + trans_list

    def sphere_quad_func(x, y, z):
        if trans_list is not []:
            [x, y, z] = reduce(lambda vec, f: f(vec), trans_list[::-1], [x, y, z])
        return gf * mf * gyromagnetic_ratio_classical * const.hbar\
                  * np.sqrt((0.5*B_grad*x)**2 + (0.5*B_grad*y)**2 + (B_grad*z)**2)
    sphere_quad_pot = E6utils.func3d_xr(sphere_quad_func, x0, xf, n_steps)
    return sphere_quad_pot


def ODT_params_analytic(atom, power, waist, wavelength, quiet=False):
    # Analytic calculation of trap depth and frequencies for simple Gaussian beam running wave ODT
    # Sanity check for more complicated functionality in OptTrap class.
    I0 = Beam.power_to_max_intensity(power, waist)
    zr = Beam.w0_to_zr(waist, wavelength)
    trap_depth = np.abs(atom.optical_potential_from_intensity(I0, wavelength))
    omega_radial = np.sqrt(4*trap_depth/(atom.mass*waist**2))
    omega_axial = np.sqrt(2*trap_depth/(atom.mass*zr**2))
    trap_freqs = np.array([omega_radial, omega_radial, omega_axial])
    omega_geom_mean = np.prod(trap_freqs) ** (1 / 3)
    if not quiet:
        print(f'Trap Depth = {(trap_depth / const.h) * 1e-6:.2f} MHz = {(trap_depth / const.k) * 1e6:.2f} \\mu K')
        print('Trap Frequencies = \n'
              + '\n'.join([f'{trap_freqs[i] / (2 * np.pi):.2f} Hz' for i in range(3)])
              )
        print(f'Geometric Mean = {omega_geom_mean / (2 * np.pi):.2f} Hz')
    return trap_depth, omega_radial, omega_axial


def cloud_size_single_beam(T, atom, power, waist, wavelength, quiet=False):
    V0, omega_radial, omega_axial = ODT_params_analytic(atom, power, waist, wavelength, quiet=True)
    zr = Beam.w0_to_zr(waist, wavelength)
    sigma_radial = np.sqrt(waist ** 2 * (const.k * T) / (4 * V0))
    sigma_axial = np.sqrt(zr ** 2 * (const.k * T) / (2 * V0))
    if not quiet:
        print(f'Cloud size (sx, sy, sz) = '
              f'({sigma_radial*1e6:.2f}, {sigma_radial*1e6:.2f}, {sigma_axial*1e6:.2f}) \\mu m')
    return sigma_radial, sigma_axial


def calc_cloud_sizes(self, T, quiet=None):
    sigma_list = [None, None, None]
    for i in range(3):
        sigma_list[i] = np.sqrt(const.k * T / (self.atom.mass * self.trap_freqs[i]**2))
    if quiet is None:
        quiet = self.quiet
    if not quiet:
        print(f'Cloud size (sx, sy, sz) = (' + ', '.join([f'{sigma*1e6:.2f}' for sigma in sigma_list]) + ') \\mu m')
    return sigma_list
