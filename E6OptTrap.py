import numpy as np
import scipy.constants as const
from functools import reduce
from . import E6utils
from .Atoms import Rb87_Atom

hbar = const.hbar
c = const.c
ep0 = const.epsilon_0


class Transition:

    def __init__(self, transition_data):
        self.data = transition_data
        self.name = self.data['name']
        self.Jg = self.data['Jg']
        self.Je = self.data['Je']
        self.frequency = self.data['frequency']
        self.omega_0 = 2*np.pi*self.frequency
        self.lifetime = self.data['lifetime']
        self.gamma = 1/self.lifetime
        self.dJJ = self.reduced_dipole_element()
        self.dEff = self.dJJ_to_dEff()
        self.Isat_eff = self.calc_Isat(d=self.dEff)

    def reduced_dipole_element(self):
        """ Convert transition lifetime to reduced dipole element

        dJJ is the reduced dipole element arising from the Wigner-Eckart theorem which couples fine structure
        state Jg and Je. This element can be calculated from a measured transition lifetime.
        See Steck Eq. (38)
        """
        degeneracyFactor = (2*self.Je+1)/(2*self.Jg+1)
        return np.sqrt((3*np.pi*ep0*hbar*c**3*self.gamma)/(self.omega_0**3))*np.sqrt(degeneracyFactor)

    def dJJ_to_dEff(self):
        """ Convert reduced dipole element to effect far-detuned dipole element

        Effective far detuned dipole moment for pi polarized light. This is the effective dipole moment which couples
        any hyperfine ground state to a single fine structure excited state manifold. The factor of
        1/3 arises from summing the coupling of the ground hyperfine state to each possible excited state
        hyperfine state. It can also be viewed as *averaging* the coupling of each fine structure ground state to
        the fine structure excited states over all of the possible fine structure ground states.
        See Steck Eq. (45)
        """
        return self.dJJ/np.sqrt(3)

    def calc_Isat(self, d=None):
        """ Calculate saturation intensity for a given transition matrix element d

        Saturation intensity is defined by the saturation parameter I/Isat = 2 Omega**2/gamma**2
        where Omega is the rabi frequency and gamma is the transition decay rate. Two words of caution.
        First, the excited state decay rate gamma has a fixed relationship with the reduce dipole element dJJ however
        the dipole element used to calculate Isat may not be dJJ. For example, it may be dEff or the d corresponding
        to a particular fine or hyperfine transition. The idea of Isat also makes most sense when multi-level
        effects beyond the two level approximation are not present.
        Second, the saturation intensity makes the most sense when considering the interaction between the atom
        and a *single* laser beam with fixed intensity. The story is complicated when multiple laser beams are
        involved (even if they have the same time-dependence) and it becomes more sensible to consider the electric
        field magnitude rather than field intensity.
        See Steck Eq. (50)
        """
        if d is None:
            d = self.dJJ
        return (c*ep0*hbar**2*self.gamma**2)/(4*d**2)


Rb87_D2_transition_data = {
    'name': 'D2',
    'Jg': 0.5,
    'Je': 1.5,
    'frequency': 384.2304844685e12,
    'lifetime': 26.2348e-9
}
Rb87_D1_transition_data = {
    'name': 'D1',
    'Jg': 0.5,
    'Je': 0.5,
    'frequency': 377.107463380e12,
    'lifetime': 27.679e-9
}
Rb87_D2_transition = Transition(Rb87_D2_transition_data)
Rb87_D1_transition = Transition(Rb87_D1_transition_data)
Rb87_data = {
    'is_boson': True,
    'mass': 1.443160648e-25,
    'transitions': {'D2': Rb87_D2_transition, 'D1': Rb87_D1_transition}
}


class Atom2:
    def __init__(self, atom_data, quiet=True):
        self.data = atom_data
        self.is_boson = self.data['is_boson']
        self.mass = self.data['mass']
        self.transitions = self.data['transitions']
        self.default_transition = self.transitions['D2']

        self.quiet = quiet

    def rabi_freq(self, E_field, transition_name='D2'):
        transition = self.transitions[transition_name]
        d = transition.dEff
        rabi_freq = -E_field * d / hbar
        if not self.quiet:
            print(f'rabi_freq = {rabi_freq:.3f}')
        return rabi_freq

    def optical_potential(self, e_field, wavelength=None, f_field=None):
        # Convert an electric field into an optical potential
        if f_field is None:
            try:
                f_field = const.c / wavelength
            except TypeError:
                print('Must provide wavelength if frequency is not specified')
        omega_field = 2 * np.pi * f_field
        rabi_D1 = self.rabi_freq(e_field, transition_name='D1')
        rabi_D2 = self.rabi_freq(e_field, transition_name='D2')
        omega_0_D1 = self.transitions['D1'].omega_0
        omega_0_D2 = self.transitions['D2'].omega_0
        detuning_D1 = omega_field - omega_0_D1
        detuning_D2 = omega_field - omega_0_D2
        counter_detuning_D1 = omega_field + omega_0_D1
        counter_detuning_D2 = omega_field + omega_0_D2
        U_rotating = (hbar / 4) * (rabi_D1**2 / detuning_D1 + rabi_D2**2 / detuning_D2)
        U_counterrotating = (hbar / 4) * (rabi_D1**2 / counter_detuning_D1 + rabi_D2**2 / counter_detuning_D2)
        return U_rotating + U_counterrotating


RbAtom = Atom2(Rb87_data)


class Atom:
    """Data container for atom"""
    def __init__(self, element):
        if element == 'Rb87':
            self.isboson = True

            self.gammaD1 = 1 / 27.679e-9
            self.transitionfrequencyD1 = 377.107463380e12  # Center transition frequency D1 line (Steck)
            self.IsatD1 = 4.4876 * 1e-3 * (1e2 ** 2)  # Far-Detuned, pi-polarized, W/m^2
            self.deffD1 = np.sqrt((c * ep0 * hbar**2 * self.gammaD1**2) / (4 * self.IsatD1))

            self.gammaD2 = 1 / 26.2348e-9
            self.transitionfrequencyD2 = 384.2304844685e12  # Center transition frequency of D2 line (Steck)
            self.IsatD2 = 2.50399 * 1e-3 * (1e2 ** 2)  # Far-Detuned, pi-polarized, W/m^2
            self.deffD2 = np.sqrt((c * ep0 * hbar**2 * self.gammaD2**2) / (4 * self.IsatD2))

            self.gamma = 0.5 * (self.gammaD1 + self.gammaD2)  # Decay from 5P state (Steck)
            self.transitionfrequency = 0.5 * (
                    self.transitionfrequencyD1 + self.transitionfrequencyD2)  # Arbitrary! Check
            self.finestructure_splitting = self.transitionfrequencyD2 - self.transitionfrequencyD1
            self.nuclearspin = 1.5
            self.spin = 0.5
            self.gj = 2.00233113  # Value from Steck
            self.mass = 1.443160648e-25  # Value from Steck
            self.scattering_length = 100.4 * const.value('Bohr radius')
        else:
            print('Element not defined. Using Rubidium-87')
            self.isboson = True

            self.gammaD1 = 1 / 27.679e-9
            self.transitionfrequencyD1 = 377.107463380e12  # Center transition frequency D1 line (Steck)
            self.IsatD1 = 4.4876 * 1e-3 * (1e2 ** 2)  # Far-Detuned, pi-polarized, W/m^2
            self.deffD1 = np.sqrt((c * ep0 * hbar ** 2 * self.gammaD1 ** 2) / (4 * self.IsatD1))

            self.gammaD2 = 1 / 26.2348e-9
            self.transitionfrequencyD2 = 384.2304844685e12  # Center transition frequency of D2 line (Steck)
            self.IsatD2 = 2.50399 * 1e-3 * (1e2 ** 2)  # Far-Detuned, pi-polarized, W/m^2
            self.deffD2 = np.sqrt((c * ep0 * hbar ** 2 * self.gammaD2 ** 2) / (4 * self.IsatD1))

            self.gamma = 0.5 * (self.gammaD1 + self.gammaD2)  # Decay from 5P state (Steck)
            self.transitionfrequency = 0.5 * (
                    self.transitionfrequencyD1 + self.transitionfrequencyD2)  # Arbitrary! Check
            self.finestructure_splitting = self.transitionfrequencyD2 - self.transitionfrequencyD1
            self.nuclearspin = 1.5
            self.spin = 0.5
            self.gj = 2.00233113  # Value from Steck
            self.mass = 1.443160648e-25  # Value from Steck
            self.scattering_length = 100.4 * const.value('Bohr radius')

    def rabi_freq(self, e_field, transition_dipole=None):
        if transition_dipole is None:
            transition_dipole = self.deffD2
        rabi_freq = -e_field*transition_dipole/hbar
        print(f'rabi_freq = {rabi_freq:.3f}')
        return rabi_freq

    def optical_potential(self, intensity, wavelength=None, f_field=None):
        # Convert an optical intensity into an optical potential
        if f_field is None:
            try:
                f_field = const.c / wavelength
            except TypeError:
                print('Must provide wavelength if frequency is not specified')
        f0_D1 = self.transitionfrequencyD1
        f0_D2 = self.transitionfrequencyD2
        U_rotating = (const.hbar * intensity / 8) * \
                     (
                             (self.gammaD1 ** 2 / (2 * np.pi * (f_field - f0_D1) * self.IsatD1))
                             + (self.gammaD2 ** 2 / (2 * np.pi * (f_field - f0_D2) * self.IsatD2))
                     )
        U_counterrotating = (const.hbar * intensity / 8) * \
                            (
                                    (self.gammaD1 ** 2 / (2 * np.pi * (f_field + f0_D1) * self.IsatD1))
                                    + (self.gammaD2 ** 2 / (2 * np.pi * (f_field + f0_D2) * self.IsatD2))
                            )
        return U_rotating + U_counterrotating

    def optical_potential_2(self, e_field, wavelength=None, f_field=None):
        # Convert an electric field into an optical potential
        if f_field is None:
            try:
                f_field = const.c / wavelength
            except TypeError:
                print('Must provide wavelength if frequency is not specified')
        rabi_D1 = self.rabi_freq(e_field, transition_dipole=self.deffD1)
        rabi_D2 = self.rabi_freq(e_field, transition_dipole=self.deffD2)
        f0_D1 = self.transitionfrequencyD1
        f0_D2 = self.transitionfrequencyD2
        detuning_D1 = 2 * np.pi * (f_field - f0_D1)
        detuning_D2 = 2 * np.pi * (f_field - f0_D2)
        counter_detuning_D1 = 2 * np.pi * (f_field + f0_D1)
        counter_detuning_D2 = 2 * np.pi * (f_field + f0_D2)
        U_rotating = (hbar / 4) * (rabi_D1**2 / detuning_D1 + rabi_D2**2 / detuning_D2)
        U_counterrotating = (hbar / 4) * (rabi_D1**2 / counter_detuning_D1 + rabi_D2**2 / counter_detuning_D2)
        return U_rotating + U_counterrotating


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

    # def beam_intensity_profile(self, x, y, z):
    #     # Gaussian intensity function. Electric field functionality not implemented
    #     [x, y, z] = reduce(lambda vec, f: f(vec), self.trans_list[::-1], [x, y, z])
    #     I0 = self.power_to_max_intensity(self.power, self.waist_x, self.waist_y)
    #     return self.gaussian_beam_profile(x, y, z,
    #                                       I0=I0, wavelength=self.wavelength,
    #                                       w0_x=self.waist_x, w0_y=self.waist_y,
    #                                       z0_x=self.z0_x, z0_y=self.z0_y)

    def make_field(self, x0=(-1, -1, -1), xf=(1, 1, 1), n_steps=(10, 10, 10)):
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
        return

    def transform(self, trans_mat):
        # Add matrix transformation to self.trans_list
        self.trans_list.append(lambda v: E6utils.transform_vec(v, trans_mat))

    def rotate(self, axis=(1, 0, 0), angle=0):
        # Add rotation transformation to self.trans_list
        rot_mat = E6utils.rot_mat(axis, -angle)
        self.transform(rot_mat)
        return

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
        tot_pot_field = self.make_pot_field(x0=(-1e-6,) * 3, xf=(1e-6,) * 3, n_steps=(10,) * 3)
        hess = E6utils.hessian(tot_pot_field, x0=0, y0=0, z0=0)
        vals = np.linalg.eig(hess)[0]  # [0] makes it only return eigenvalues and not eigenvectors
        self.trap_freqs = np.sqrt(vals / self.atom.mass)
        self.trap_freq_geom_mean = np.prod(self.trap_freqs) ** (1 / 3)
        self.trap_depth = -tot_pot_field.max().values
        return self.trap_depth, self.trap_freqs

    def make_pot_field(self, x0=(-1e-6,) * 3, xf=(1e-6,) * 3, n_steps=(10,) * 3):
        x0 = E6utils.single_to_triple(x0)
        xf = E6utils.single_to_triple(xf)
        n_steps = E6utils.single_to_triple(n_steps)
        beam = self.beams[0]
        tot_pot_field = beam.make_field(x0, xf, n_steps)
        tot_pot_field = tot_pot_field.pipe(lambda x: self.atom.optical_potential(x, beam.wavelength))
        for i in range(1, len(self.beams)):
            beam = self.beams[i]
            beam_opt_field = beam.make_field(x0, xf, n_steps)
            beam_pot_field = beam_opt_field.pipe(lambda x: self.atom.optical_potential(x, beam.wavelength))
            tot_pot_field = tot_pot_field + beam_pot_field
        self.pot_field = tot_pot_field
        return self.pot_field

    def calc_psd(self, N, T, quiet=None):
        if quiet is None:
            quiet = self.quiet
        psd = N * ((const.hbar * self.trap_freq_geom_mean) / (const.k * T)) ** 3
        if not quiet:
            print(f'PSD = {psd:.1e}')
        return psd

    def calc_peak_density(self, N, T, quiet=None):
        if quiet is None:
            quiet = self.quiet
        lambda_db = const.h / np.sqrt(2 * np.pi * self.atom.mass * const.k * T)
        n0 = self.calc_psd(N, T, quiet=True) / (lambda_db ** 3)
        if not quiet:
            print(f'Peak Density = {n0*(1e-2 ** 3):.1e} cm^-3')
        return n0

    def print_properties(self):
        print(f'Trap Depth = {(self.trap_depth/const.h)*1e-6:.2f} MHz = {(self.trap_depth/const.k)*1e6:.2f} \\mu K')
        print('Trap Frequencies = \n'
              + '\n'.join([f'{self.trap_freqs[i]/(2*np.pi):.2f} Hz' for i in range(3)])
              )
        print(f'Geometric Mean = {self.trap_freq_geom_mean / (2 * np.pi):.2f} Hz')
        return


def make_grav_pot(mass=Atom('Rb87').mass, grav_vec=(0, 0, -1),
                  x0=(-1,)*3, xf=(1,)*3, n_steps=(10,)*3):
    def grav_func(x, y, z):
        grav_comp = np.dot([x, y, z], grav_vec)
        return -mass*const.g*grav_comp
    grav_pot = E6utils.func3d_xr(grav_func, x0, xf, n_steps)
    return grav_pot


def make_sphere_quad_pot(gf=-(1/2), mf=-1, B_grad=1, units='T/m', trans_list=[],
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
    if not (np.array([strong_axis]) == np.array([0, 0, 1])).all():
        trans_list = [lambda vec: E6utils.transform_vec(vec, mat)] + trans_list

    def sphere_quad_func(x, y, z):
        if trans_list != []:
            [x, y, z] = reduce(lambda vec, f: f(vec), trans_list[::-1], [x, y, z])
        return gf * mf * gyromagnetic_ratio_classical * const.hbar\
                  * np.sqrt((0.5*B_grad*x)**2 + (0.5*B_grad*y)**2 + (B_grad*z)**2)
    sphere_quad_pot = E6utils.func3d_xr(sphere_quad_func, x0, xf, n_steps)
    return sphere_quad_pot
