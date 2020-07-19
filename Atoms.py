import numpy as np
import scipy.constants as const
from sympy.physics.wigner import wigner_3j, wigner_6j
from E6py import E6utils

hbar = const.hbar
c = const.c
ep0 = const.epsilon_0


class FineTransition:
    def __init__(self, transition_data):
        self.data = transition_data
        self.name = self.data['name']
        self.Jg = self.data['Jg']
        self.Je = self.data['Je']
        self.Inuc = self.data['Inuc']
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
        degeneracy_factor = (2*self.Je+1)/(2*self.Jg+1)
        return np.sqrt((3*np.pi*ep0*hbar*c**3*self.gamma)/(self.omega_0**3))*np.sqrt(degeneracy_factor)

    def dJJ_to_dEff(self):
        """ Convert reduced dipole element to effect far-detuned dipole element

        Effective far detuned dipole moment for pi polarized light. This is the effective dipole moment which couples
        any hyperfine ground state to a single fine structure excited state manifold. The factor of
        1/3 arises from summing the coupling of the ground hyperfine state to each possible excited state
        hyperfine state. It can also be viewed as *averaging* the coupling of each fine structure ground state to
        the fine structure excited states over all of the possible fine structure ground states.
        Note that this simplifcation may only be valid for Jg = 1/2
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

    def calc_hyperfine_transition_dipole(self, Fg, Fe, mFg, mFe, q):
        dFF = (self.dJJ * (-1)**(Fe + self.Jg + 1 + self.Inuc) * np.sqrt((2 * Fe + 1) * (2 * self.Jg + 1))
               * wigner_6j(self.Jg, self.Je, 1, Fe, Fg, self.Inuc).evalf())
        d_hf = dFF * (-1)**(Fe - 1 + mFg) * np.sqrt(2 * Fg + 1) * wigner_3j(Fe, 1, Fg, mFe, q, -mFg).evalf()
        return d_hf


class Atom:
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

    def rabi_freq_from_intensity(self, intensity, transition_name='D2'):
        return self.rabi_freq(E6utils.intensity_to_e_field(intensity), transition_name)

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
        counter_detuning_D1 = - omega_field - omega_0_D1
        counter_detuning_D2 = - omega_field - omega_0_D2
        U_rotating = (hbar / 4) * (np.abs(rabi_D1)**2 / detuning_D1
                                   + np.abs(rabi_D2)**2 / detuning_D2)
        U_counterrotating = (hbar / 4) * (np.abs(rabi_D1)**2 / counter_detuning_D1
                                          + np.abs(rabi_D2)**2 / counter_detuning_D2)
        return U_rotating + U_counterrotating

    def optical_potential_from_intensity(self, intensity, wavelength=None, f_field=None):
        return self.optical_potential(E6utils.intensity_to_e_field(intensity), wavelength, f_field)


Rb87_Atom = Atom(Rb87_data)
