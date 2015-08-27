#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Piezoelectric materials class with wurtzite structure
#
# Evgeny Shevchenko
# 2012
#
# shevchenko@beam.ioffe.ru
#



from math import pi
from math import exp
import numpy as np
import pylab as pl
import matplotlib as mpl
# from scipy.integrate import simps
# from tm_airy import *

# SI
E_C = 1.602176565e-19   # Electron charge, Coloumbs
HJs = 6.62606957e-34    # Planck constant, J * s

# CGS
E = 4.803250e-10        # Electron charge, CGS = L**1.5 * M**0.5 * T**-1
H = HJs * 1e7           # Planck constant, erg * s
HBAR = H / (2 * pi)     # Dirak (reduced Planck) constant, erg * s

# Common
ME = 9.10938188e-28     # Free electron mass, g
RY = 13.60569253        # Rydberg, eV
# HeV = HJs / E           # Planck constant, eV * s
# HBAReV = HeV / (2 * pi) # Dirak (reduced Planck) constant, eV * s

LSPEED = 29979245800    # cm/s

VACUUM_DIEL = 8.854e-12         # Electric constant in SI, Farad/m

Vpcm2CGS = lambda v: v * 1e8 / LSPEED     # V/cm to CGS electric field units
CGS2Vpcm = lambda v: v * 1e-8 * LSPEED    # ...in reverse

eV2erg   = lambda e: e * E_C * 1e7        # electronvolts to ergs
erg2eV   = lambda e: e / E_C * 1e-7       # ...in reverse

a2cm     = lambda a: a * 1e-8             # Angstrom to cm

VERBOSE = True
# VERBOSE = False




def vegard(x, A, B, bowing=0):
    '''Vegard's law for compounds'''
    return A * x + B * (1 - x) - bowing * x * (1 - x)

def varshni(T, eg0, alpha, beta):
    '''Varshni empirical expression for energy gap'''
    return eg0 - alpha * T**2 / (beta + T)

def energy_gap(T, compounds, energyGaps, alphas, betas):
    '''Energy gap calculation from temperature and compound
    
    Doesn't work yet.
    FIXME do i really need this function?'''
    return 1
    
def debye(T, prec=1e4):
    '''Debye function for temperature dependence of lattice constants

    T    -- considered temperature
    prec -- precision chosen to integrate'''
    from scipy.integrate import simps
    tr = [x/float(prec) for x in range(1,int(prec+1))] 
    y = [t**3/(exp(t * T) - 1) for t in tr]
    return 3 * simps(y, tr, axis=-1, even='avg')
    
def einstein(x):
    '''Debye function for temperature dependence of lattice constants'''
    return 1 / (exp(x) - 1)
    
def lattice_par(T, a0, ainf, Tchr, f=debye):
    '''Lattice parameter depending on temperature
    
    lattice_par(T, a0, ainf, Tchr, f=debye)
    
    T    -- Temperature in K
    a0   -- Lattice constant at T=0
    ainf -- Thermal expansion coefficients at infinity temperature, K**-1
    Tchr -- Characteristic temperature, T
    f    -- Function which may be either \'debye\' or \'einstein\'
    
    The Debye function is more accurate at low temperatures but it's slower
    in calculation. Returns in units of input lattice parameter.'''
    if T == 0: # Avoiding division on zero
        return a0
    else:
        return a0 * (1 + ainf * Tchr * f(Tchr/T))
    







class SemiconductorW(object):
    '''Wurtzite semiconductor implementation
    The only dynamic attribute is temperature, all the orhers 
    are to belong to considered material'''
    def __init__(self,               
                energy_gap=0., energy_gap_0=0.,               
                alpha=0.,  beta=0.,
                diel_st=10,
                a=0., c=0., 
                a0=0., c0=0.,
                a_Tch=0., a_TEC=0., 
                c_Tch=0., c_TEC=0.,
                mass_elx=0.1, mass_elz=0.1, 
                mass_hhx=1.2, mass_lhx=0.8, 
                mass_lhz=0.8, mass_hhz=1.2, 
                c11=0., c12=0., c13=0., c33=0., c44=0., 
                pz13=0., pz33=0., pz15=0., 
                polar_sp=0., 
                name="HeZe", 
                T=300):
        self.name = name
        
        self.energy_gap = energy_gap  # Energy gap, attribute for farther storage
        self.a = a              # Equilibrium lattice parameters
        self.c = c              # These use to be set for T = 300 K
        
        self.alpha = alpha      # Temperature coefficients for Varshni equation
        self.beta = beta
        self.energy_gap_0 = energy_gap_0  # Eg at T=0
        
        self.a0 = a0        # Equilibrium lattice parameters at T=0
        self.c0 = c0
        self.a_Tch = a_Tch  # Characteristic temperature, K
        self.c_Tch = c_Tch
        self.a_TEC = a_TEC  # Thermal expansion coefficient, K**-1
        self.c_TEC = c_TEC
        
        self.set_temp(T)    # Sets energy gap and lattice constants if possible
        
        self.cell_vol = self.cell_vol()
        self.diel_st = diel_st      # Static dielectric constant
        self.mass_elx = mass_elx    # Carriers efficient masses
        self.mass_elz = mass_elz
        self.mass_lhx = mass_lhx
        self.mass_lhz = mass_lhz
        self.mass_hhx = mass_hhx
        self.mass_hhz = mass_hhz
        self.reduced_mass = self.calc_reduced_mass()
        
        self.c11 = c11 # Elastic coefficients, GPa
        self.c12 = c12
        self.c13 = c13
        self.c33 = c33
        self.c44 = c44
        
        self.pz13 = pz13 # Piezoelectric tensor components C / cm**2
        self.pz33 = pz33 
        self.pz15 = pz15 

        self.polar_sp = polar_sp # Spontaneous polarization, C / cm**2
        
        self.exc_bohr_radius = self.calc_exc_bohr_rad() # Exciton Bohr radius, cm
        self.exc_bind_energy = self.calc_exc_bind_energy() # Exciton binding energy, eV
    
    def __calc_energy_gap(self):
        if self.alpha and self.beta and self.energy_gap_0:
            self.energy_gap = varshni(self.temp, self.energy_gap_0, self.alpha, self.beta)
        
    def __calc_lattice_params(self):
        if self.a0 and self.a_Tch and self.a_TEC:
            self.a = lattice_par(self.temp, self.a0, self.a_TEC, self.a_Tch)
        if self.c0 and self.c_Tch and self.c_TEC:
            self.c = lattice_par(self.temp, self.c0, self.c_TEC, self.c_Tch)
        
    def set_temp(self, T):
        '''Sets temperature and calcultes energy gap and lattice constants
        '''
        self.temp = T
        self.__calc_energy_gap()
        self.__calc_lattice_params()
    
    def cell_vol(self):
        '''Lattice cell volume, in angstroms**3
        '''
        return 0.5 * 3.0**1.5 * self.a**2 * self.c
        
    def calc_reduced_mass(self):
        '''Calculates reduced mass of exciton
        FIXME implement for other e-h pairs (z direction, light holes)
        '''
        return (self.mass_elx**-1 + self.mass_hhx**-1)**-1
        
    def calc_exc_bind_energy(self, n=1):
        '''Exciton binding energy, eV
        The only argument of hydrogen-like energy level is set n=1 by default
        '''
        return E**2 / 2 / self.diel_st / self.exc_bohr_radius / n**2 * 1e-7 / E_C
        
    def calc_exc_bohr_rad(self):
        '''Exciton Borh radius, in cm'''
        return self.diel_st * HBAR**2 / (self.reduced_mass * ME * E**2)
        
        






        
class CompoundW(SemiconductorW):
    '''Compound of two wurzite semiconductors'''
    def __init__(self, 
                A, B, 
                x=0.5, T=300,
                energy_gap_bow=0., alpha_bow=0., beta_bow=0., polar_sp_bow=0):
        if (0 <= x <= 1):
            self.x = x
        elif (1 < x <= 100):
            self.x = x / 100.0
        else:
            print "WARNING! Invalid compound! Deleting the instance..."
            self.__del__()
        self.energy_gap_bow = energy_gap_bow
        
        self.A = A
        self.B = B
        self.set_temp(T)
        
        self.name_chooser(self.A, self.B)
        
        self.a = self.vegard(A.a, B.a)
        self.c = self.vegard(A.c, B.c)

        self.alpha = self.vegard(A.alpha, B.alpha, alpha_bow)   # Vegard for Varshni
        self.beta  = self.vegard(A.beta, B.beta, beta_bow)
        self.energy_gap_0 = self.vegard(A.energy_gap_0, B.energy_gap_0, energy_gap_bow)
        
        self.diel_st  = self.vegard(A.diel_st,  B.diel_st)
        self.mass_elx = self.vegard(A.mass_elx, B.mass_elx)
        self.mass_elz = self.vegard(A.mass_elz, B.mass_elz)
        self.mass_lhx = self.vegard(A.mass_lhx, B.mass_lhx)
        self.mass_lhz = self.vegard(A.mass_lhz, B.mass_lhz)
        self.mass_hhx = self.vegard(A.mass_hhx, B.mass_hhx)
        self.mass_hhz = self.vegard(A.mass_hhz, B.mass_hhz)
        
        self.c11 = self.vegard(A.c11, B.c11)
        self.c12 = self.vegard(A.c12, B.c12)
        self.c13 = self.vegard(A.c13, B.c13)
        self.c33 = self.vegard(A.c33, B.c33)
        self.c44 = self.vegard(A.c44, B.c44)
        
        self.pz13 = self.vegard(A.pz13, B.pz13)
        self.pz33 = self.vegard(A.pz33, B.pz33)
        self.pz15 = self.vegard(A.pz15, B.pz15)
        
        self.polar_sp_bow = polar_sp_bow
        self.polar_sp = self.vegard(A.polar_sp, B.polar_sp, self.polar_sp_bow)
        
        self.exc_bohr_radius = self.vegard(A.exc_bohr_radius, B.exc_bohr_radius)
        self.exc_bind_energy = self.vegard(A.exc_bind_energy, B.exc_bind_energy)
        
    def vegard(self, Apar, Bpar, bowing=0):
        return vegard(self.x, Apar, Bpar, bowing)

    def set_temp(self, T):
        self.A.set_temp(T)
        self.B.set_temp(T)
        self.a = self.vegard(self.A.a, self.B.a)
        self.c = self.vegard(self.A.c, self.B.c)
        self.energy_gap = self.vegard(self.A.energy_gap, self.B.energy_gap, self.energy_gap_bow)
        
    def name_chooser(self, A, B):
        if self.__chk_compound_name(A, B, 'GaN', 'AlN'):
            self.name = 'AlGaN'
        elif self.__chk_compound_name(A, B, 'GaN', 'InN'):
            self.name = 'InGaN'
        elif self.__chk_compound_name(A, B, 'AlN', 'InN'):
            self.name = 'InAlN'
        else:
            self.name = 'Unknown'
        
    def __chk_compound_name(self, A, B, nA, nB):
        '''Compound name check'''
        return ((A.name == nA and B.name == nB) or (A.name == nB and B.name == nA))







class LayerW(CompoundW):
    '''Layer class for wurtzite heterostructure'''
    def __init__(self, A, B, x, L,
                T=300., 
                energy_gap_bow=0., alpha_bow=0., beta_bow=0., polar_sp_bow=0.,
                prev_lr=None, next_lr=None):
        super(LayerW, self).__init__(A, B, x, T, energy_gap_bow, alpha_bow, beta_bow, polar_sp_bow)
        self.width = L
        self.a_eq = self.vegard(A.a, B.a)  # Equilibrium lattice constants
        self.c_eq = self.vegard(A.c, B.c)
        self.D = 0.          # D coefficient used in shear modulus calculation
        self.shear = 0.      # Shear modulus (G)
        self.prev_lr = None
        self.next_lr = None
        
        if isinstance(prev_lr, self.__class__):
            self.prev_lr = prev_lr  # Previous layer in heterostructure
            prev_lr.next_lr = self
        else:
            self.prev_lr = None 
            
        if isinstance(next_lr, self.__class__):
            self.next_lr = next_lr  # Next layer in heterostructure
            next_lr.prev_lr = self
        else:
            self.next_lr = None

        self.strain_xx = 0.  # Strain tensor components
        self.strain_zz = 0.
        self.polar_pz = 0.   # Piezoelectric polarization
        self.polar_tot = self.get_polar_tot()

    def get_D(self):
        '''Coefficient D for shear modulus G'''
        if not self.D == 0:
            return self.D
        self.D = 2 * (self.c12 + 2*(self.c13 * self.c13 / self.c33)) / (self.c11 + 2*self.c12)
        return self.D
    
    def get_shear(self):
        '''Shear modulus for wurtzite structure'''
        if not self.shear == 0:
            return self.shear
        self.shear = 2*(self.c11 + self.c12 - 2*(self.c13**2 / self.c33));
        return self.shear
        
    def get_int_a(self):
        '''Calculation of interface lattice parameter a'''
        if (self.prev_lr == None):
            self.a = self.a_eq
        elif not self.a == self.a_eq:
            return self.a
        else:
            self.a = (self.prev_lr.a * self.prev_lr.get_shear() * self.prev_lr.width + \
                      self.get_shear() * self.width * self.a) / \
                     (self.prev_lr.get_shear() * self.prev_lr.width + self.get_shear() * self.width)
        return self.a
        
    def get_int_c(self):
        '''Calculation of interface lattice parameter c'''
        self.c = self.c_eq * (1 - self.get_D() * self.get_strain_xx())
        return self.c
        
    def get_strain_xx(self):
        '''Strain in X and Y directions'''
        if not self.strain_xx == 0.:
            return self.strain_xx
        self.strain_xx = self.get_int_a() / self.a_eq - 1
        return self.strain_xx
        
    def get_strain_zz(self):
        '''Strain in Z direction'''
        if not self.strain_zz == 0.:
            return self.strain_zz
        self.strain_zz = self.get_int_c() / self.c_eq - 1
        return self.strain_zz
        
    def get_polar_pz(self):
        '''Piezoelectric polarization'''
        if not self.polar_pz == 0:
            return self.polar_pz
        self.polar_pz = self.pz13 * self.get_strain_xx() * 2 + \
                        self.pz33 * self.get_strain_zz()
        return self.polar_pz
        
    def get_polar_tot(self):
        '''Total polarization'''
        self.polar_tot = self.get_polar_pz() + self.polar_sp
        return self.polar_tot

    def get_exciton_energy(self):
        pass
        




# Heterostructure class is in heterostructure.py







        
        
# Wurtzite materials declaration  
# TODO Fill properties sources (papers, books, tailoring, etc.)

# [1] Jmerik et al. Semiconductors Vol. 42 No. 12 (2008)
# [2] Caro, Schulz, Healy, O'Reilly, J. Appl. Phys. 109, 084110 (2011)
# [3] Figge, Kroncke, Hommel, Epelbaum, Appl. Phys. Lett. 94, 101915 (2009)
# [4] Vurgaftman, Meyer, Ram-Mohan, J. Appl. Phys. 89, 11, 5815 (2001)
# [5] Junqiao Wu, J. Appl. Phys. 106, 011101 (2009)
# [6] http://www.ioffe.ru/SVA/NSM/Semicond/
# [7] Vurgaftman, Meyer, J. Appl. Phys. 94, 6, (2003)




GaN = SemiconductorW(
    energy_gap = 3.42,      # Energy gap (eV)                           [1]
    alpha = 0.94e-3,        # Coefficient for Varshni equation          [4]
    beta = 791,             # ...                                       [4]
    energy_gap_0 = 3.51,    # Energy gap at T = 0 (eV)                  [7]
    diel_st = 8.9,          # Electric constant                         [5]
    a = 3.189,              # Lattice parameter T = 300 K a (angstrom)  [7]
    c = 5.185,              #                         ... c (angstrom)  [7]
    a0 = 3.1872,            # Lattice parameterat T = 0, a              tailored from [2]
    c0 = 5.1825,            #                        ... c              tailored from [2] 
    a_Tch = 868,            # Characteristic temperature for a, K       [3]
    c_Tch = 898,            #                        ... for c, K       [3]
    a_TEC = 6.24e-6,        # Thermal expansion coeff. for a, K**-1     [3]
    c_TEC = 5.73e-6,        #                      ... for c            [3]
    mass_elx = 0.2,         # Efficient electron mass in x direction    [7]
    mass_elz = 0.2,         # Efficient electron mass in z direction    [7]
    mass_hhx = 1.6,         # Efficient heavy hole mass in x direction  [6]
    mass_hhz = 1.75,        #                       ... in z direction  [6]
    mass_lhx = 1.1,         # Efficient light hole mass in x direction  [6]
    mass_lhz = 0.15,        #                       ... in z direction  [6]
    c11 = 390.,             # Elastic constant c11 (GPa)                [7]
    c12 = 145.,             #              ... c12 (GPa)                [7]
    c13 = 106.,             #              ... c13 (GPa)                [7]
    c33 = 398.,             #              ... c33 (GPa)                [7]
    c44 = 105.,             #              ... c44 (GPa)                [7]
    pz13 = -0.45,           # Piezoelectric constant pz13 (C/m**2)      [2]
    pz33 = 0.83,            #                    ... pz33 (C/m**2)      [2]
    pz15 = -0.38,           #                    ... pz15 (C/m**2)      [2]
    polar_sp = -0.034,      # Spontaneous polarization (C/m**2)         [7]  
    name = 'GaN')     
                    



AlN = SemiconductorW(
    energy_gap = 6.08,     # Energy gap (eV) [Jmerik] 
    alpha = 2.63e-3,       # coefficient for Varshni equation
    beta = 2082,           # coefficient for Varshni equation
    energy_gap_0 = 6.25,   # Energy gap at T = 0                        [7]
    diel_st = 8.5,         # Electric constant [1]                     
    a = 3.112,             # Lattice par a at T = 300 K (Angstrom)      [7] 
    c = 4.982,             # Lattice par c at T = 300 K (Angstrom)      [7]               
    a0 = 3.1112,           # Lattice par a at T = 0
    c0 = 4.9807,           # Lattice par a at T = 0
    a_Tch = 1455,          # Characteristic temperature, K
    c_Tch = 1317,          #       
    a_TEC = 7.1e-6,        # Infinity coefficient, K**-1
    c_TEC = 5.8e-6,        #
    mass_elx = 0.32,       # Efficient electron mass in x direction     [7]                 
    mass_elz = 0.3,        # Efficient electron mass in z direction     [7]                 
    mass_hhx = 10.42,      # Efficient heavy hole mass in x direction   
    mass_hhz = 3.53,       # Efficient heavy hole mass in z direction   [1]
    mass_lhx = 0.24,       # Efficient light hole mass in x direction   
    mass_lhz = 3.53,       # Efficient light hole mass in z direction   [1] 
    c11 = 410.,            # Elastic constant c11, GPa                         
    c12 = 149.,            # Elastic constant c12, GPa                         
    c13 = 99.,             # Elastic constant c13, GPa                          
    c33 = 389.,            # Elastic constant c33, GPa                         
    c44 = 125.,            # Elastic constant c44, GPa                         
    pz13 = -0.48,          # Piezoelectric constant pz13 (C / m**2)     [1]      
    pz33 = 1.55,           # Piezoelectric constant pz33 (C / m**2)     [1]      
    pz15 = 1.,             # FIXME Piezoelectric constant pz15 (C / m**2) [1]          
    polar_sp = -0.09,      # Spontaneous polarization (C / m**2)        [7]   
    name = 'AlN')                 




# TODO InN properties
InN = SemiconductorW(      # InN PROPERTIES ARE NOT SPECIFIED CORRECTLY!
    energy_gap = 0,        # Energy gap (eV) [Jmerik]  
    alpha = 0.245e-3,      # coefficient for Varshni equation           [7]
    beta = 624,            # coefficient for Varshni equation           [7]
    energy_gap_0 = 0.78,   # Energy gap at T = 0                        [7]
    diel_st = 8.5,         # Electric constant [1]
    a = 3.545,             # Lattice par a (A)                          [7]
    c = 5.703,             # Lattice par c (A)                          [7]
    a0 = 0,                # Lattice par a at T = 0
    c0 = 0,                # Lattice par a at T = 0
    a_Tch = 0,             # Characteristic temperature, K
    c_Tch = 0,             
    a_TEC = 0,             # Infinity coefficient, K**-1
    c_TEC = 0,              
    mass_elx = 0.3,        # Efficient electron mass in x direction [1]
    mass_elz = 0.3,        # Efficient electron mass in z direction [1]
    mass_hhx = 10.42,      # Efficient heavy hole mass in x direction
    mass_hhz = 3.53,       # Efficient heavy hole mass in z direction [1]
    mass_lhx = 0.24,       # Efficient light hole mass in x direction
    mass_lhz = 3.53,       # Efficient light hole mass in z direction [1] 
    c11 = 223.,            # Elastic constant c11, GPa                  [7]
    c12 = 115.,            # Elastic constant c12, GPa                  [7]
    c13 = 92.,             # Elastic constant c13, GPa                  [7]
    c33 = 224.,            # Elastic constant c33, GPa                  [7]
    c44 = 48.,             # Elastic constant c44, GPa                  [7]
    pz13 = -0.48,          # Piezoelectric constant pz13 (C / m**2) [1]
    pz33 = 1.55,           # Piezoelectric constant pz33 (C / m**2) [1]
    pz15 = 1.,             # Piezoelectric constant pz15 (C / m**2) [1]
    polar_sp = -0.042,     # Spontaneous polarization (C / m**2)        [7]
    name='InN')





AlGaN = lambda x, T: CompoundW(AlN, GaN, x, T, energy_gap_bow=0.7, polar_sp_bow=-0.021,
                                                        alpha_bow=2.15e-3, beta_bow=1561)


AlGaNLayer = lambda x, width, T: LayerW(AlN, GaN, x, width, T,
                                                energy_gap_bow=0.7, polar_sp_bow=-0.021, 
                                                        alpha_bow=2.15e-3, beta_bow=1561)


# FIXME alpha and beta bowings
AlInN = lambda x, T: CompoundW(AlN, InN, x, T, energy_gap_bow=2.5, alpha_bow=2.15e-3, beta_bow=1561)



# FIXME alpha and beta bowings
InGaN = lambda x, T: CompoundW(InN, GaN, x, T, energy_gap_bow=1.4, alpha_bow=2.15e-3, beta_bow=1561)






# TODO these materials
# InGaN
# AlInN
# AlInGaN
# ZnO
# ZnMgO









if __name__ == '__main__':
    
    T = 20 
    # layers = [  AlGaNLayer(0.65,  80.,      T),
    #             AlGaNLayer(0.55,  80,       T),
    #             AlGaNLayer(0.65,  80.,      T)]
    
    # hs = Heterostructure(layers)
    # print "Overlap integral:", hs.get_overlap(0, 0)
    # hs.plot_wavefunctions(1, wfmul=1, save=True)





# T = input("Please, set temperature: ")
    # print "x".ljust(4), "Bind energy".ljust(12), "Bohr radius".ljust(12)
    # print "".ljust(7), "meV".ljust(12), "nm".ljust(12)
    # for x in xrange(0, 101, 10):
    #     a = AlGaN(x/100.,T)
    #     print str(x).ljust(7), \
    #
      # str(round(a.exc_bind_energy * 1e3, 2)).ljust(12), \
    #           str(round(a.exc_bohr_radius * 1e7,2)).ljust(12)
