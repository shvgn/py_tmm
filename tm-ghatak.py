#!/usr/bin/env python
# coding: utf-8
#
# E. Shevchenko
# shevchenko@beam.ioffe.ru
# 

from matplotlib import rc
import numpy as np
import pylab as pl
from material import *

CALC_TOLERANCE = 1e-16
CALC_STEPS_NUM = 1e3
CALC_STEP = CALC_STEPS_NUM ** -2
FAKE_ZERO = np.finfo(np.float).tiny

VERBOSE = True
# VERBOSE = False


def m2prod(*mx):
    m = np.matrix(np.eye(2))
    for mi in mx: m *= mi
    return m


class Interval(object):
    """Potential interval with electric field"""
    def __init__(self, width=1, angle=0.2, v_start=0, mass=0.2):
        self.width = width
        self.v_start = v_start
        self.angle = angle
        self.v_end = v_start + angle * width
        self.mass = mass
    
    def __repr__(self):
        return "Interval: " + str(self.width) + " width, V: [" + str(self.v_start) + ", " + \
                str(self.v_end) + "], angle: " + str(self.angle)



class IntervalStack(object):
    """Class for stack of potential intervals"""
    def __init__(self, *intervals):
        self.data = []
        self.dots = []
        for i in intervals: self.__append(i)
        self.__process()

    def __len__(self):
        return len(self.data)

    def __append(self, *intervals):
        for interval in intervals:
            if isinstance(interval, Interval):
                self.data.append(interval)

    def __process(self):
        length = len(self.data)
        if length == 0: return
        coordinate = 0
        for i in range(length-1):
            coordinate += self.data[i].width
            self.dots.append({
                'height': self.data[i+1].v_start - self.data[i].v_end,
                'coordinate': coordinate
                })
        # Moving common potential minimum to zero
        vmin = min([min(interval.v_start, interval.v_end) for interval in self.data])
        for interval in self.data:
            interval.v_start -= vmin
            interval.v_end   -= vmin
        
    def __repr__(self):
        cw = 20
        aw = 25
        repstr = ""
        for i in range(len(self.data)):
            repstr += str(i) + ": "
            anglestr  = "angle: " + str(self.data[i].angle)
            if i == 0:
                coordstr = "[0, " + str(self.dots[i]['coordinate']) + "]"
            else:
                coordstr = "["  + str(self.dots[i]['coordinate']) + \
                           ", " + str(self.dots[i]['coordinate'] + self.data[i].width) + "]"
            vstr = "V: [" + str(self.data[i].v_start) + ", " + str(self.data[i].v_end) + "]"
            repstr += coordstr.ljust(cw) + anglestr.ljust(aw) + vstr
            repstr += "\n"
        return repstr



class MatterExpInterval(Interval):
    """Rectangle region for calculating quasi-bound states in arbitral potential"""
    def __init__(self, width, height, mass):
        if width > 0.1: width *= 1e-7 # Converting cm to nm
        self.height_ev = height
        self.height = eV2erg(self.height_ev)
        self.label = ''
        super(MatterExpInterval, self).__init__(width=width, angle=0.0, v_start=self.height, mass=mass)



class MatterStack(IntervalStack):
    """docstring for MatterStack"""
    def __init__(self, *intervals): #, ficticious_width=1e-3, ficticious_mass=0.25):
        # self.ficticious_mass = ficticious_mass
        # self.ficticious_width = ficticious_width
        super(MatterStack, self).__init__(*intervals)
        for i in intervals: self.__append(intervals)
        self.__process()
        # self.states = []
        self.eigen_energy = []

    def __append(self, *intervals):
        for interval in intervals:
            if isinstance(interval, MatterExpInterval):
                self.data.append(interval)

    def __process(self):
        # Moving common potential minimum to zero
        vmin = min([interval.height for interval in self.data])
        for interval in self.data:
            interval.height -= vmin
            interval.height_ev = erg2eV(interval.height)
        # Adding ficticious interval
        # self.data.insert(0, MatterExpInterval(self.ficticious_width, 0, self.ficticious_mass))
        self.data.insert(0, MatterExpInterval(3e-7, 0, 0.067))
        if len(self.data) == 0: return
        coordinate = 0
        for i in range(len(self.data)-1):
            coordinate += self.data[i].width
            self.dots.append({
                'height': self.data[i+1].height - self.data[i].height,
                'height_ev': erg2eV(self.data[i+1].height - self.data[i].height),
                'coordinate': coordinate
                })
        for i in range(1,len(self.data)-1):
            if self.data[i].height < self.data[i-1].height and self.data[i].height < self.data[i+1].height:
                self.data[i].label = 'well'
            elif self.data[i].height < self.data[i-1].height or self.data[i].height < self.data[i+1].height:
                self.data[i].label = 'stair'
            else:
                self.data[i].label = 'barrier'

    def __repr__(self):
        cw = 20
        repstr = ""
        for i in range(len(self.data)):
            repstr += str(i) + ": "
            if i == 0:
                coordstr = "[0, " + str(self.dots[i]['coordinate'] * 1e7) + "] nm"
            else:
                coordstr = "["  + str( self.dots[i-1]['coordinate'] * 1e7) + \
                           ", " + str((self.dots[i-1]['coordinate'] + self.data[i].width) * 1e7) + "] nm"
            vstr = "V = " + str(erg2eV(self.data[i].height)) + " eV"
            repstr += coordstr.ljust(cw) + vstr + " label: " + self.data[i].label
            repstr += "\n"
        return repstr

    def get_wave_number(self, index, energy):
        return np.sqrt(2 * self.data[index].mass * ME * np.complex(energy - self.data[index].height)) / HBAR

        
    def get_delta(self, index, energy):
        if index == 0 or index == 1: 
            return np.complex(0)
        return self.get_wave_number(index, energy) * np.sum([interval.width for interval in self.data[1:index]])
        

    def get_transfer_matrix_gh(self, index, energy):
        # Wave numbers manipulations
        wn_this = self.get_wave_number(index, energy)
        wn_next = self.get_wave_number(index+1, energy)
        wn_sum  = wn_this + wn_next
        r = (wn_this - wn_next) / wn_sum
        t = wn_sum / wn_this / 2
        d = wn_this * self.data[index].width
        return t * np.matrix([[    np.exp(-1j * d), r * np.exp(-1j * d) ],
                              [r * np.exp( 1j * d),     np.exp( 1j * d) ]])

    def get_transfer_matrix_mine(self, index, energy):
        if energy == self.data[index+1].height:
            km_ratio = np.sqrt(self.data[index].mass * np.complex(energy - self.data[index].height) / \
                               self.data[index+1].mass / FAKE_ZERO)
        else:
            km_ratio = np.sqrt(self.data[index].mass * np.complex(energy - self.data[index].height) / \
                               self.data[index+1].mass / np.complex(energy - self.data[index+1].height))
        kcp = 1 + km_ratio
        kcm = 1 - km_ratio
        d_this = self.get_delta(index,   energy)
        d_next = self.get_delta(index+1, energy)
        kd = self.get_wave_number(index, energy) * self.data[index].width
        # print "d_this =", d_this, "d_next =", d_next, "kd", kd, "1j * ( d_this + d_next - kd) =", 1j * ( d_this + d_next - kd)
        return np.matrix([[kcp / 2 * np.exp( 1j * (d_this - d_next - kd)),
                           kcm / 2 * np.exp( 1j * (d_this + d_next - kd))],
                          [kcp / 2 * np.exp(-1j * (d_this + d_next - kd)),
                           kcm / 2 * np.exp(-1j * (d_this - d_next - kd))]])

    def get_transfer_matrix(self, index, energy):
        return self.get_transfer_matrix_gh(index, energy)
        # return self.get_transfer_matrix_mine(index, energy)

    def get_interval_object(self, index, energy):
        obj = {'energy': energy, 'energy_ev': erg2eV(energy)}
        uvec = m2prod(*[self.get_transfer_matrix(i, energy) for i in range(index, len(self.data)-1)]) \
                                                                                * np.matrix([[1], [0]])
        obj['up'] = np.float(uvec[0])
        obj['un'] = np.float(uvec[1])
        obj['delta'] = self.get_delta(index, energy)
        obj['wavenum'] = self.get_wave_number(index, energy)
        return obj

    def get_first_interval_object(self, energy):
        return self.get_interval_object(0, energy)
        
    def get_energy_range(self, energy_min=None, energy_max=None, stepsnum=CALC_STEPS_NUM):
        if energy_min == None:
            energy_min = 0
        if energy_max == None: 
            energy_max = max([i.height for i in self.data])
        return np.linspace(energy_min, energy_max, stepsnum)

    def get_lorenzian_ratio(self, index, energy):
        return (np.abs(self.get_interval_object(index, energy)['up']) / \
                np.abs(self.get_first_interval_object(energy)['up']))**2

    def get_lorenzian_ratio_derivative(self, index, energy, step=CALC_TOLERANCE):
        return (self.get_lorenzian_ratio(index, energy + step) - \
                self.get_lorenzian_ratio(index, energy - step)) / 2 / step

    def get_lorenzian_ratio_derivative2(self, index, energy, step=CALC_TOLERANCE):
        return (self.get_lorenzian_ratio_derivative(index, energy + step) - \
                self.get_lorenzian_ratio_derivative(index, energy - step)) / 2 / step

    def get_lorenzians(self, index):
        energy_range = get_energy_range()
        spectra = np.array([self.get_lorenzian_ratio(index, e) for e in energy_range])
        return (energy_range, spectra)

    def get_interval_eigen_energy(self, index):
        eigen_energy = []
        energy_range = self.get_energy_range()
        lorenz_range = np.array([ self.get_lorenzian_ratio(index, e) for e in energy_range])
        lorenz_deriv = np.array([ self.get_lorenzian_ratio_derivative(index, e) for e in energy_range])
        for i in range(1, len(energy_range)-1):
            if np.nan in (lorenz_deriv[i-1], lorenz_deriv[i], lorenz_deriv[i+1]):
                continue
            elif lorenz_deriv[i-1] * lorenz_deriv[i+1] < 0 and \
                 lorenz_range[i-1] < lorenz_range[i] > lorenz_range[i+1]:
                if VERBOSE:
                    print "***  ***  Peak detected near", erg2eV(energy_range[i]), "eV!"
                    print "* Derivative before the dot:", lorenz_deriv[i-1]
                    print "* Derivative  after the dot:", lorenz_deriv[i+1]
                eigen_energy.append(self.get_peak_energy(index, energy_range[i-1], energy_range[i+1]))
        return eigen_energy

    def get_eigen_energy(self):
        '''Returns all energy eigen values (in ergs) from all wells and stairs'''
        if not self.eigen_energy == []:
            return self.eigen_energy
        eigen_energy = []
        for index in range(len(self.data)):
            if self.data[index].label not in ('well', 'stair'):
                continue
            for energy in self.get_interval_eigen_energy(index):
                if energy in eigen_energy:
                    continue
                else:
                    eigen_energy.append(energy)
        self.eigen_energy = sorted(eigen_energy)
        return self.eigen_energy

    def get_peak_energy_by_bisections(self, index, energy_min, energy_max, tol=CALC_TOLERANCE):
        """Bisection method"""
        energy_mid = (energy_min + energy_max) / 2
        if VERBOSE:
            print "Starting mid energy =", erg2eV(energy_mid)
        func = self.get_lorenzian_ratio_derivative
        f = func(index, energy_mid)
        while np.abs(f) > tol:
            if np.sign(func(index, energy_min)) * np.sign(f) < 0:
                energy_max = energy_mid
            elif np.sign(func(index, energy_max)) * np.sign(f) < 0:
                energy_min = energy_mid
            if energy_mid == (energy_min + energy_max) / 2:
                return energy_mid
            energy_mid = (energy_min + energy_max) / 2
            f  = func(index, energy_mid)
            if VERBOSE:
                print "Mid energy =", erg2eV(energy_mid), "\tFunc =", f
        if VERBOSE:
            print " :::::: Final mid energy:", erg2eV(energy_mid)
        return energy_mid

    def get_peak_energy_by_newtonraphson(self, index, energy_min, energy_max, tol=CALC_TOLERANCE):
        """Newton-Raphson method
        Doesn't work properly"""
        energy_mid = (energy_min + energy_max) / 2
        if VERBOSE:
            print "Starting mid energy =", erg2eV(energy_mid)
        func = self.get_lorenzian_ratio_derivative
        dfunc = self.get_lorenzian_ratio_derivative2
        f = func(index, energy_mid)
        df = func(index, energy_mid)
        while np.abs(f) > tol:
            energy_mid = energy_mid - f / df
            f = func(index, energy_mid)
            df = func(index, energy_mid)
            if VERBOSE:
                print "Mid energy =", erg2eV(energy_mid), "\tFunc =", f, "\tdFunc =", df
        if VERBOSE:
            print " :::::: Final mid energy:", erg2eV(energy_mid)
        return energy_mid

    def get_peak_energy(self, index, energy_min, energy_max, tol=CALC_TOLERANCE):
        return self.get_peak_energy_by_bisections(index, energy_min, energy_max, tol=tol)
        # return self.get_peak_energy_by_newtonraphson(index, energy_min, energy_max, tol=tol)

    def get_states(self):
        pass
        


    def plot_wells_lorenzians(self, energy_min=None, energy_max=None, stepsnum=CALC_STEPS_NUM):
        energy_range = self.get_energy_range(energy_min, energy_max, stepsnum)
        spectra = []
        for i in range(len(self.data)):
            if self.data[i].label in ('well', 'stair'): 
                spectra.append(np.array([ self.get_lorenzian_ratio(i, e) for e in energy_range]))
            else:
                spectra.append(None)

        # rc('font',**{'family':'sans-serif','sans-serif':['Palatino']})
        rc('text', usetex=True)
        figure = pl.figure()
        
        # Subplots
        axlin  = figure.add_subplot(121)
        # pl.title(r"Ratio $\left|u_i/u_1\right|^2$")
        pl.title(r"Ratio $\left|\frac{u_i}{u_1}\right|^2$")
        pl.xlim(xmin = erg2eV(energy_range[0] * 0.9), xmax = erg2eV(energy_range[-1] * 1.1))
        pl.grid()

        axlogy = figure.add_subplot(122)
        pl.title(r"Ratio $\lg_{10}\left|\frac{u_i}{u_1}\right|^2$")
        pl.xlim(xmin = erg2eV(energy_range[0] * 0.9), xmax = erg2eV(energy_range[-1] * 1.1))
        pl.grid()

        for i in range(len(self.data)):
            if self.data[i].label not in ('well', 'stair'): 
                continue
            axlin.plot( erg2eV(energy_range), spectra[i], label = self.data[i].label + ' \#' + str(i) \
                                                                     + " %.1f nm" % (self.data[i].width * 1e7))
            axlogy.semilogy( erg2eV(energy_range), spectra[i], label = self.data[i].label + ' \#' + str(i) \
                                                                     + " %.1f nm" % (self.data[i].width * 1e7))
        axlin.legend(loc=2) 
        axlogy.legend(loc=2)
        pl.show()

    def get_stack_profile(self):
        offset = -self.data[0].width # Converting to nm
        x, y = [], []
        for interval in self.data:
            # Interval begins
            x.append(offset)
            y.append(interval.height_ev)
            # Interval ends
            x.append(interval.width + offset) # Converting to nm
            y.append(interval.height_ev)
            offset += interval.width
        return (np.array(x), np.array(y))

    def get_wavefunction(self, energy, stepsnum=CALC_STEPS_NUM):
        if not energy in self.eigen_energy:
            print " * * * WARNING! Getting wavefunction of unexisting energy eigen value! * * *"
        # An interval obj = {'energy':, 'energy_ev':, 'up':, 'un':, 'delta':, 'wavenum'}
        offset = -self.data[0].width # Width of the ficticious interval
        step = np.sum([interval.width for interval in self.data]) / stepsnum
        x, y = [], []
        for i in range(len(self.data)):
            obj = self.get_interval_object(i, energy)
            # xloc = np.arange(offset, self.data[i].width + offset, step)
            xloc = np.linspace(offset, self.data[i].width + offset, stepsnum)
            yloc = obj['up'] * np.exp( 1j * (obj['wavenum'] * xloc - obj['delta'])) + \
                   obj['un'] * np.exp(-1j * (obj['wavenum'] * xloc - obj['delta'])) 
            x = np.concatenate((x, xloc))
            y = np.concatenate((y, yloc))
            offset += self.data[i].width
            y = self.normilized(step, y)
        return (np.array(x), np.array(y))

    def normilized(self, step, y):
        S = 0
        imin = int(self.data[0].width / step)
        imax = int(self.data[-1].width / step)
        for i in range(len(y)-1):
            S += step * (np.abs(y[i])**2 + np.abs(y[i+1])**2) / 2   # y * dx
        if VERBOSE: print "S is", S
        y = y / np.power(S, 0.5)
        limit = erg2eV(max([i.height for i in self.data])) / len(self.eigen_energy) / 2
        y /= np.max(np.abs(y))
        y *= limit
        # while np.max(np.abs(y[imin : -imax])) > limit:
        #     y /= 2
        # while np.max(np.abs(y[imin : -imax])) < limit:
        #     y *= 2
        if VERBOSE: print "Wavefunction maximum:", np.max(y)
        return y# / divider

    def plot_states(self, nwf=None):
        toNm = 1e7
        self.eigen_energy = self.get_eigen_energy()
        if nwf == None:
            nwf = len(self.eigen_energy)
        fig = pl.figure()
        pl.grid(True)
        vmax = erg2eV(max([i.height for i in self.data]))
        pl.ylim(-0.1 * vmax, 1.1 * vmax)
        pl.xlim(xmin=-1, xmax=(np.sum([interval.width for interval in self.data]) - self.data[0].width) * toNm)
        bandx, bandy = self.get_stack_profile()
        pl.plot(bandx * toNm, bandy, lw=5, color='0.65')
        for i in range(nwf):
            energy = self.eigen_energy[i]
            x, y = self.get_wavefunction(energy)
            pl.plot(x * toNm, y + erg2eV(energy), lw=2)
            pl.plot(np.array([x[0], x[-1]]) * toNm, np.array([erg2eV(energy)]*2), lw=1)
        pl.show()




if __name__ == '__main__':
    Eg = 4.
    dEg = 0.3
    i1 = MatterExpInterval(5e-7, Eg,        0.25)
    i2 = MatterExpInterval(3e-7, Eg - dEg,  0.2 ) # Well
    i3 = MatterExpInterval(2e-7, Eg,        0.25)
    i4 = MatterExpInterval(3e-7, Eg - dEg,  0.2 ) # Well
    i5 = MatterExpInterval(2e-7, Eg,        0.25)
    i6 = MatterExpInterval(3e-7, Eg - dEg,  0.2 ) # Well
    i7 = MatterExpInterval(2e-7, Eg,        0.25)
    i8 = MatterExpInterval(3e-7, Eg - dEg,  0.2 ) # Well
    i9 = MatterExpInterval(1e-6, Eg,        0.25)

    i1 = MatterExpInterval(1e-6, 0.4, 0.1)
    i2 = MatterExpInterval(3e-7, 0.,  0.067)
    i3 = MatterExpInterval(1e-6, 0.4, 0.1)
    ms = MatterStack(i1, i2, i3)
    # ms = MatterStack(i1, i2, i3, i4, i5)
    # ms = MatterStack(i1, i2, i3, i4, i5, i6, i7)
    # ms = MatterStack(i1, i2, i3, i4, i5, i6, i7, i8, i9)
    print ms
    # print ms.get_transfer_matrix(0, 0.2)
    # print ms.get_transfer_matrix(1, 0.2)
    # print ms.get_transfer_matrix(2, 0.2)
    # ms.plot_wells_lorenzians()
    # for i in range(2,9,2):
    #     ee = erg2eV(ms.get_eigen_energy(i))
    #     print "Energy eigenvalues in " + str(i) + "th interval:", ee
    # ms.plot_wells_lorenzians()
    # print ms.get_interval_object(9, eV2erg(0.39753244))
    ms.plot_states()
    print ms.get_first_interval_object(eV2erg(0.204838642308))