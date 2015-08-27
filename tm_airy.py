#!/usr/bin/env python
# coding: utf-8
#
# E. Shevchenko
# shevchenko@beam.ioffe.ru
# 

import numpy as np
import pylab as pl
from scipy.special import airy
import matplotlib

from material import *
from tm import *

CALC_TOLERANCE = 1e-16
CALC_STEPS_NUM = 1e3
CALC_STEP = 1e-18

# VERBOSE = True
VERBOSE = False



class MatterUniformFieldInterval(Interval):
    """docstring for MatterUniformFieldInterval"""
    def __init__(self, width=1, field=1e4, v_start=0, mass=0.2):
        self.v_start_ev = v_start               # eV
        self.field_Vpcm = field                 # V/cm
        self.field = Vpcm2CGS(self.field_Vpcm)  # CGS: g**1/2 * cm**-1/2 * sec**-1
        if width > 0.1: width *= 1e-7           # Converting from nm to cm
        super(MatterUniformFieldInterval, self).__init__(width, self.field * E, eV2erg(v_start), mass)
        self.v_end_ev = erg2eV(self.v_end)
        self.label = ''
        self.v_mid = (self.v_start + self.v_end) / 2
        self.v_mid_ev = erg2eV(self.v_mid)

    def get_z(self, energy, x):
        return np.power(2 * self.mass * ME, 1./3) / \
               np.power(np.power(self.angle * HBAR, 2), 1./3) * \
               (self.v_start - energy + self.angle * x)

    def reverse(self):
        self.v_start, self.v_end = self.v_end, self.v_start
        self.v_end_ev = erg2eV(self.v_end)
        self.v_start_ev = erg2eV(self.v_start)
        self.field *= -1
        self.field_Vpcm *= 1
        self.angle *= -1

    def __repr__(self):
        return "Width: % .1f nm\t\tPotentials: [% .3f, % .3f] eV\t\tField: % .2e V/cm\n" % \
                            (self.width*1e7, self.v_start_ev, self.v_end_ev, self.field_Vpcm)






class MatterAiryStack(IntervalStack):
    """docstring for MatterAiryStack"""
    def __init__(self, intervals):
        super(MatterAiryStack, self).__init__(intervals)
        self.eigen_energy = []
        self.eigen_energy_ev = []
        self.c1 = []
        self.states = []
        self.lifetimes = []
        self.direction = 1
        self.__process()

    def set_direction(self, d):
        self.direction = d

    # def __append(self, *intervals):
    #     for interval in intervals:
    #         if isinstance(interval, MatterUniformFieldInterval):
    #             self.data.append(interval)

    def __process(self):
        # Moving common potential minimum to zero
        vmin = min([min(ivl.v_start, ivl.v_end) for ivl in self.data]) 
        for interval in self.data:
            interval.v_start -= vmin
            interval.v_end   -= vmin
            interval.v_start_ev = erg2eV(interval.v_start)
            interval.v_end_ev   = erg2eV(interval.v_end)
        
        if self.data[0].field < 0 and self.data[-1].field < 0:
            self.__reverse()

        if self.data[0].v_end > self.data[1].v_start: 
            self.data[0].label = 'barrier'
        else:
            self.data[0].label = ''

        if self.data[-1].v_start > self.data[-2].v_end: 
            self.data[-1].label = 'barrier'
        else:
            self.data[-1].label = ''

        for i in range(1,len(self.data)-1):
            if self.data[i].v_start < self.data[i-1].v_end and self.data[i].v_end < self.data[i+1].v_start:
                self.data[i].label = 'well'
            elif self.data[i].v_start < self.data[i-1].v_end or self.data[i].v_end < self.data[i+1].v_start:
                self.data[i].label = 'stair'
            elif self.data[i].v_start > self.data[i-1].v_end and self.data[i].v_end > self.data[i+1].v_start:
                self.data[i].label = 'barrier'
            else:
                self.data[i].label = 'extension'

    def __reverse(self):
        self.data.reverse()
        self.direction = -1
        for interval in self.data:
            interval.reverse()

    def __checkdirection(self, d):
        isdirect = self.data[0].field > 0 and self.data[-1].field > 0
        if isdirect and d < 0:
            self.data.reverse()
            self.direction = -1
            for interval in self.data:
                interval.reverse()
        elif not isdirect and d > 0:
            self.data.reverse()
            self.direction = 1
            for interval in self.data:
                interval.reverse()

    def get_transfer_matrix(self, index, energy):
        '''Returns Numpy matrix of the chosen energy level and interface index'''
        self.__process()
        # if VERBOSE: print "Getting transfer matrix for index", index, "and energy", erg2eV(energy)
        lr1 = self.data[index]
        lr2 = self.data[index+1]
        # print "Z_1_end:", lr1.get_z(energy, lr1.width), "Z_2_start:", lr2.get_z(energy, 0)
        old_settings = np.seterr(over='ignore')
        (a1, ad1, b1, bd1) = airy( lr1.get_z(energy, lr1.width) )
        (a2, ad2, b2, bd2) = airy( lr2.get_z(energy, 0) )
        # print "UNDER POWER", lr2.mass * lr2.angle / lr1.mass / lr1.angle
        msu = lr2.mass * lr2.angle / lr1.mass / lr1.angle
        ms = np.sign(msu) * np.power(np.abs(msu), 1./3)
        p = np.pi * (a2 * bd1 - ms * ad2 * b1)
        q = np.pi * (b2 * bd1 - ms * bd2 * b1)
        r = np.pi * (ms * ad2 * a1 - a2 * ad1)
        s = np.pi * (ms * bd2 * a1 - b2 * ad1)
        np.seterr(**old_settings)
        return np.matrix([[p, q], [r, s]])

    def get_c_d(self, mx):
        return tuple([np.float(n[0]) for n in (mx * np.matrix( [[1],[0]] )).tolist()])

    def get_interval_obj(self, index, energy):
        int_obj = {'tms': [], 'energy': energy, 'energy_ev': erg2eV(energy)}
        int_obj['tms'] = [self.get_transfer_matrix(i, energy) for i in range(index, len(self.data)-1)]
        mx = m2prod(*int_obj['tms'])
        (int_obj['c'], int_obj['d']) = self.get_c_d(mx)
        return int_obj

    def get_first_interval_obj(self, energy, diff_step=CALC_TOLERANCE):
        int1_obj = self.get_interval_obj(0, energy)
        int1_obj['c_d'] = int1_obj['c']
        int1_obj['d_d'] = int1_obj['d']
        while int1_obj['c_d'] == int1_obj['c']:
            (int1_obj['c_d'], int1_obj['d_d']) = self.get_c_d( m2prod(
                *[self.get_transfer_matrix(i, energy + diff_step) for i in range(len(self.data)-1)]))
            diff_step *= 2
        # Derivatives of C and D in the first interval
        int1_obj['c_d'] = (int1_obj['c_d'] - int1_obj['c']) / diff_step
        int1_obj['d_d'] = (int1_obj['d_d'] - int1_obj['d']) / diff_step
        return int1_obj

    def get_eigen_states(self):
        if not self.states == []: 
            return self.states
        if VERBOSE: 
            print "Entering eigen states calculation function..."
        energy_min = 0
        energy_max = max([nrg for nrg in [min(i.v_start, i.v_end) for i in self.data]])
        # energy_max = max([nrg for nrg in [min(i.v_start, i.v_end) 
        #     for i in self.data if i.label == 'barrier']])
        if VERBOSE: 
            print "Emax =", erg2eV(energy_max)
        energy_range = np.linspace(energy_min, energy_max, CALC_STEPS_NUM)
        self.c1 = np.array([self.get_first_interval_obj(e)['c'] for e in energy_range])
        clim = min(np.abs(np.min(self.c1)), np.abs(np.max(self.c1)))
        # pl.plot(erg2eV(energy_range), np.array(self.c1))
        # pl.ylim(-clim,clim)
        # pl.grid()
        # pl.show()
        for i in range(int(CALC_STEPS_NUM-1)):
            if self.c1[i] == np.nan: continue
            # elif np.abs(self.c1[i]) == np.inf: continue 
            elif self.c1[i+1] == np.nan: continue
            # elif np.abs(self.c1[i+1]) == np.inf: continue
            if np.sign(self.c1[i]) * np.sign(self.c1[i+1]) < 0:
                self.eigen_energy.append(self.get_eigen_values(energy_range[i], energy_range[i+1]))
        self.eigen_energy_ev = [erg2eV(e) for e in self.eigen_energy]
        self.states = [self.get_first_interval_obj(e) for e in self.eigen_energy]
        return self.states

    def get_eigen_energy(self):
        if not self.eigen_energy == []: 
            return self.eigen_energy
        self.get_eigen_states()
        return self.eigen_energy

    def get_eigen_energy_ev(self):
        if not self.eigen_energy_ev == []: 
            return self.eigen_energy_ev
        self.get_eigen_states()
        return self.eigen_energy_ev

    def get_eigen_values(self, energy_min, energy_max, tol=CALC_TOLERANCE):
        if VERBOSE: print "Calculating eigen value with bisection"
        energy_mid = (energy_min + energy_max) / 2
        while np.abs(self.get_first_interval_obj(energy_mid)['c']) >= tol:
            energy_mid = (energy_min + energy_max) / 2
            if VERBOSE: 
                print "Energy Min:", erg2eV(energy_min), "Mid:", erg2eV(energy_mid), "Max:", erg2eV(energy_max) 
                print "C         :", self.get_first_interval_obj(energy_min)['c'],
                print "     ", self.get_first_interval_obj(energy_mid)['c'],
                print "    ", self.get_first_interval_obj(energy_max)['c']
            if self.get_first_interval_obj(energy_mid)['c'] == 0: 
                return energy_mid
            elif np.sign(self.get_first_interval_obj(energy_mid)['c']) == \
                 np.sign(self.get_first_interval_obj(energy_min)['c']):
                energy_min = energy_mid
            else:
                energy_max = energy_mid
            if energy_mid == (energy_min + energy_max) / 2: # Sometimes the accuracy of float64 is not enough 
                if VERBOSE: print "returning", erg2eV(energy_mid), "eV -- extremely fast zero"
                return energy_mid
        if VERBOSE: print "returning", erg2eV(energy_mid), "eV"
        return energy_mid

    def get_potential_profile(self):
        """Returns tuple of two numpy arrays (np.array(x), np.array(y)) for plotting the band structure"""
        if self.direction == -1:
            self.__reverse()
        offset = 0
        x, y = [], []
        for interval in self.data:
            x.append(offset)
            y.append(interval.v_start_ev)
            x.append(interval.width + offset)
            y.append(interval.v_end_ev)
            offset += interval.width
        if self.direction == -1:
            self.__reverse()
        return (np.array(x), np.array(y))

    def get_wavefunction(self, index, dots_num=CALC_STEPS_NUM):
        if self.states == []:
            self.states = self.get_eigen_states()
        wf_obj = self.states[index] # {'energy':, tms: [mx mx mx], 'c':, 'd': }
        if VERBOSE: print "Getting wavefunction for state", index, "with energy", erg2eV(wf_obj['energy'])
        dot_step = sum([interval.width for interval in self.data]) / dots_num
        x, y = [], []
        offset = 0
        for i in range(len(self.data)):
            # Getting wavefunction coefficients for the i-th region
            if i == 0:
                c, d = wf_obj['c'], wf_obj['d']
            elif i == len(self.data)-1:
                c, d = 1, 0
            else:
                c, d = self.get_c_d( m2prod( *wf_obj['tms'][i:] ))
            # Local read coordinate for the i-th region
            xloc = np.arange(0, self.data[i].width, dot_step)
            # Reduced coordinate, the argument of Airy wavefunction, for the i-th region
            z = np.array( [self.data[i].get_z(wf_obj['energy'], dx) for dx in xloc] )
            (Ai, Aid, Bi, Bid) = airy(z)
            # Moving i-th x to its place
            x = np.concatenate((x, xloc + offset))
            offset += self.data[i].width
            y = np.concatenate((y, c*Ai + d*Bi))
            if VERBOSE:
                print "Layer", str(i) + ": ", "c =", c, "\t", "d =", d
        y = self.normilized(dot_step, y)
        # return (x, y + erg2eV(wf_obj['energy']))
        if self.direction == -1:
            x = x[::-1]
        return (x, y)

    def normilized(self, step, y):
        S = 0
        imin = self.data[0].width / step
        imax = self.data[-1].width / step
        for i in range(int(imin),len(y)-1):
            S += step * (y[i]**2 + y[i+1]**2) / 2   # y * dx
        if VERBOSE: print "S is", S
        # divider = 3e4 # Magic number, the empirical value
        # y = y / np.power(S, 0.5)
        # vmin = min([min(ivl.v_end_ev, ivl.v_start_ev) for ivl in self.data])
        # vmax = max([max(ivl.v_end_ev, ivl.v_start_ev) for ivl in self.data])
        # dv = vmax - vmin
        # while np.max(y[imin : -imax]) - np.min(y[imin : -imax]) > dv/4:
        #     divider *= 2
        #     y /= 2
        # if VERBOSE: print "Divider is", divider
        y = y / np.power(S, 0.5)
        if VERBOSE: print "Wavefunction maximum:", np.max(y)
        return y

    def plot(self, nwf=10, xmin=None, xmax=None, ymin=None, ymax=None):
        """Plotting the structure with chosen bumber of wavefuntion"""
        divider = 3e4
        toNm = 1e7
        (band_x, band_y) = self.get_potential_profile()
        pl.figure()
        pl.plot(band_x * toNm, band_y, color='0.65', linewidth=3)
        if len(self.states) < nwf: 
            nwf = len(self.states)
        for i in range(nwf):
            (x, y) = self.get_wavefunction(i)
            xlevel = np.array([x[0], x[-1]])
            pl.plot(xlevel * toNm, np.array([self.states[i]['energy_ev']]*2))#, color=colors[i]) # level line
            pl.plot(x * toNm, y / divider + self.states[i]['energy_ev'], linewidth=2) #, color=colors[i]) # wavefuntion
        vmin = min([min(ivl.v_end_ev, ivl.v_start_ev) for ivl in self.data])
        vmax = max([max(ivl.v_end_ev, ivl.v_start_ev) for ivl in self.data])
        dv = vmax - vmin
        if xmin != None: 
            pl.xlim(xmin=xmin)
        if xmax != None: 
            pl.xlim(xmax=xmax)
        if ymin != None: 
            pl.ylim(ymin=ymin)
        else: 
            pl.ylim(ymin = vmin - 0.1 * dv)
        if ymax != None: 
            pl.ylim(ymax=ymax)
        else: 
            pl.ylim(ymax = vmax + 0.2 * dv)
        pl.grid()
        pl.show()

    def get_lifetime(self, index):
        """Lifetime of a particle in the state of the passed index"""
        if self.states == []: self.states = self.get_eigen_states()
        state_obj = self.states[index]
        gamma = state_obj['d'] / state_obj['c_d']
        if VERBOSE:
            print "C1' =", state_obj['c_d']
            print "D1 =", state_obj['d']
            print "Gamma =", gamma
            print "Mass =", self.data[index].mass * ME
            print "Planck const =", HBAR
            print "Lifetime =", self.data[index].mass * ME / gamma / HBAR
        return self.data[index].mass * ME / gamma / HBAR

    def get_lifetimes(self):
        """Lifetimes array of a particle in the eigen states"""
        if self.states == []: self.states = self.get_eigen_states()
        if not self.lifetimes == []: return self.lifetimes
        for i in range(len(self.states)):
            t = self.get_lifetime(i)
            self.lifetimes.append(t)
            self.states[i]['lifetime'] = t
        return self.lifetimes




if __name__ == '__main__':
    
    Eg = 4.05
    dEg = 0.05

    # int1 = Interval(10, 2.5)
    # print int1
    # int2 = Interval(10, -5, int1.v_end - dEg)
    # print int2
    # int3 = Interval(10, 2, int2.v_end + dEg)
    # print int3

    # stack = IntervalStack(int1, int2, int3)
    # print stack
    
    F = 2e+6
    Fb = F/15

    regions = [MatterUniformFieldInterval(10e-7, Fb, Eg, 0.25)]
    # for i in range(0,8,2):
    #     regions.append(MatterUniformFieldInterval(2e-7,  -F, regions[i].v_end_ev - dEg, 0.15))
    #     regions.append(MatterUniformFieldInterval(2e-7, Fb, regions[i+1].v_end_ev + dEg, 0.25))
    for i in range(0,3,3):
        regions.append(MatterUniformFieldInterval(2e-7,  -F, regions[i].v_end_ev - dEg, 0.15))
        regions.append(MatterUniformFieldInterval(3e-7,  Fb*2, regions[i+1].v_end_ev + dEg/4, 0.18))
        regions.append(MatterUniformFieldInterval(3e-7,   Fb, regions[i+2].v_end_ev + dEg*3/4, 0.25))
    regions.append(MatterUniformFieldInterval(6e-7, Fb, regions[-1].v_end_ev, 0.25))

    mstack = MatterAiryStack(*regions)

    
    print "\n"
    print "Eigen energy (eV):"
    print "\n".join(["%.4f eV" % e for e in mstack.get_eigen_energy_ev()])
    print "States lifetimes:"
    print "\n".join(["%.2e nsec" % t for t in mstack.get_lifetimes()])
    print "Number of eigen states:", len(mstack.eigen_energy)
    print 
    mstack.plot()