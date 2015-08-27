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
CALC_STEPS_NUM = 3e3
CALC_STEP = CALC_STEPS_NUM ** -2
FAKE_ZERO = np.finfo(np.float).tiny

VERBOSE = False


def m2prod(*mxarray):
    '''If input is array of matrices 2x2:
        [m1, m2, m3, m4, ...]
    m2prod returns their production m1 * m2 * m3 * m4
    '''
    m = np.matrix(np.eye(2))
    for mi in mxarray: 
        m = m.dot(mi)
    return m

def m2prod_test(n):
    e = np.matrix(np.eye(2))
    rand = np.random.rand
    mxarr = [np.matrix([ [rand(), rand()] , [rand(), rand()] ] ) for i in range(n)]
    def revmul(index):
        i = index
        m = e
        while i >= 0:
            m = mxarr[-1-i] * m
            i -= 1
        return m
    print "Matrices array:" 
    print "\n\n".join([str(m) for m in mxarr])
    for i in range(len(mxarr)):
        print "\n"
        print "Reverse multiplication (" + str(i) + "):"
        print revmul(i)
        print "\n"
        print "Reverse m2prod (" + str(i) + "):"
        print "Range for m2prod", range(len(mxarr)-1, len(mxarr)-i-1, -1)
        print m2prod(*[mxarr[k] for k in range(len(mxarr)-1, len(mxarr)-i-2, -1)])



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
    
    def reverse(self):
        self.v_start, self.v_end = self.v_end, self.v_start 
        self.angle *= -1



class IntervalStack(object):
    """Class for stack of potential intervals"""
    def __init__(self, intervals):
        self.data = []
        self.dots = []
        for i in intervals: 
            self.__append(i)
        self.__process()

    def __len__(self):
        return len(self.data)

    def __append(self, interval):
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
        if width > 0.1: 
            width *= 1e-7 # Converting cm to nm
        self.height_ev = height
        self.height = eV2erg(self.height_ev) # In ergs
        self.label = ''
        super(MatterExpInterval, self).__init__(width=width, angle=0.0, v_start=self.height, mass=mass)



class MatterStack(IntervalStack):
    """docstring for MatterStack"""
    def __init__(self, intervals, showfict=False): #, ficticious_width=1e-3, ficticious_mass=0.25):
        super(MatterStack, self).__init__(intervals)
        # self.data = []
        self.showfict = showfict
        # for i in intervals: 
        #     self.__append(i)
        self.__process()
        self.eigen_energy = []

    def __append(self, interval):
        if isinstance(interval, MatterExpInterval):
            self.data.append(interval)

    def __process(self):
        # Moving common potential minimum to zero
        vmin = min([interval.height for interval in self.data])
        for interval in self.data:
            interval.height -= vmin
            interval.height_ev = erg2eV(interval.height)
        # Adding ficticious interval
        minmass = self.data[0].mass
        for interval in self.data:
            if interval.height == 0:
                minmass = interval.mass
                break
        if self.showfict:
            self.data.insert(0, MatterExpInterval(self.data[0].width, 0, minmass)) # Ficticious region
        else:
            self.data.insert(0, MatterExpInterval(0, 0, minmass))
        if len(self.data) == 0: 
            return
        coordinate = 0
        for i in range(len(self.data)-1):
            coordinate += self.data[i].width
            self.dots.append({
                'height': self.data[i+1].height - self.data[i].height,
                'height_ev': erg2eV(self.data[i+1].height - self.data[i].height),
                'coordinate': coordinate
                })
        for i in range(1,len(self.data)-1):
            if self.data[i-1].height > self.data[i].height < self.data[i+1].height:
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
                coordstr = "[0, " + str(self.data[i].width * 1e7) + "] nm"
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
        if index == 0: 
            # return np.complex(0)
            return -self.data[0].width * self.get_wave_number(0, energy) # by Evropetcev
        if index == 1: 
            return np.complex(0)
        return self.get_wave_number(index, energy) * np.sum([interval.width for interval in self.data[1:index]])

    def get_transfer_matrix_wo_mass(self, index, energy):
        wnsum = self.get_wave_number(index, energy) + self.get_wave_number(index+1, energy)
        r = (self.get_wave_number(index, energy) - self.get_wave_number(index+1, energy)) / wnsum
        t = 2 * self.get_wave_number(index, energy) / wnsum
        d = self.get_wave_number(index, energy) * self.data[index].width
        return 1 / t * np.matrix([[    np.exp(-1j * d), r * np.exp(-1j * d)],
                                  [r * np.exp( 1j * d),     np.exp( 1j * d)] ])

    def get_transfer_matrix_w_mass(self, index, energy):
        if energy == self.data[index].height:
            km_ratio = np.sqrt(self.data[ index ].mass * np.complex(energy - self.data[index+1].height) / \
                               self.data[index+1].mass / FAKE_ZERO)
        else:
            km_ratio = np.sqrt(self.data[ index ].mass * np.complex(energy - self.data[index+1].height) / \
                               self.data[index+1].mass / np.complex(energy - self.data[ index ].height)) 
        kcp = 1 + km_ratio
        kcm = 1 - km_ratio
        kd = self.get_wave_number(index, energy) * self.data[index].width
        return 0.5 * np.matrix([[kcp * np.exp(-1j * kd), kcm * np.exp(-1j * kd)],
                                [kcm * np.exp( 1j * kd), kcp * np.exp( 1j * kd)] ])

    def get_transfer_matrix(self, index, energy):
        # return self.get_transfer_matrix_wo_mass(index, energy)
        return self.get_transfer_matrix_w_mass(index, energy)

    def get_interval_object(self, index, energy):
        obj = {'energy': energy, 'energy_ev': erg2eV(energy)}
        uvec = m2prod(*[ self.get_transfer_matrix(i, energy) for i in range(index, len(self.data)-1) ]) \
                                                                                * np.matrix([[1], [0]])
        if self.showfict:
            obj['up'] = uvec[0, 0]
        elif index == 1:
            obj['up'] = np.complex(0)
        else:
            obj['up'] = uvec[0, 0]
        # obj['up'] = uvec[0, 0]
        obj['un'] = uvec[1, 0]
        obj['delta'] = self.get_delta(index, energy)
        obj['wavenum'] = self.get_wave_number(index, energy)
        return obj

    def get_first_interval_object(self, energy):
        return self.get_interval_object(0, energy)
        
    def get_energy_range(self, energy_min=None, energy_max=None, stepsnum=CALC_STEPS_NUM):
        if energy_min == None:
            energy_min = 0
        if energy_max == None: 
            energy_max = sorted([i.height for i in self.data])[-2] # Taking second highest barrier
        return np.linspace(energy_min, energy_max, stepsnum)

    def get_lorenzian_ratio(self, index, energy):
        return (np.abs(self.get_interval_object(index, energy)['up']) / \
                np.abs(self.get_first_interval_object(energy)['up'])) ** 2

    def get_lorenzian_ratio_derivative(self, index, energy, step=CALC_TOLERANCE):
        return (self.get_lorenzian_ratio(index, energy + step) \
              - self.get_lorenzian_ratio(index, energy - step)) / 2 / step

    def get_lorenzian_ratio_derivative2(self, index, energy, step=CALC_TOLERANCE):
        return (self.get_lorenzian_ratio_derivative(index, energy + step) \
              - self.get_lorenzian_ratio_derivative(index, energy - step)) / 2 / step

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
        # if not self.eigen_energy.tolist() == []:
            return self.eigen_energy
        eigen_energy = []
        for index in range(len(self.data)):
            # if self.data[index].label not in ('well', 'stair'): # Stair layers bring bullshit
            if self.data[index].label not in ('well'):
                continue
            for energy in self.get_interval_eigen_energy(index):
                if energy not in eigen_energy:
                    eigen_energy.append(energy)
        self.eigen_energy = sorted(eigen_energy)
        return self.eigen_energy

    def get_peak_energy_by_bisections(self, index, energy_min, energy_max, tol=CALC_TOLERANCE):
        '''Bisection method'''
        energy_mid = (energy_min + energy_max) / 2
        if VERBOSE:
            print "Starting mid energy =", erg2eV(energy_mid)
        func = self.get_lorenzian_ratio_derivative
        f = func(index, energy_mid)
        while np.abs(f) > tol:
            if np.sign(func(index, energy_min)) * np.sign(f) > 0:
                energy_min = energy_mid
            elif np.sign(func(index, energy_max)) * np.sign(f) > 0:
                energy_max = energy_mid
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
        '''Newton-Raphson method
        Doesn't work properly'''
        energy_mid = (energy_min + energy_max) / 2
        if VERBOSE:
            print "Starting mid energy =", erg2eV(energy_mid)
        func  = self.get_lorenzian_ratio_derivative
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
            toNm = 1e7
            if self.data[i].label not in ('well', 'stair'): 
                continue
            axlin.plot( erg2eV(energy_range), spectra[i], 
                label = self.data[i].label + ' \#' + str(i) + " %.1f nm" % (self.data[i].width * toNm))
            axlogy.semilogy( erg2eV(energy_range), spectra[i], 
                label = self.data[i].label + ' \#' + str(i) + " %.1f nm" % (self.data[i].width * toNm))
        axlin.legend(loc=2) 
        axlogy.legend(loc=2)
        pl.show()

    def get_stack_profile(self):
        offset = -self.data[0].width
        x, y = [], []
        for interval in self.data:
            x.append(offset)                    # Interval begins
            y.append(interval.height_ev)
            x.append(interval.width + offset)   # Interval ends
            y.append(interval.height_ev)
            offset += interval.width
        return (np.array(x), np.array(y))

    def get_wavefunction(self, energy, stepsnum=CALC_STEPS_NUM):
        if not energy in self.eigen_energy:
            print " * * * WARNING! Getting wavefunction of unexisting energy eigen value! * * *"
        # An interval obj = {'energy':, 'energy_ev':, 'up':, 'un':, 'delta':, 'wavenum'}
        offset = -self.data[0].width # Width of the ficticious interval, x is negative there
        step = np.sum([interval.width for interval in self.data]) / stepsnum
        x, y = [], []
        ymax = []
        for i in range(len(self.data)):
            interval = self.get_interval_object(i, energy)
            xloc = np.arange(offset, self.data[i].width + offset, step)
            yloc = interval['up'] * np.exp( 1j * (interval['wavenum'] * xloc - interval['delta'])) + \
                   interval['un'] * np.exp(-1j * (interval['wavenum'] * xloc - interval['delta'])) 
            if self.data[i].label in ('well') and i > 1:
                ymax.append(np.max(np.abs(yloc)))
            x = np.concatenate((x, xloc))
            y = np.concatenate((y, yloc))
            offset += self.data[i].width
        y = y / np.max(ymax)
        return (np.array(x), np.array(y))

    def plot_states(self, nwf=None, xmin=None, xmax=None, ymin=None, ymax=None):
        toNm = 1e7
        self.eigen_energy = self.get_eigen_energy()
        if nwf == None or nwf > len(self.eigen_energy):
            nwf = len(self.eigen_energy)
        fig = pl.figure()
        fig.patch.set_facecolor('white')
        pl.grid(True)
        
        bandx, bandy = self.get_stack_profile()
        vmin = np.min(bandy)
        vmax = np.max(bandy)
        dv = vmax - vmin
        
        pl.plot(bandx * toNm, bandy, lw=2, color='0.85')

        # Plot Y limits
        if ymax == None:
            ymax = vmax + dv/3
            pl.ylim(ymax=ymax)
        elif ymax == 'auto':
            pass
        else:
            pl.ylim(ymax=ymax)
        if ymin == None:
            ymin = vmin - dv/8
            pl.ylim(ymin=ymin)
        elif ymin == 'auto':
            pass
        else:
            pl.ylim(ymin=ymin)
        
        pl.fill_between(bandx * toNm, bandy, y2=-2 * np.abs(ymin), facecolor='0.95', color='0.95')
        
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
        hatch = [ '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*' ]
        # hatch = [ '/', '\\', '|', 'o', '+', 'x', 'O', '.', '*' ]
        for i in range(nwf):
            if i == len(colors):
                colors *= 2
            if i == len(hatch):
                hatch *= 2

            energy = self.eigen_energy[i]

            x, y = self.get_wavefunction(energy)
            x *= toNm
            # y = self.reduce_wavefunction(y, dv/nwf) + erg2eV(energy)
            energy_ev = erg2eV(energy)
            y = y * dv/(nwf + 1) + energy_ev

            xlvl = [x[0], x[-1]]
            ylvl = [erg2eV(energy)] * 2

            pl.plot(x   , y,    colors[i], lw=2, alpha=0.65, label="%.3f meV" % (energy_ev * 1e3))
            pl.plot(xlvl, ylvl, colors[i], lw=1, alpha=0.5)
            pl.fill_between(x, y, y2=energy_ev, color=colors[i], facecolor=colors[i], 
                                                     alpha='0.05', hatch=hatch[i])
        

        # Plot X limits
        fullwidth = np.sum([interval.width for interval in self.data[1:]])
        activereg = [interval.width for interval in self.data[1:] if interval.label in ('well', 'stair')]
        averageqw = np.sum(activereg) / len(activereg)

        if xmin != None:
            pl.xlim(xmin=xmin)
        elif self.showfict:
            pl.xlim(xmin=-ms.data[1].width * toNm)
        else:
            pl.xlim(xmin = (self.data[1].width - 5 * averageqw) * toNm)
        if xmax != None:
            pl.xlim(xmax=xmax)
        else:
            pl.xlim(xmax = (fullwidth - self.data[-1].width + 5 * averageqw) * toNm)

        # Legend
        # reverse the order
        ax = pl.gca()
        handles, labels = ax.get_legend_handles_labels()
        pl.legend(handles[::-1], labels[::-1])
        pl.ylabel('Energy, eV')
        pl.xlabel('Coordinate, nm')
        pl.show()

    def reduce_wavefunction(self, y, value, dst=5):
        extrm = []
        for i in range(dst, len(y) - dst):
            y1 = np.abs(y[i-dst])
            y2 = np.abs(y[i+dst])
            if np.abs(y[i]) >  max(y1, y2) and (y[i-dst+1]-y1) * (y2-y[i+dst-1]) < 0:
                extrm.append(y[i])
        if extrm == []:
            return y * value / np.max(np.abs(y))
        return y * value / np.max(np.abs(extrm)) # avoiding exponential tails



if __name__ == '__main__':

    # m2prod_test(4)
    # import sys
    # sys.exit(0)
    Eg = 4.0
    dEg = 0.1
    qww = 2e-7
    qwb = qww / 2
    b1w = qww * 1e2
    b2w = b1w
    i1 = MatterExpInterval(b1w, Eg,         0.20)
    i2 = MatterExpInterval(qww, Eg - dEg,   0.25) # Well
    i8 = MatterExpInterval(qww, Eg - dEg/2, 0.25) # Stair
    i3 = MatterExpInterval(qwb, Eg,         0.20)
    i5 = MatterExpInterval(qww, Eg - dEg,   0.25) # Well
    i9 = MatterExpInterval(qww, Eg - dEg/3, 0.25) # Stair
    i6 = MatterExpInterval(qwb, Eg,         0.20)
    i7 = MatterExpInterval(qww, Eg - dEg,   0.25) # Well
    i4 = MatterExpInterval(b2w, Eg,         0.20)


    # i1 = MatterExpInterval(2e-6, 0.334,  0.1002)
    # i2 = MatterExpInterval(4e-7, 0,       0.067) # Well
    # i3 = MatterExpInterval(2e-6, 0.334,  0.1002)

    # ms = MatterStack([i1, i2, i3])
    # ms = MatterStack([i1, i2, i3, i4, i5])
    # ms = MatterStack([i1, i2, i3, i2, i3, i2, i5])
    # ms = MatterStack([i1, i2, i3, i2, i3, i2, i3, i2, i1])
    ms = MatterStack([i1, i2, i8, i3, i5, i9, i6, i7, i4])
    print "\n".join([str(erg2eV(e)) + " eV" for e in ms.get_eigen_energy()])
    print ms
    # ms.plot_wells_lorenzians()
    ms.plot_states()


