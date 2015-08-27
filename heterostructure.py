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
from material import *
from tm_airy import *
import numpy as np
from scipy.integrate import simps



# VERBOSE = True
VERBOSE = False




class Heterostructure(object):
    """docstring for heterostructure"""
    def __init__(self, arg):
        super(Heterostructure, self).__init__()
        self.layers = []
        self.interface_a = []
        self.field = []
        self.states = []
        self.potential_profile = []
        for layer in arg:
            self.layers.append(layer)
        self.__process()

    def __process(self):
        for i in range(1,len(self.layers)):
            lr_prev = self.layers[i-1]
            lr_this = self.layers[i]
            lr_this.prev_lr = lr_prev
            lr_prev.next_lr = lr_this

    def __len__(self):
        return len(self.layers)

    def get_electric_field(self):
        '''Returns array of built-in electric fields in the heterostructure layers
        
        See Smith, Mailhiot, J. Appl. Phys. 63, 2717, (1988)
        
        Equations system in CGS:
        D = eps*F = F + 4*pi*P
        D1 = D2 => F1 - F2 = 4*pi*(P2 - P1)
        
        Equations system in SI:
        D = e0*eps*F + P
        D1 = D2 => eps1*F1 - eps2*F2 = (P2 - P1) / e0
        
        Thus coefficient (say K) is either 4*pi for CGS or VACUUM_DIEL**-1 for SI system
        L1 * eps1 * F1 + L2 * eps2 * F2 + ... = 0
        
        A * F = b
        F = A \ b
        Exapmle for 4 layers (e* are dielectric constants):
        
                | e1     -e2      0       0     |   |F1|     |P2 - P1|
        A * F = | 0       e2     -e3      0     | * |F2|  =  |P3 - P2| * K
                | 0       0       e3     -e4    |   |F3|     |P4 - P3|
                | e1*L1   e2*L2   e3*L3   e4*L4 |   |F4|     |   0   |
        
        Here we get V/cm for the SI system
        F = np.linalg.solve(A, b) / 1e2 '''
        if not self.field == []:
            return self.field

        msize = len(self.layers)
        A = np.matrix([[0.] * msize] * msize)
        polar_vec = np.matrix([[0.]] * msize)
        for i in range(msize):
            if i < msize - 1:
                A[i, i] = self.layers[i].diel_st
                A[i, i+1] = - self.layers[i+1].diel_st
                polar_vec[i] = (self.layers[i+1].get_polar_tot() - self.layers[i].get_polar_tot()) / VACUUM_DIEL
            A[msize-1, i] = self.layers[i].diel_st * self.layers[i].width
        # Electric field in HS layers, V/cm
        field = np.linalg.solve(A, polar_vec) / 1e2
        self.field = field.T.tolist()[0]

        # Printing values of interim calculations
        if VERBOSE:
            print 'Equilibrium lattice constant a, A:'
            print '\t'.join(["%f" % i.a_eq for i in self.layers]), '\n'
            print 'Interface lattice constant a, A:'
            print '\t'.join(["%f" % i.a for i in self.layers]), '\n'
            print 'Shear modulus:'
            print '\t'.join(["%f" % i.shear for i in self.layers]), '\n'
            print 'Strain tensor xx components:'
            print '\t'.join(["%.2e" % i.strain_xx for i in self.layers]), '\n'
            print 'Strain tensor zz components:'
            print '\t'.join(["%.2e" % i.strain_zz for i in self.layers]), '\n'
            print 'Spontaneous polarization, C/m**2:'
            print '\t'.join(["%f" % i.polar_sp for i in self.layers]), '\n'
            print 'Piezoelectric polarization, C/m**2:'
            print '\t'.join(["%f" % i.polar_pz for i in self.layers]), '\n'
            print 'Total polarization, C/m**2:'
            print '\t'.join(["%f" % i.polar_tot for i in self.layers]), '\n'
            print 'Matrix of equations system:\n', A, '\n'
            print 'Right part polarization vector, V/m:'
            print polar_vec * VACUUM_DIEL, "\n"
            print 'Electric field, V/cm:\n',              \
                    '\t'.join(["%.2e" % i for i in self.field]),    '\n'

        return self.field

    def get_bands(self):
        gaps = [lr.energy_gap for lr in self.layers]
        mingap = min(gaps)
        band_e = (np.array(gaps) - mingap) * 0.75
        band_h = (np.array(gaps) - mingap) * 0.25
        # print "GAPS:", gaps
        # print "Minimum gap is", mingap
        # print "Band for electrons:", band_e
        # print "Band for holes:", band_h
        return (band_e, band_h)

    def get_wavefunctions_airy(self, el=True, lh=True, hh=True):
        '''Getting objects in a list
            [{
                'el_energy': energy_value,
                'el_wavefunction': (x,y),
                'lh_energy': energy_value,
                'lh_wavefunction': (x,y),
                'hh_energy': energy_value,
                'hh_wavefunction': (x,y),
            }, ...]'''
        if not self.states == [] and not self.potential_profile == []:
            return self.states, self.potential_profile 
        band_e, band_h = self.get_bands()
        self.field = self.get_electric_field()
        potential_profile = {'el': None, 'hh': None, 'lh': None}
        states = []

        def get_the_values(band, mass, mark=''):
            if mark in ('lh', 'hh'):
                mul = -1 # Field correction for holes
            else:
                mul = 1
            regions = [MatterUniformFieldInterval(a2cm(self.layers[0].width), self.field[0] * mul, band[0], mass[0])]
            for i in range(1,len(self)):
                regions.append(MatterUniformFieldInterval(a2cm(self.layers[i].width), self.field[i] * mul, 
                                                       regions[i-1].v_end_ev + band[i] - band[i-1], mass[i]))
            stack = MatterAiryStack(regions)
            energy_levels = stack.get_eigen_energy_ev()
            # stack.plot()
            if potential_profile[mark] == None:
                potential_profile[mark] = stack.get_potential_profile()
            print "Energy levels for", mark, "\n".join([" %.3f eV" % e for e in energy_levels])
            if len(states) == 0:
                for energy in energy_levels:
                    states.append({mark + '_energy': energy})
            elif len(states) < len(energy_levels):
                for i in range(len(states)):
                    states[i][mark + '_energy'] = energy_levels[i]
                for i in range(len(energy_levels) - len(states)):
                    states.append({mark + '_energy': energy_levels[i]})
            else:
                for i in range(len(energy_levels)):
                    states[i][mark + '_energy'] = energy_levels[i]
            for i in range(len(energy_levels)):
                states[i][mark + '_wavefunction'] = stack.get_wavefunction(i)

        if el:
            mass = [lr.mass_elz for lr in self.layers]
            get_the_values(band_e, mass, 'el')
        if hh:
            mass = [lr.mass_hhz for lr in self.layers]
            get_the_values(band_h, mass, 'hh')
        if lh:
            mass = [lr.mass_lhz for lr in self.layers]
            get_the_values(band_h, mass, 'lh')

        self.states = states
        self.potential_profile = potential_profile
        return states, potential_profile

    def get_wavefunction_airy(self, index, mark='el'):
        '''get_wavefunction_airy(index, mark):

        Returns wavefunction of the indexed state (ground state corresponds to zero) for the selected
        particle which can be 'el' (default), 'hh' or 'lh' '''
        self.field = self.get_electric_field()
        band_e, band_h = self.get_bands()
        if mark == 'lh':
            mul = -1 # Field correction for holes
            band = band_h
            mass = [lr.mass_lhz for lr in self.layers]
        elif mark == 'lh':
            mul = -1
            band = band_h
            mass = [lr.mass_hhz for lr in self.layers]            
        else:
            mul = 1
            band = band_e
            mass = [lr.mass_elz for lr in self.layers]
            
        regions = [MatterUniformFieldInterval(a2cm(self.layers[0].width), self.field[0] * mul, band[0], mass[0])]
        for i in range(1, len(self)):
            regions.append(MatterUniformFieldInterval(a2cm(self.layers[i].width), self.field[i] * mul, 
                                                       regions[i-1].v_end_ev + band[i] - band[i-1], mass[i]))
        stack = MatterAiryStack(regions)
        return stack.get_wavefunction(index)

    def get_overlap(self, index1, index2, htype='hh'):
        '''Overlap integral calculated for index1 of electron wavefunction and index2 hole's one'''
        (xe, ye) = self.get_wavefunction_airy(index1, mark='el')
        (xh, yh) = self.get_wavefunction_airy(index2, mark=htype)
        return simps(ye*yh, xe) # Simpson method from SciPy


    def plot_wavefunctions(self, nwf=None, el=True, lh=True, hh=False, 
                                 xmin=None, xmax=None, ymin=None, ymax=None,
                                 wfmul=1, save=False, show=True, reverseholes=False):
        toNm = 1e7
        states, pprofile = self.get_wavefunctions_airy(el, lh, hh)
        fig, (axe, axh) = pl.subplots(2, 1, sharex=True)
        fig.patch.set_facecolor('white') # White figure background
        band_ex, band_ey = pprofile['el']
        band_hx, band_hy = pprofile['lh']

        evmin = min(band_ey)
        evmax = max(band_ey)
        hvmin = min(band_hy)
        hvmax = max(band_hy)
        
        def apply_plot_params(mark):
            if mark == 'e':
                vmax, vmin = evmax, evmin
            else:
                vmax, vmin = hvmax, hvmin
            dv = hvmax - hvmin
            vshift = dv * 0.1
            if xmin != None:
                pl.xlim(xmin = xmin)
            if xmax != None:
                pl.xlim(xmax = np.floor(np.max(band_ex)))
            if ymin != None:
                pl.ylim(ymin = ymin)
            else: 
                pl.ylim(ymin = vmin - vshift)
            if ymax != None: 
                pl.ylim(ymax = ymax)
            else: 
                pl.ylim(ymax = vmax + vshift)

        axe.plot(band_ex * toNm, band_ey, lw=3, color='0.65')
        apply_plot_params('e')
        axe.set_autoscaley_on(False)
        
        axh.plot(band_hx * toNm, band_hy, lw=3, color='0.65')
        apply_plot_params('h')
        axh.set_autoscaley_on(False)


        if len(states) < nwf or nwf == None:
            nwf = len(states)

        colors = ['r', 'y', 'b', 'g', 'm']

        def plot_wavefunction(mark, i):
            if mark in ['hh', 'lh']:
                ax = axh
                vmax, vmin = hvmax, hvmin
            else:
                ax = axe
                vmax, vmin = evmax, evmin
            wflim = (hvmax - hvmin)

            (x, y) = states[i][mark + '_wavefunction']
            x *= toNm
            y = self.reduce_wavefunction(y, wflim / nwf)
            y = y * wfmul + states[i][mark + '_energy']
            
            xlvl = [x[0]]*2
            ylvl = [states[i][mark + '_energy']]*2

            ax.plot(xlvl, ylvl, colors[i], linewidth=2, alpha=0.65)
            ax.plot(x, y,       colors[i], linewidth=3) 

            # p = pl.fill_between(ax, x, ylvl, y, facecolor=colors[i])
            # p.set_alpha(0.2)

        for i in range(nwf):
            if i == len(colors) - 1:
                colors = colors * 2

            if states[i].has_key('el_energy'):
                plot_wavefunction('el', i)
            if states[i].has_key('lh_energy'):
                plot_wavefunction('lh', i)
            if states[i].has_key('hh_energy'):
                plot_wavefunction('hh', i)

        axe.spines['bottom'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.spines['left'].set_visible(False)
        axe.xaxis.tick_bottom()
        axe.tick_params(labelbottom=False, labeltop=True, labelleft=False, labelright=False, 
                        bottom=False, top=True, left=False, right=False)
        axh.spines['top'].set_visible(False)
        axh.spines['right'].set_visible(False)
        axh.spines['left'].set_visible(False)
        axh.xaxis.tick_top()
        axh.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                        bottom=True, top=False, left=False, right=False) 
        # axh.set_xticklabels('', fontdict={'size': 'xx-large'}, minor=False)
        pl.subplots_adjust(hspace=0.1)
        if save:
            filename = "qw__" + "__".join([str(int(lr.x * 1e2)) + '-' + str(lr.width) for lr in self.layers])
            pl.savefig(filename + '.png', bbox_inches=0)
        if show:
            pl.show()

    def reduce_wavefunction(self, y, value, dst=5):
        extrm = []
        for i in range(dst, len(y) - dst):
            if min(y[i-dst], y[i+dst]) < y[i] < max(y[i-dst], y[i+dst]) and \
               (y[i-dst+1]-y[i-dst]) * (y[i+dst]-y[i+dst-1]) < 0:
                extrm.append(y[i])
        return y * value / np.max(np.abs(extrm)) # avoiding exponential tails







if __name__ == '__main__':
    
    T = 20

    # config = lambda x, xb, d: [       AlGaNLayer(xb,  70.,      T),
    #                                   AlGaNLayer(x,   d,        T),
    #                                   AlGaNLayer(xb,  70.,      T)]

    # config2 = lambda x, xb, d1, d2: [ AlGaNLayer(xb,           70.,      T),
    #                                   AlGaNLayer(x,            d1,        T),
    #                                   AlGaNLayer((x + xb)/2,   d2,        T),
    #                                   AlGaNLayer(xb,           70.,      T)]
    
    # drange = np.arange(0.5, 8, 0.5)
    # rrange = np.arange(0.1, 3, 0.2)
    

    qwc = 0.55 # Quantum well compound
    brc = 0.65 # Barrier compound


    layers = [
        AlGaNLayer(0.60, 20 * 10,   T),
        AlGaNLayer(0.50, 2.25 * 10, T),
        AlGaNLayer(0.60, 20 * 10,   T),
    ]

    hs = Heterostructure(layers)
    overlap = hs.get_overlap(0,0)
    print "Overlap:", overlap
    print "Electric field: ", "  ".join(["%.2e" % f for f in hs.get_electric_field()]), "V/cm"
    hs.plot_wavefunctions(nwf=1)




    # 65/85 % w/o stairs
    # overlaps = np.array([1.2485920617491499, 
    #                      1.1394260877037126, 
    #                      1.0668562127264256, 
    #                      1.0308980056404338, 
    #                      1.0103408277377552, 
    #                      1.002321565347539, 
    #                      np.nan,
    #                      0.99645979881002855, 
    #                      0.99612121248694607, 
    #                      0.99551649460508684, 
    #                      0.99490678457028658, 
    #                      np.nan, 
    #                      0.99501194928206294, 
    #                      0.99639379417098772, 
    #                      0.99911002776899471])


    # 65/85 % w/ stairs
    # ol = np.matrix([[ 1.24859206 , 1.13942609 , 1.06685621 , 1.03089801 , 1.01034083 , 1.00232157 , np.nan , 0.9964598  , 0.99612121 , 0.99551649 , 0.99490678         , np.nan, 0.99501195 , 0.99639379 , 0.99911003],
    #      [ 1.23694225 , 1.2226008  , 1.2213968  , 1.20483655 , 1.18715543 , 1.17568199, 1.16369785 , 1.1519852  , 1.13967091 , 1.12748144 , 1.11554863 , 1.11012154, 1.09863852 , 1.08759943 , 1.08246665],
    #      [ 1.12315715 , 1.1063313  , 1.08962722 , 1.0793459  , 1.06890881 , 1.05154352, 1.04650577 , 1.04112615 , 1.03543688 , 1.02454937 , 1.02271379 , 1.02032624, 1.01403019 , 1.01100491 , 1.00741294],
    #      [ 1.05715403 , 1.03665004 , 1.02916418 , 1.02567652 , 1.01450846 , 1.01327394, 1.01409607 , 1.00864207 , 1.0030291  , 1.00354928 , 1.00299051 , 0.99854343, 1.00154808 , 0.99834115 , 0.99764866],
    #      [ 1.01902878 , 1.0116768  , 1.01113082 , 1.00856083 , 1.00147351 , 1.00491039, 1.00146717 , 0.99998027 , 0.99726796 , 0.9990461  , 0.99916177 , 0.9972951, 0.99672798 , 0.99762726 , 0.99668805],
    #      [ 1.00558917 , 1.00475668 , 1.00216808 , 1.00315353, np.nan ,      1.00067267, 0.99841506 , 0.99701568 , 0.99896997 , 0.99835646 , 0.99803643 , 0.99795281, 0.99906953 , 0.99767627 , 0.99746773],
    #      [ 0.99733056 , 0.99728556 , 0.99349971 , 0.99527729 , 0.99496323 , 0.99855065, 0.99801874 , 0.99745542 , 0.99914639 , 0.99691621 , 0.99802863 , 0.99900644, 0.99717373 , 0.99904676 , 0.99844231],
    #      [ 0.99385388 , 0.99734794 , 0.999059   , 0.99661873 , 0.99690135 , 0.99838948, 0.99791983 , 0.99923841 , 0.99889476 , 0.99811177 , 0.99930921 , 0.99720629, 0.99797133 , 0.99892452 , 0.99723058],
    #      [ 0.99543019 , 0.99409103, np.nan ,      0.99546745 , 0.99599072 , 0.99767162, 0.99714318, np.nan , 0.99764498 , 0.99958262 , 0.99769337 , 0.99783103, 0.9975721, np.nan , 0.99968545],
    #      [ 0.99560164 , 0.99552179 , 0.99814245 , 0.99839656 , 0.99760124 , 0.99796297, 0.99809205 , 0.99804043 , 0.99870668 , 0.99760149 , 0.99765002 , 0.99933912, 0.99775684 , 0.99842615 , 0.99899868],
    #      [ 0.99470131 , 0.9975306  , 0.99593532 , 0.99521196 , 0.99835412 , 0.9972932, 0.99907188 , 0.9976844  , 0.99784574 , 0.99838404 , 0.9998299  , 0.99768994, 0.99956421 , 0.9984734  , 0.99958523],
    #      [ 0.99327288 , 0.99104126 , 0.9940388  , 0.9974512  , 0.99905839 , 0.99839674, 0.99822658 , 0.99837029 , 0.99801327 , 0.99980445 , 0.99995352 , 0.99879559, 0.99893473 , 0.99843251 , 0.99983471],
    #      [ 0.9919165  , 0.99344179 , 0.99847396 , 0.99917996 , 0.99687736 , 0.99797386, 0.99914518 , 0.99872516 , 0.99948465 , 0.99810546 , 0.99897882 , 0.99852489, 0.99910145 , 0.99897398 , 0.99839968],
    #      [ 0.99437481 , 0.996198   , 0.99768955 , 0.99690017 , 0.99788145 , 0.99805137, 0.99958857 , 0.99879973 , 0.99861562 , 0.99990897 , 0.99945179 , 0.99947089, 0.99845048 , 0.99854402 , 0.99815868],
    #      [ 0.99471012 , 0.99240102 , 0.99790465 , 0.99541601 , 0.99769805 , 0.99902503, 0.99884646 , 0.99889548 , 0.99951566  , 0.99959221  , 0.99960414  , 0.9999352 , 0.99922043  , 0.99944948  , 0.99916886], 
    #      [ 0.99246604  , 0.99804535  , 0.99288384  , 0.99599971  , 0.99797097  , 0.9977592 , 1.          , 0.999235    , 0.99897268  , 0.99942246  , 0.99972978  , 0.99875633 , 0.99975098  , 0.99852736  , 0.99983043]]
    # )




    # # # # # # # # # # # # # # # # # AlGaN QW with stair 55% / 65%

    # drange = np.arange(0.5, 8, 0.25) # QW width
    # rrange = np.arange(0.1, 2, 0.1)  # stair/QW widths ratio


    # ol = np.matrix([
    # [ 1.23694225, 1.23320737, 1.2226008, 1.21843496,  1.2213968, 1.20979266, 1.20483655,  1.19966292,  1.18715543,  1.18152832,  1.17568199,  1.16976491, 1.16369785,  1.15816347,  1.1519852, 1.14584961,  1.13967091,  1.13357773, 1.12748144],
    # [1.17508291, 1.16478008, 1.15411494, 1.15851467, 1.14736248, 1.13608554, 1.12485158, 1.12164079, 1.11703042, 1.1059734, 1.10126305, 1.09792532, 1.08738218, 1.08275428, 1.07822603, 1.0683874, 1.07036009, 1.0659257, 1.05660526],
    # [1.12315715, 1.11522404, 1.1063313, 1.10538531, 1.08962722, 1.08051634, 1.0793459, 1.07023981, 1.06890881, 1.06007858, 1.05154352, 1.05004917, 1.04650577, 1.04474129, 1.04112615, 1.03331348, 1.03543688, 1.02784907, 1.02454937], 
    # [1.07922317, 1.07394319, np.nan, 1.06157134, 1.05637312, 1.04877846, 1.04375177, 1.04360667, 1.03597421, 1.03539609, 1.02795828, 1.02708921, 1.01996244, 1.02254186, 1.02091701, 1.01762812, 1.01557876, 1.01237057, 1.01337224], 
    # [1.05715403, 1.05371136, 1.03665004, 1.03294675, 1.02916418, 1.02979674, 1.02567652, 1.02554152, 1.01450846, 1.01406412, 1.01327394, 1.01217891, 1.01409607, 1.00722881, 1.00864207, 1.00637618, 1.0030291, 1.0035098, 1.00354928], 
    # [1.03958964, 1.03152183, 1.02350476, 1.02027295, 1.01663255, 1.01281054, 1.008874, 1.00832678, 1.00743857, np.nan, 1.0048355, 1.00315204, 1.00120454, 1.00195279, 1.00235507, 1.00238356, 1.00202793, 1.00124978, 1.00003966], 
    # [1.01902878, 1.01911723, 1.0116768, 1.00811105, 1.01113082, 1.00654118, 1.00856083, 1.00663982, 1.00147351, 0.99923133, 1.00491039, 1.0018627, 1.00146717, 1.0008521, 0.99998027, 0.99880276, 0.99726796, 0.99833423, 0.9990461 ], 
    # [1.01237381, 1.01205061, 1.01101662, 1.0023766, 1.00415529, 1.00527475, 0.99925731, 1.00233648, 0.99882161, 1.00069099, 0.9968329, 0.99999466, 0.99827583, 0.99985957, 1.00022878, 0.99757623, 0.99752163, 0.99881465, 0.99769552], 
    # [1.00558917, 0.99975619, 1.00475668, 1.00210704, 1.00216808, 1.00156278, 1.00315353, 0.99546977, np.nan, 0.99881586, 1.00067267, 0.99774608, 0.99841506, np.nan, 0.99701568, 0.99781868, 0.99896997, 0.99689237, 0.99835646], 
    # [0.99966283, 1.00425301, 1.00038114, 0.99573023, 1.00060876, 0.99793489, 0.99504281, 0.99434461, 0.99563587, 0.99413577, 0.99687287, 0.99660296, 0.99810908, 0.99883007, 0.99863855, 0.99594405, 0.99702128, 0.99715762, 0.9987887 ], 
    # [0.99733056, 0.99593516, 0.99728556, 1.00068871, 0.99349971, 0.9979288, 0.99527729, 0.99818969, 0.99496323, np.nan, 0.99855065, 0.99853503, 0.99801874, 0.99681229, 0.99745542, 0.9972958, 0.99914639, 0.99720584, 0.99691621], 
    # [0.99973373, 0.99631987, 0.99509134, 0.9930293, 0.99314916, 0.99502976, 0.99625769, 0.99894794, 0.99594203, 0.99732085, 0.99616937, 0.99774942, 0.99797034, 0.99775929, 0.9981034, 0.99875613, 0.99863632, 0.99765134, 0.99874396], 
    # [0.99385388, 0.99435005, 0.99734794, 0.99253078, 0.999059, 0.99583076, 0.99661873, 0.99701419, 0.99690135, 0.9981239, 0.99838948, 0.99721861, 0.99791983, 0.9983273, 0.99923841, 0.99807677, 0.99889476, 0.99886204, 0.99811177], 
    # [0.99887459, 0.99695463, 0.9969983, 0.99878117, 0.99308329, 0.99604428, 0.99437382, 0.99789314, 0.99749054, 0.99674953, 0.99745204, 0.99727552, 0.99831162, 0.99703084, 0.99860551, 0.99893853, 0.99772575, 0.9984128, 0.99833996], 
    # [0.99543019, 0.9972161,  0.99409103, 0.99284982, np.nan, 0.99368502, 0.99546745, 0.99836579, 0.99599072, 0.99904582, 0.99767162, 0.99783658, 0.99714318, 0.99774622, 0.99979719, 0.99809885, 0.99764498, 0.99916825, 0.99958262], 
    # [0.99649919, 0.99513536, 0.99596132, 0.99157553, 0.99597109, 0.99516597, 0.9943716, 0.99503411, 0.99680058, 0.99944317, 0.99836339, 0.99699293, 0.99828495, 0.99937949, 0.99950102, 0.99848872, 0.99893982, 0.99821491, 0.99926818], 
    # [0.99560164, 0.99797795, 0.99552179, 0.99706352, 0.99814245, 0.99444348, 0.99839656, 0.99805133, 0.99760124, 0.99830725, 0.99796297, 0.99876169, 0.99809205, 0.99736066, 0.99804043, 0.99767864, 0.99870668, 0.99867289, 0.99760149], 
    # [0.99277213, 0.99141578, 0.9950658, 0.9912423, 0.99807444, 0.99935982, 0.99497978, 0.99760715, 0.99683165, 0.99596519, 0.99806711, 0.99718905, 0.99904924, 0.9990222, 0.99834942, 0.99846928, 0.99752437, 0.9979865, 0.99979888], 
    # [0.99470131, np.nan, 0.9975306, 0.99221858, 0.99593532, 0.99466687, 0.99521196, 0.99712331, 0.99835412, 0.99705215, 0.9972932, 0.99801193, 0.99907188, 0.99797546, 0.9976844, 0.99823966, 0.99784574, 0.99861386, 0.99838404], 
    # [0.99486,    0.99645805, 0.99257125, 0.99824781, 1.00000579, 0.9957033, 0.99382724, 0.99824489, 0.99856707, 0.99698059, 0.99920924, 0.99858049, 0.99902048, 0.99828755, 0.99841039, 0.99952584, 0.99943133, 0.99832232, 0.99841867], 
    # [0.99327288, 0.99841448, 0.99104126, 0.9951355, 0.9940388, 0.99499468, 0.9974512, 0.99501079, 0.99905839, np.nan, 0.99839674, 0.99947714, 0.99822658, 0.99786636, 0.99837029, 0.99983688, 0.99801327, 0.9993883, 0.99980445], 
    # [0.99685169, 0.99107608, 0.99693198, 0.99735396, 0.99312319, 0.9987183, 0.99841043, 0.9961338, 0.99845496, 0.99800654, 0.99852015, 0.99960689, 0.9987297, 0.99909755, 0.99987145, 0.99953145, 0.99807879, 0.99966969, 0.99821254], 
    # [0.9919165,  0.99168069, 0.99344179, 0.99782815, 0.99847396, 0.99488316, 0.99917996, 0.99921447, 0.99687736, 0.99912538, 0.99797386, 0.99911964, 0.99914518, 0.99937445, 0.99872516, 0.99865658, 0.99948465, 0.99928902, 0.99810546], 
    # [0.99558042, 0.99580258, 0.99567129, 0.99785134, 0.99547039, 0.99571565, 0.99864923, 0.9969236, 0.99777489, 0.99944378, 0.99831888, 0.99815075, 0.9985271, 0.99938444, 0.9989987, 0.99918697, 0.99835102, 0.99836809, 0.99931899], 
    # [0.99437481, 0.99282825, 0.996198, 0.99498653, 0.99768955, 0.99612215, 0.99690017, 0.99901643, 0.99788145, 0.99906713, 0.99805137, 0.99914735, 0.99958857, 0.99894848, 0.99879973, 0.99920086, 0.99861562, 0.99883065, 0.99990897], 
    # [0.9917177,  0.99377927, 0.99508149, 0.99796451, 0.99848146, 0.99435495, 0.99939703, 0.99565513, 0.99821067, 0.99807274, 0.99875592, 0.99963599, 0.99863161, 0.99975073, 0.99985167, 0.99885512, 0.99846021, 0.99879747, 0.99996145], 
    # [0.99471012, 0.99514907, 0.99240102, 0.99313937, 0.99790465, 0.99835996, 0.99541601, 0.99577757, 0.99769805, 1.00000001, 0.99902503, 0.99882424, 0.99884646, 0.99862753, 0.99889548, 0.99969403, 0.99951566, np.nan, 0.99959221], 
    # [0.99939346, 0.99494051, 0.99590444, 0.99329802, 0.99601862, 0.99489234, 0.99651504, 0.99500452, 0.99634679, 0.99783711, 0.99835133, 0.9985198, 0.99871628, 0.99893703, 0.99927617, 0.99870165, 0.99864692, 0.999183, 0.99891485], 
    # [0.99246604, 0.9915068,  0.99804535, 0.99298841, 0.99288384, 0.99600132, 0.99599971, 0.99818585, 0.99797097, 0.99863906, 0.9977592, 0.99890273, 1., 0.99873337, 0.999235, 0.99891462, 0.99897268, 0.99958376, 0.99942246], 
    # [0.99148983, 0.99618322, 0.9988446, 0.99883291, 0.99555699, 0.99668008,  0.99505424, 0.99627039, 0.99907423, 0.99893389, 0.99962182, 0.9990811, 0.99926212, 0.99971605, 0.9990604, 0.99886613, 0.99901045, 0.99966516, 0.99959322] ])

    
    # overlaps = [1.2485920617491499, 1.1934328846161741, 1.1394260877037126, 1.0994713109043879,
    #             1.0668562127264256, 1.0403241440052269, 1.0308980056404338, 1.0170582213763799,
    #             1.0103408277377552, 1.0037432798607497, 1.002321565347539, 0.99997347083174903,
    #             np.nan, 0.99717892288411736, 0.99645979881002855, 0.99972487771098351, 0.99612121248694607,
    #             0.99650762065644816, 0.99551649460508684, 0.99854815502970384, 0.99490678457028658,
    #             0.99536525612389859, np.nan, 0.99805209863638533, 0.99501194928206294, 0.99622189800805216,
    #             0.99639379417098772, 0.99556148832328628, 0.99911002776899471, 0.99633515207700374]



    # counter = len(drange)
    # for d in drange:
    #     hs = Heterostructure(config(0.55, 0.65, d*10))
    #     print " * " * 20, counter, "calculations left"
    #     counter -= 1
    #     overlaps.append(hs.get_overlap(0,0))

    # print "Overlaps w/o stairs", overlaps
    




    # count_all = np.size(drange) * np.size(rrange)
    # counter = 1
    # for d in drange:
    #     ol_r = []
    #     for r in rrange:
    #         hs = Heterostructure(config2(0.55, 0.65, d*10, d*10*r))
    #         print "*" * 20, "Entering ", counter, "/", count_all
    #         counter += 1
    #         ol_r.append(hs.get_overlap(0,0))
    #     ol.append(ol_r)

    # ol = np.array(ol)
    # print "OL STAIRS", ol
    
    # print "drange", len(drange)
    # print "rrange", len(rrange)
    # print "overlaps", len(overlaps)
    # print "ol size", np.size(ol)
    # print "ol", len(ol)
    # print "ol[:,1]", len(ol[:,1])
    # print "ol[1,:]", len(ol[1,:])

    # font = {'family' : 'normal', 'weight' : 'bold', 'size': 16}
    # mpl.rc('font', **font)
    # fig = pl.figure()
    # pl.plot(drange, overlaps, label=str(0), lw=2)
    # for i in range(0, len(rrange)):
    #     pl.plot(drange, ol[:,i][1:], label=str(rrange[i]), lw=2)
    # pl.legend()
    # pl.xlabel(u"Quantum well width, nm")
    # pl.ylabel(u"Overlap integral magnitude")
    # fig.patch.set_facecolor('white')
    # filename = 'Overlap ' + "_".join([str(i) for i in [qwc, brc]])
    # pl.savefig(filename + '.png', bbox_inches=0)
    # pl.show()



