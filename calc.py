#!/usr/bin/env python

from material import *
from tm_airy import *

a2cm = lambda a: a * 1e-8

if __name__ == '__main__':
    
    bfLayer = AlGaNLayer(0.77,  5e3, 20)
    # b1Layer = AlGaNLayer(0.4, 4250., 20)
    b1Layer = AlGaNLayer(0.4, 120., 20)
    qwLayer = AlGaNLayer(0.3,   60., 20)
    b2Layer = AlGaNLayer(0.4,  120., 20)
    
    
    layers = [b1Layer, qwLayer, b2Layer]
    # [bfLayer, b1Layer, qwLayer, b2Layer]


    hs = Heterostructure(layers)
    field = hs.get_electric_field()
    print field

    regions = []
    for i in range(len(layers)):
        regions.append(MatterUniformFieldInterval(a2cm(layers[i].width), field[i], 
                                                       layers[i].energy_gap, layers[i].mass_elz))

    mstack = MatterAiryStack(*regions)

    print "\n"
    print "Eigen energy (eV):"
    print "\n".join(["%.4f eV" % e for e in mstack.get_eigen_energy_ev()])
    print "States lifetimes:"
    print "\n".join(["%.2e nsec" % t for t in mstack.get_lifetimes()])
    print "Number of eigen states:", len(mstack.eigen_energy)
    print 
    mstack.plot()

