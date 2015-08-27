#!/usr/bin/env python

from scipy import matrix

# Electric constant, Farad/m
e0 = 8.854e-12

class AlGaNLayer:
    """Simple Al(x)Ga(1-x)N properties implementation class"""

    def __init__(self, x, d=0):
        if 1 < x <= 100: 
            x = x / 100
        # Compound
        self.x = x
        # Layer width, A
        self.width = d
        # Energy gap, eV
        self.Eg = self.calc(3.42, 6.08, -1.1)
        # Lattice parameters, A
        self.a  = self.calc(3.18, 3.11)
        self.c  = self.calc(5.18, 4.98)
        # Static dielectric constant
        self.diel = self.calc(8.9, 8.5)
        # Piezoelectric constants, C/m**2
        self.pz13 = self.calc(-0.33, -0.58)
        self.pz33 = self.calc( 0.65, 1.55)
        # Spontaneous polarization, C/m**2
        self.Psp = self.calc(-0.029, -0.081)
        # Elastic constants, GPa
        self.c11 = self.calc(390, 410)
        self.c12 = self.calc(145, 149)
        self.c13 = self.calc(106,  99)
        self.c33 = self.calc(398, 389)
        self.c44 = self.calc(105, 125)
        # Coeffficients for lattice calculations
        self.G = 2*(self.c11+self.c12 - 2*(self.c13**2/self.c33))
        self.D = 2*(self.c12+2*(self.c13**2/self.c33))/(self.c11+2*self.c12)
        # Efficient masses, in units of free electron mass
        self.me   = self.calc(0.2,  0.4 )
        self.mhhx = self.calc(1.6, 10.42)
        self.mhhz = self.calc(1.1,  3.53)
        self.mlhx = self.calc(1.1,  0.24)
        self.mlhz = self.calc(0.15, 3.53)
        # References to control neighbour layers
        self.prevL = self
        self.nextL = self
        # Functions for alloy properties calculation
    def calc(self, a, b, c=0):
        return a * (1-self.x) + b * self.x - c * self.x * (1 - self.x)




def ev2nm(x):
    '''This function transforms wavelength to electronvolts and vice versa'''
    return 1239.842 / x


def pzqwfield(HS, verbose=False):
    '''Calculating electric field in multilayer heterostructure (HS)
    
    mlpzfield(HS, verbose=0)
        This function takes a HS tuple of several AlGaNLayer objects
        A verbose parameter sets what is shown in console output:
            <= 0 -- no output
             > 0 -- interim values
             > 1 -- more interim values, useful for debugging'''
        
    # TODO automatic detection of QW layer based on Eg minima --- what for?
    
    # Number of layers in heterostructure
    Lrs = len(HS)
    # Index range for the HS in order not to generate the same range() maby times
    Lrng = range(Lrs)
    # Setting references to neighbour layers
    for i in Lrng:
        if i > 0:     HS[i].prevL = HS[i-1] 
        if i < Lrs-1: HS[i].nextL = HS[i+1] 

    # Strained interface lattice peremeter, A 
    # FIXME calculation of interface lattice parameter should use the previous
    # interface's one.  The layers are not free-standing in the HS and
    # relaxation length of AlGaN is about hundreds of nm
    aInt = [(lambda L: 
        (L.prevL.a*L.prevL.G*L.prevL.width + L.G*L.width*L.a) /
                  (L.prevL.G*L.prevL.width + L.G*L.width))(L) 
                                                       for L in HS[1:]]

    # Strain tensor components needed to calculate polarization, no units
    epsxx = [(k==0 and 
             (lambda k: aInt[k]   / HS[k].a - 1) or 
             (lambda k: aInt[k-1] / HS[k].a - 1))(k) for k in Lrng]
    epszz = [(lambda j: -2*HS[j].c13 / HS[j].c33 * epsxx[j])(j) for j in Lrng]

    # Piezoelectric and total polarization, C/m**2
    Ppz  = [(lambda i: 
        2*HS[i].pz13*epsxx[i] + HS[i].pz33*epszz[i])(i) for i in Lrng]
    Ptot = [Ppz[k]+HS[k].Psp for k in Lrng ]

    # See [Smith, Mailhiot, J. Appl. Phys. 63, 2717, (1988), doi:]
    # Equations system in CGS:
    # D = eps*F = F + 4*pi*P
    # D1 = D2 => F1 - F2 = 4*pi*(P2 - P1)
    # Equations system in SI:
    # D = e0*eps*F + P
    # D1 = D2 => eps1*F1 - eps2*F2 = (P2 - P1) / e0
    # Thus coefficient (say K) is either 4*pi for CGS or e0**-1 for SI system
    # L1*eps1*F1 + L2*eps2*F2 + ... = 0
    # 
    # A * F = b
    # F = A \ b
    # Exapmle for 4 layers (e* are dielectric constants):
    # 
    #         | e1     -e2      0       0     |   |F1|     |P2 - P1|
    # A * F = | 0       e2     -e3      0     | * |F2|  =  |P3 - P2| * K
    #         | 0       0       e3     -e4    |   |F3|     |P4 - P3|
    #         | e1*L1   e2*L2   e3*L3   e4*L4 |   |F4|     |   0   |
    # 
    # Here we get V/cm for the SI system
    # F = A \ b / 1e2 
    A = matrix([[0]*Lrs]*Lrs)
    for i in Lrng[:-1]:
        A[i,i] = HS[i].diel
        A[i,i+1] = -HS[i+1].diel
    #   A[i,i] = 1
    #   A[i,i+1] = -1
    A[Lrs-1] = [HS[i].diel*HS[i].width for i in Lrng]
    P = e0**-1 * matrix([(int(k<Lrs-1) and Ptot[k+1]-Ptot[k]) 
                                       for k in Lrng]).transpose()

    # Electric field in HS layers, V/cm
    F = A**-1 * P / 1e2

    # Printing values of interim calculations
    if verbose:
        print 'Interface lattice constant a, A:'
        print '\t'.join(["%f" % v for v in aInt ]), '\n'
        print 'Strain tensor xx components:'
        print '\t'.join(["%f" % i for i in epsxx]), '\n'
        print 'Strain tensor zz components:'
        print '\t'.join(["%f" % i for i in epszz]), '\n'
        print 'Piezoelectric polarization, C/m**2:'
        print '\t'.join(["%f" % i for i in Ppz  ]), '\n'
        print 'Total polarization, C/m**2:'
        print '\t'.join(["%f" % i for i in Ptot ]), '\n'
        if verbose > 1:                                
            print 'Matrix of equations system:\n', A, '\n'
            print 'Right part vector, V/m:\n',     P, '\n'
        print 'Electric field, V/cm:\n',              \
                '\t'.join(["%.2e" % i for i in F]),    '\n'

    return F



if __name__ == '__main__':

#   bre = AlGaNLayer(0.77, 4000)
    br1 = AlGaNLayer(0.4,4250)
    qw  = AlGaNLayer(0.3,  60)
    br2 = AlGaNLayer(0.4, 750)

    HS = (br1, qw, br2)
#   HS = (bre, br1, qw, br2)
    pzqwfield(HS, verbose=1)

