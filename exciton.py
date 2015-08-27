import material
from math import sqrt

def primitive_cell_volume(a):
	return a**3 / 2**0.5
	
def dEdx(x,T,interval=2e-4):
	return Eg(x + interval/2,T) - Eg(x-interval/2,T) / interval
	
def excitonBohrRadius:
	return eps * h**2 / reducedMass / electronCharge**2

def Eg(x=0.5, T=300):
    return 4
    
def sigma(x,T):
	return 0.41 * dEdx(x,T) * sqrt(8*log(2)*x*(1-x)*primitiveCellVolume(x)
    
if __name__ == '__main__':
	print primitiveCellVolume(2)
	print sqrt(10)