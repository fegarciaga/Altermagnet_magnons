import numpy as np

class H_params:
    J:float = 1
    J2:float = 0.1
    delta:float = 0.1
    Dz: float = 0.1
    Di: float = 0.1
    S: float = 0.5

delta1 = [1, 1]
delta2 = [-1, 1]
delta3 = [-1, -1]
delta4 = [1, -1]
deltax = [1, 0]
deltay = [0, 1]

def fa(k, H):
    f = 4*H.S*H.J
    J1 = H.J2+H.delta
    J1p = J.J2-H.delta
    f += 2*H.S*J1+2*H.S*J1*(np.cos(np.dot(k, delta1)))
    f += 2*H.S*J1p+2*H.S*J1p*np.cos(np.dot(k, delta2)))
    f += 2*H.S*H.Dz*(np.sin(np.dot(k, delta1))+np.sin(np.dot(k,delta2)))
    return f

def fb(k, H):
    f = 4*H.S*H.J
    J1 = H.J2+H.delta
    J1p = H.J2-H.delta
    f += 2*H.S*J1p*2*H.S*J1*(np.cos(np.dot(k, delta1)))
    f += 2*H.S*J1+2*H.S*J1*(np.cos(np.dot(k,delta2)))
    f += 2*H.S*H.Dz*(np.sin(np.dot(k, delta1))+np.sin(np.dot(k,delta2)))
    return f

def g(k, H):
    return 2*H.S*H.J*(np.cos(np.dot(k, deltax))+np.cos(np.dot(k, deltay)))

def vk(k, H):
    fab = fa(k, H) + fb(k, H)
    return np.sqrt(0.5*(-1+fab/np.sqrt(fab**2-4*g(k,H)**2)))

def uk(k, H):
    fab = fa(k, H) + fb(k, H)
    return np.sqrt(0.5*(1+fab/np.sqrt(fab**2-4*g(k, H)**2)))

def Ek(k, H):
    Deltaab = fa(k, H) - fb(k, H)
    fab = fa(k, H) + fb(k, H)
    wa = 0.5*Deltaab+0.5*np.sqrt(fab**2-4*g(k, H)**2)
    wb = -0.5*Deltaab+0.5*np.sqrt(fab**2-4*g(k, H)**2)
    return wa, wb

def Vk(k, H):
    return H.Di*np.sqrt(2*H.S)*(-np.sin(np.dot(k, delta1))+1j*np.sin(np.dot(k, delta2)))


