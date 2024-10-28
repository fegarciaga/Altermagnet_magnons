import numpy as np
import matplotlib.pyplot as plt

def Init(Lx, Ly):
    """
    Initializes some random configuration of classical localized magnetic moments
    Lx, Ly: size of the finite size sample
    """
    return 2*np.pi*np.random.rand(Lx, Ly), np.pi*np.random.rand(Lx, Ly)

def Heisenberg_E(ThetaM, PhiM, J, i, j):
    """
    Computes the classical Heisenberg exchange energy between moments i and j
    """
    i1, j1 = i
    i2, j2 = j
    return J*(np.cos(ThetaM[i1,j1])*np.cos(ThetaM[i2,j2])+np.sin(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])*np.cos(PhiM[i1,j1]-PhiM[i2,j2]))

def DMI_E(ThetaM, PhiM, D, i, j):
    """
    Computes the DMI-associated energy
    """
    i1, j1 = i
    i2, j2 = j
    return D*np.sin(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])*np.sin(PhiM[i2,j2]-PhiM[i1,j1])

def Heisenberg_T(ThetaM, PhiM, J, i, j):
    """
    Computes the Heisenberg-associated torque
    """
    i1, j1 = i
    i2, j2 = j
    T1theta = J*(-np.sin(ThetaM[i1,j1])*np.cos(ThetaM[i2,j2])+np.cos(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])*np.cos(PhiM[i1,j1]-PhiM[i2,j2]))
    T2theta = J*(-np.cos(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])+np.sin(ThetaM[i1,j1])*np.cos(ThetaM[i2,j2])*np.cos(PhiM[i1,j1]-PhiM[i2,j2]))
    T1phi = -J*np.sin(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])*np.sin(PhiM[i1,j1]-PhiM[i2,j2])
    T2phi = J*np.sin(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])*np.sin(PhiM[i1,j1]-PhiM[i2,j2])
    return T1theta, T2theta,T1phi, T2phi

def DMI_T(ThetaM, PhiM, D, i, j):
    """
    Computes the DMI-associated torque
    """
    i1, j1 = i
    i2, j2 = j
    T1theta = D*np.cos(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])*np.sin(PhiM[i2,j2]-PhiM[i1,j1])
    T2theta = D*np.sin(ThetaM[i1,j1])*np.cos(ThetaM[i2,j2])*np.sin(PhiM[i2,j2]-PhiM[i1,j1])
    T1phi = -D*np.sin(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])*np.cos(PhiM[i2,j2]-PhiM[i1,j1])
    T2phi = D*np.sin(ThetaM[i1,j1])*np.sin(ThetaM[i2,j2])*np.cos(PhiM[i2,j2]-PhiM[i1,j1])
    return T1theta, T2theta, T1phi, T2phi

def Compute_Heis(ThetaM, PhiM, J, i, j, E, ThetaT, PhiT):
    E += Heisenberg_E(ThetaM, PhiM, J, i, j)
    a, b, c, d = Heisenberg_T(ThetaM, PhiM, J, i, j)
    i1, j1 = i
    i2, j2 = j
    ThetaT[i1,j1]+=a
    ThetaT[i2,j2]+=b
    PhiT[i1,j1]+=c
    PhiT[i2,j2]+=d
    return E, ThetaT, PhiT

def Compute_DMI(ThetaM, PhiM, D, i, j, E, ThetaT, PhiT):
    E += DMI_E(ThetaM, PhiM, D, i, j)
    a, b, c, d = DMI_T(ThetaM, PhiM, D, i, j)
    i1, j1 = i
    i2, j2 = j
    ThetaT[i1,j1]+=a
    ThetaT[i2,j2]+=b
    PhiT[i1,j1]+=c
    PhiT[i2,j2]+=d
    return E, ThetaT, PhiT

def Optimize(ThetaM, PhiM, J, D, alpha):
    J1, J2, delta = J
    Lx = ThetaM.shape[0]
    Ly = ThetaM.shape[1]
    ThetaT = np.zeros((Lx, Ly))
    PhiT = np.zeros((Lx, Ly))
    E = 0
    flag = 1
    for i in range(Lx):
        flag *=-1
        for j in range(Ly):
            flag *=-1
            if flag >0:
                JA = J2+delta
                JB = J2-delta
            else:
                JA = J2-delta
                JB = J2+delta

            # First the nearest neighbor coupling with antiferromagnetic coupling
            inew = (i+1)%Lx
            jnew = j

            E, ThetaT, PhiT = Compute_Heis(ThetaM, PhiM, J1, [i,j], [inew,jnew], E, ThetaT, PhiT)

            jnew = (j+1)%Ly
            # Altermagnetic coupling
            E, ThetaT, PhiT = Compute_Heis(ThetaM, PhiM, JA, [i,j], [inew,jnew], E, ThetaT, PhiT)
            # DMI coupling
            E, ThetaT, PhiT = Compute_DMI(ThetaM, PhiM, D, [i,j], [inew,jnew], E, ThetaT, PhiT)

            inew = (i-1)%Lx
            jnew = j

            E, ThetaT, PhiT = Compute_Heis(ThetaM, PhiM, J1, [i,j], [inew,jnew], E, ThetaT, PhiT)

            jnew = (j-1)%Ly
            # Altermangtic coupling
            E, ThetaT, PhiT = Compute_Heis(ThetaM, PhiM, JA, [i,j], [inew,jnew], E, ThetaT, PhiT)
            # DMI coupling
            E, ThetaT, PhiT = Compute_DMI(ThetaM, PhiM, D, [i,j], [inew,jnew], E, ThetaT, PhiT)

            inew = i
            jnew = (j+1)%Ly

            E, ThetaT, PhiT = Compute_Heis(ThetaM, PhiM, J1, [i,j], [inew,jnew], E, ThetaT, PhiT)

            inew = (i-1)%Lx
            # Altermagnetic coupling
            E, ThetaT, PhiT = Compute_Heis(ThetaM, PhiM, JB, [i,j], [inew,jnew], E, ThetaT, PhiT)
            # DMI coupling
            E, ThetaT, PhiT = Compute_DMI(ThetaM, PhiM, D, [i,j], [inew,jnew], E, ThetaT, PhiT)

            inew = i
            jnew = (j-1)%Ly

            E, ThetaT, PhiT = Compute_Heis(ThetaM, PhiM, J1, [i,j], [inew,jnew], E, ThetaT, PhiT)

            inew = (i+1)%Lx
            # Altermagnetic coupling
            E, ThetaT, PhiT = Compute_Heis(ThetaM, PhiM, J1, [i,j], [inew,jnew], E, ThetaT, PhiT)
            # DMI coupling
            E, ThetaT, PhiT = Compute_DMI(ThetaM, PhiM, D, [i,j], [inew,jnew], E, ThetaT, PhiT)

    # Now that all torques are computed, the angles are evolved
    ThetaM += -alpha*ThetaT
    PhiM += -alpha*PhiT
    return E

def Main(Lx, Ly, J, D, alpha, Nmax):
    ThetaM, PhiM = Init(Lx, Ly)
    J1, J2, delta = J
    E = 0
    for i in range(Nmax):
        E = Optimize(ThetaM, PhiM, J, D, alpha)
    np.savetxt(f"Classical_st/ThetaJ1_{J1}_J2_{J2}_delta_{delta}_D_{D}.txt", ThetaM)
    np.savetxt(f"Classical_st/PhiJ1_{J1}_J2_{J2}_delta_{delta}_D_{D}.txt", PhiM)
    return


Main(10, 10, [1,0.,0.], 0.1, 0.01, 7000)

