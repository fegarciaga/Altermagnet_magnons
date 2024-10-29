using ITensors
using ITensorMPS
using DelimitedFiles

# This computes the dynamical structure factor of an altermagnetic model running on a cilider with 4 sites on its transverse direction

function ITensors.space(::SiteType"2S")
    return 4
end

ITensors.val(::ValName"00", ::SiteType"2S") = 1
ITensors.val(::ValName"01", ::SiteType"2S") = 2
ITensors.val(::ValName"10", ::SiteType"2S") = 3
ITensors.val(::ValName"11", ::SiteType"2S") = 4

ITensors.state(::StateName"00", ::SiteType"2S") = [1.0, 0, 0, 0]
ITensors.state(::StateName"01", ::SiteType"2S") = [0, 1.0, 0, 0]
ITensors.state(::StateName"10", ::SiteType"2S") = [0, 0, 1.0, 0]
ITensors.state(::StateName"11", ::SiteType"2S") = [0, 0, 0, 1.0]

ITensors.op(::OpName"SZ1", ::SiteType"2S")=
 [0.5   0       0       0
  0     0.5     0       0
  0     0       -0.5    0
  0     0       0       -0.5]

ITensors.op(::OpName"SZ2", ::SiteType"2S")=
 [0.5   0       0       0
  0     -0.5    0       0
  0     0       0.5     0
  0     0       0       -0.5]

ITensors.op(::OpName"SP1", ::SiteType"2S")=
 [0     0       1       0
  0     0       0       1
  0     0       0       0
  0     0       0       0]

ITensors.op(::OpName"SP2", ::SiteType"2S")=
 [0     1       0       0
  0     0       0       0
  0     0       0       1
  0     0       0       0]

ITensors.op(::OpName"SM1", ::SiteType"2S")=
 [0     0       0       0
  0     0       0       0
  1     0       0       0
  0     1       0       0]


ITensors.op(::OpName"SM2", ::SiteType"2S")=
 [0     0       0       0
  1     0       0       0
  0     0       0       0
  0     0       1       0]

function Add_Heisenberg(os, J, any, i1, i2, ind1, ind2)
    szi1 = "SZ"*string(ind1)
    szi2 = "SZ"*string(ind2)
    spi1 = "SP"*string(ind1)
    spi2 = "SP"*string(ind2)
    smi1 = "SM"*string(ind1)
    smi2 = "SM"*string(ind2)
    os += any*J, szi1, i1, szi2, i2
    os += 0.5*J, spi1, i1, smi2, i2
    os += 0.5*J, smi1, i1, spi2, i2
    return os
end

function Add_NNN_DMI(os, D, i1, i2, ind1, ind2)
    spi1 = "SP"*string(ind1)
    spi2 = "SP"*string(ind2)
    smi1 = "SM"*string(ind1)
    smi2 = "SM"*string(ind2)
    os += -D/2*1im, smi1, i1, spi2, i2
    os += D/2*1im, spi1, i1, smi2, i2
    return os

function Add_NN_DMI(os, D, i1, i2, ind1, ind2, flag)
    szi1 = "SZ"*string(ind1)
    szi2 = "SZ"*string(ind2)
    spi1 = "SP"*string(ind1)
    spi2 = "SP"*string(ind2)
    smi1 = "SM"*string(ind1)
    smi2 = "SM"*string(ind2)
    if flag
        os += -D/2*1im, spi1, i1, szi2, i2
        os += D/2*1im, smi1, i1, szi2, i2
        os += D/2*1im, szi1, i1, spi2, i2
        os += -D/2*1im, szi1, i1, smi2, i2
    else
        os += D/2, szi1, i1, spi2, i2
        os += D/2, szi1, i1, smi2, i2
        os += -D/2, spi1, i1, szi2, i2
        os += -D/2, smi1, i1, szi2, i2
    end
    return os


function Build_H(n, J1, J2, delta, D1, D2, any)
    # Build_H construct the Hamiltonian as an MPO
    # n: number of cells
    # J1: nearest neighbor coupling
    # J2, J3: next nearest neighor couplings (needed for an altermagnetic model)
    os = OpSum()
    for i in 1:n
        os = Add_Heisenberg(os, J1, any, i, i, 1, 2)
        os = Add_NN_DMI(os, D1, i, i, 1, 2, false)
    end
    for i in 1:n
        if isodd(i)
            os = Add_Heisenberg(os, J1, any, i, i+1, 2, 1)
            os = Add_NN_DMI(os, D1, i, i+1, 2, 1, false)
        else
            # Periodic boundary conditions on the y direction are applied
            os = Add_Heisenberg(os, J1, any, i, i-1, 2, 1)
            os = Add_NN_DMI(os, D1, i, i-1, 2, 1, false)
        end
    end
    Nx = Int.(n/2)

    for i in 1:(Nx-1)
        os = Add_Heisenberg(os, J1, any, 2*i-1, 2*i+1, 1, 1)
        os = Add_NN_DMI(os, D1, 2*i-1, 2*i+1, 1, 1, true)

        os = Add_Heisenberg(os, J1, any, 2*i-1, 2*i+1, 2, 2)
        os = Add_NN_DMI(os, D1, 2*i-1, 2*i+1, 2, 2, true)
        
        os = Add_Heisenberg(os, J1, any, 2*i, 2*i+2, 1, 1)
        os = Add_NN_DMI(os, D1, 2*i, 2*i+2, 1, 1, true)

        os = Add_Heisenberg(os, J1, any, 2*i, 2*i+2, 2, 2)
        os = Add_NN_DMI(os, D1, 2*i, 2*i+2, 2, 2, true)
        # Now including the altermagnetic terms

        if isodd(i)
            JA = J2+delta
            JB = J2-delta
        else
            JA = J2-delta
            JB = J2+delta
        end
        
        os = Add_Heisenberg(os, JA, any, 2*i-1, 2*i+1, 1, 2)
        os = Add_NNN_DMI(os, D, 2*i-1, 2*i+1, 1, 2)

        os = Add_Heisenberg(os, JA, any, 2*i-1, 2*i+1, 2, 1)
        os = Add_NNN_DMI(os, D, 2*i-1, 2*i+1, 2, 1)
            
        os = Add_Heisenberg(os, JB, any, 2*i-1, 2*i+2, 2, 1)
        os = Add_NNN_DMI(os, D, 2*i-1, 2*i+2, 2, 1)
           
        os = Add_Heisenberg(os, JB, any, 2*i-1, 2*i+2, 1, 2)
        os = Add_NNN_DMI(os, D, 2*i-1, 2*i+2, 1, 2)

        os = Add_Heisenberg(os, JA, any, 2*i, 2*i+2, 1, 2)
        os = Add_NNN_DMI(os, D, 2*i, 2*i+2, 1, 2)

        os = Add_Heisenberg(os, JA, any, 2*i, 2*i+2, 2, 1)
        os = Add_NNN_DMI(os, D, 2*i, 2*i+2, 2, 1)

        os = Add_Heisenberg(os, JB, any, 2*i, 2*i+1, 1, 2)
        os = Add_NNN_DMI(os, D, 2*i, 2*i+1, 1, 2)

        os = Add_Heisenberg(os, JB, any, 2*i, 2*i+1, 2, 1)
        os = Add_NNN_DMI(os, D, 2*i, 2*i+1, 2, 1)
    end
    return os
end

function Apply_Sz(ϕ, s, idx, flag)
    # Apply_Sz applies a local operator to an MPS
    # ϕ: the MPS wavefunction
    # idx: index where the operator is located
    # flag: since the super cell has two spins, flag specifies which of the operators is used
    # s: siteinds ITensor structure
    if flag==0
        Szop = op("SZ1", s[idx])
    else
        Szop = op("SZ2", s[idx])
    end
    ϕ1 = copy(ϕ)
    orthogonalize!(ϕ1, idx)
    new_ϕ = Szop*ϕ1[idx]
    noprime!(new_ϕ)
    ϕ1[idx] = new_ϕ
    return ϕ1
end

n = 152
s = siteinds("2S",n)

state = ["00" for n in 1:n]
for i in 1:n
    state[i] = (isodd(i) ? "01" : "10")
end

let
    J1 = parse(Float64, ARGS[1])
    J2 = parse(Float64, ARGS[2])
    delta = parse(Float64, ARGS[3])
    D1 = parse(Float64, ARGS[4])
    D2 = parse(Float64, ARGS[5])
    any = parse(Float64, ARGS[6])
    println(J1,"\t", J2, "\t", delta)
    H = MPO(Build_H(n, J1, J2, delta, D1, D2, any), s)
    ψ = randomMPS(s, state, 10)
    sweeps = Sweeps(35)
    maxdim!(sweeps, 10, 10, 20, 20, 20, 20, 50, 50, 50, 50, 100, 100, 100, 100, 200, 200, 200, 200, 400, 500, 500, 500, 500, 600, 600, 600)
    noise!(sweeps, 1E-5, 1E-5, 1E-5, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-9)
    cutoff!(sweeps,1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-9, 1E-9, 1E-9, 1E-9, 1e-9, 1e-9, 1e-10, 1e-10, 1e-10, 1e-11)

    e2, ϕ2= dmrg(H, ψ, sweeps)
    nh = 76
    Nbond = 200
    SztSz0 = zeros(ComplexF64, 151,n*2)
    ϕp = Apply_Sz(ϕ2, s, nh, 0)
    for j in 1:n
        ϕa = Apply_Sz(ϕp, s, j, 0)
        SztSz0[1,j]=inner(ϕ2, ϕa)
        ϕa = Apply_Sz(ϕp, s, j, 1)
        SztSz0[1,j+n]=inner(ϕ2, ϕa)
    end
    for i in 1:150
        ϕp = tdvp(
             H,
             -0.15*1im,
             ϕp;
             time_step = -1im*0.15,
             normalize = false,
             maxdim = Nbond,
             mindim = 100,
             cutoff = 1e-6,
             outputlevel=1,
            )
        for j in 1:n
            ϕa = Apply_Sz(ϕp, s, j, 0)
            SztSz0[i+1,j]= exp(1im*e2*i*0.125)*inner(ϕ2, ϕa)
            ϕa = Apply_Sz(ϕp, s, j, 1)
            SztSz0[i+1,j+n] = exp(1im*e2*i*0.125)*inner(ϕ2, ϕa)
        end
        println(i,"/",120)
    end
    filename  = "results/Struct_r"*string(J2)*"_"*string(delta)*"_D1_"*string(D1)*"_D2_"*string(D2)*"_Ani_"*string(any)*"_Nbond_"*string(Nbond)*".txt"
    filename1 = "results/Struct_i"*string(J2)*"_"*string(delta)*"_D1_"*string(D1)*"_D2_"*string(D2)*"_Ani_"*string(any)*"_Nbond_"*string(Nbond)*".txt"
    writedlm(filename, real.(SztSz0))
    writedlm(filename1, imag.(SztSz0))
end


