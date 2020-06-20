# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#%%


def laplacian(a): #finds laplacians of the simulation region (uses/enforces periodic boundary conditions)
    l = -4*a
    l += np.roll(a, (0,-1), (0,1))
    l += np.roll(a, (0,+1), (0,1))
    l += np.roll(a, (-1,0), (0,1))
    l += np.roll(a, (+1,0), (0,1))
    return l


def F(U, V, f): #Gray-Scott model
    return -U*V**2 + f*(1 - U)

def G(U, V, f, k):
    return U*V**2 - (f + k)*V


def runge(U, V, dt, Du, Dv, f, k): #Runge-Kutta 4 method for evolving the simulation
    k1 = k2 = k3 = k4 = j1 = j2 = j3 = j4 = np.zeros(U.shape)
    
    k1 = dt*(F(U, V, f) + Du * laplacian(U))
    j1 = dt*(G(U, V, f, k) + Dv * laplacian(V))
    
    k2 = dt*(F(U+k1, V+j1, f) + Du * laplacian(U+k1))
    j2 = dt*(G(U+k1, V+j1, f, k) + Dv * laplacian(V+j1))
    
    k3 = dt*(F(U+k2, V+j2, f) + Du * laplacian(U+k2))
    j3 = dt*(G(U+k2, V+j2, f, k) + Dv * laplacian(V+j2))
    
    k4 = dt*(F(U+k3, V+j3, f) + Du * laplacian(U+k3))
    j4 = dt*(G(U+k3, V+j3, f, k) + Dv * laplacian(V+j3))
    
    U = U + k1/6 + k2/3 + k3/3 + k4/6
    V = V + j1/6 + j2/3 + j3/3 + j4/6
    
    return U, V


def initialise(x, y, xb, yb, Ub, Vb, Un, Vn): #takes initial conditions and creates the initial populations in a simulation region    
    U = np.ones((x, y)) #creates simulation region filled with U
    V = np.zeros((x, y))
    
    xb = int(xb/2) #creates a central box of V
    yb = int(yb/2)
    xlow = int(x/2) - xb
    xhigh = int(x/2) + xb
    ylow = int(y/2) - yb
    yhigh = int(y/2) + yb
    U[xlow:xhigh, ylow:yhigh] = Ub
    V[xlow:xhigh, ylow:yhigh] = Vb
    
    U += Un * np.random.random((x, y)) #adds noise
    V += Vn * np.random.random((x, y))
    
    return U, V


def run(U, V, dt, t, Du, Dv, f, k, fps, frames): #finally runs simulation from initial conditions, using population variables passed and outputs a gif
    plt.close("all")
    fig = plt.figure()
    plt.axis("off")
    
    tt = int(t/dt)
    ims = []
    dtpf = int(tt/frames)
    
    for i in range(0, tt):
        U, V = runge(U, V, dt, Du, Dv, f, k)
        if i % int(dtpf) == 0:
            im = plt.imshow(V)
            ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
    ani.save("DR.gif")
#%%    


U, V = initialise(150, 150, 20, 20, 0.5, 0.5, 0.1, 0.1)
run(U, V, 1, 12000, 0.07, 0.03, 0.035, 0.065, 10, 20)
