import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

#params
E_l=-70 #leak potential
C_m=100 #membrane conductance
G_l=10 #leak conductance
V_th=-50 #threshold potential
V_r=-80 #reset potential
tau= C_m/G_l*10 #time constant

dt=0.01 #timestep
t=np.arange(0,1000,dt) #initialise time vector
V=np.zeros(len(t)) #initialise vector to store mem potential values

I_app=np.zeros(len(t))
I_app[0]=1000 #instantaneously applied current

V[0]=E_l #set initial condition to leak potential

#Forward euler
for i in range(len(t)-1):
    dvdt=(E_l-V[i])/tau+I_app[i]/C_m
    V[i+1]=V[i]+dt*dvdt
    if V[i+1]>V_th:
        V[i+1]=V_r

#now compute fourier
partial_n=round(2/3*len(t))
omega=np.pi/len(t)

reconstruct=np.zeros(len(t))

#loop for each fourier coefficient
for i in trange(partial_n):
    tempsum=0
    for n in range(len(t)):
        add=V[n]*np.exp(-i*omega*n*1j)
        tempsum+=add    
    coeff=tempsum/len(t)
    #init specific harmonic

    



#plot
fig=plt.figure()
plt.plot(t,V)
plt.xlabel("time")
plt.ylabel("membrane potential")
plt.show()





