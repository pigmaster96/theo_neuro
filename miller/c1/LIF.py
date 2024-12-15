import numpy as np
import matplotlib.pyplot as plt

#LIF with fwd euler

#params
E_l=-70 #leak potential
C_m=100 #membrane conductance
G_l=10 #leak conductance
V_th=-50 #threshold potential
V_r=-80 #reset potential
tau= C_m/G_l #time constant

dt=0.01 #timestep
t=np.arange(0,1000,dt) #initialise time vector
V=np.zeros(len(t)) #initialise vector to store mem potential values

I_app=np.zeros(len(t)) #initialise applied current vector
I_app[len(t)//5:len(t)-len(t)//5]=201
V[0]=E_l #set initial condition to leak potential

#Forward euler
for i in range(len(t)-1):
    dvdt=(E_l-V[i])/tau+I_app[i]/C_m
    V[i+1]=V[i]+dt*dvdt
    if V[i+1]>V_th:
        V[i+1]=V_r

#plot
fig=plt.figure()
plt.plot(t,V)
plt.xlabel("time")
plt.ylabel("membrane potential")
plt.show()





