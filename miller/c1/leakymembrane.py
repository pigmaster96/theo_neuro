import numpy as np
import matplotlib.pyplot as plt

E_l=-70 #leak potential
V_0=-50 #initial condition
C_m=100 #membrane conductance
tau=0.5 #time constant
dt=0.01 #timestep

t=np.arange(0,5,dt) #vector for x axis (time)

v=np.zeros(len(t)) #initalise y axis (mem potential)


#plot
fig=plt.figure(figsize=(6,4))
plt.plot(t,E_l+(V_0-E_l)*np.exp(-t/tau),label="tau=0.5,V(0)=-50")

#comparing differences in tau and initial condition
plt.plot(t,E_l+(V_0-E_l)*np.exp(-t/1),label="tau=1,V(0)=-50")
plt.plot(t,E_l+(-80-E_l)*np.exp(-t/tau),label="tau=0.5,V(0)=-80")
plt.hlines(E_l,0,5,linestyles='dotted')
plt.xlabel("time")
plt.ylabel("membrane potential")
plt.legend()
plt.grid()
plt.show()



