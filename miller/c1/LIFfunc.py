#see LIF.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

def LIFneuron(dt=0.01,tmin=0,tmax=1000,
              E_l=-70,C_m=100,G_l=10,
              V_th=-50,V_r=-80,I_app=201):
    '''Leaky-Integrate-and-Fire neuron. Takes input parameters:
    E_l: Leak potential(mV)
    C_m: Membrane capacitance(pF)
    G_l: Leak conductance(nS)
    V_th: Threshold potential(mV)
    V_r: Reset potential(mV)
    I_app: Applied current magnitude for 3/5ths the simulation duration(pA)

    Returns time vector, Forward Euler numerical estimate of 
    membrane potential values, and applied current vector.'''

    t=np.arange(tmin,tmax,dt) #init time vector
    V=np.zeros(len(t)) #init results vector 
    Iappvec=np.zeros(len(t)) #init vector for applied current

    tau=C_m/G_l #time constant
    Iappvec[len(t)//5:len(t)-len(t)//5]=I_app #applied current magnitude
    V[0]=E_l #set initial potential to leak potential

    #fwd euler
    for i in trange(len(t)-1):
        dvdt=(E_l-V[i])/tau+Iappvec[i]/C_m
        V[i+1]=V[i]+dt*dvdt
        if V[i+1]>V_th:
            V[i+1]=V_r

    return t,V,Iappvec


#run simulations
t1,V1,I1=LIFneuron(I_app=180)
t2,V2,I2=LIFneuron(I_app=201)

#plot
fig=plt.figure()
plt.subplot(2,2,4)
plt.plot(t2,V2)
plt.xlabel("Time(ms)")
ymin,ymax=plt.ylim()

plt.subplot(2,2,3)
plt.plot(t1,V1)
plt.xlabel("Time(ms)")
plt.ylabel("$V_m(mV)$")
plt.ylim(ymin,ymax)

plt.subplot(2,2,2)
plt.plot(t2,I2)
ymin,ymax=plt.ylim()
plt.title("$0.21nA$")

plt.subplot(2,2,1)
plt.ylabel("$I_{app}(nA)$")
plt.ylim(ymin,ymax)
plt.title("$0.18nA$")

plt.show()