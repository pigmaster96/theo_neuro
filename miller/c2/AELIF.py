import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

#AELIF uses an exponential component rather than an added artificial spike. Reset at V_max, the exponential component allows it to reach that value very quickly after threshold.


#AELIF Model
def AELIFneuron(E_l=-60e-3,V_th=-50e-3,V_r=-80e-3,delta_th=2e-3,C_m=100e-12,G_l=8e-9,a=10e-9,b=0.5e-9,tau_sra=50e-3,dt=1e-5,tmax=1.5,Iappval=400e-12,Iapp_start=0.5,V_max=50e-3,Iappvec=[]):

    #init time vector
    timevec=np.linspace(0,tmax,int(tmax/dt))

    #init membrane potential vector V_m
    V_m=np.zeros(timevec.shape)
    V_m[0]=E_l

    #init spike rate adaptation current I_sra
    I_sra=np.zeros(timevec.shape)
    I_sra[0]=0

    #applied current vector
    if len(Iappvec)==0:
        Iapp=np.zeros(timevec.shape)
        Iapp[int(Iapp_start/dt):int(timevec.shape[0]-Iapp_start/dt)]=Iappval
    else:
        Iapp=Iappvec

    #record spike times
    spikes=np.empty(0) 

    #simulate! (fwd euler)
    for t in trange(timevec.shape[0]-1):
        #check if max reached
        if V_m[t]>=V_max:
            #reset membrane voltage, set previous to V_max, record spike
            V_m[t-1]=V_max
            V_m[t]=V_r
            spikes=np.append(spikes,t*dt)
            #increment spike rate adaptation
            I_sra[t]+=b
        #fwd euler
        dv_dt=(G_l*(E_l-V_m[t]+delta_th*np.exp((V_m[t]-V_th)/delta_th))-I_sra[t]+Iapp[t])/C_m
        dI_sra_dt=(a*(V_m[t]-E_l)-I_sra[t])/tau_sra
        V_m[t+1]=V_m[t]+dv_dt*dt
        I_sra[t+1]=I_sra[t]+dI_sra_dt*dt

    return timevec,V_m,Iapp,I_sra,spikes


#timevec,V_m,Iapp,Isra=AELIFneuron()
#plt.figure()
#plt.subplot(3,1,1)
#plt.plot(timevec,V_m)
#
#plt.subplot(3,1,2)
#plt.plot(timevec,Iapp)
#
#plt.subplot(3,1,3)
#plt.plot(timevec,Isra)
#
#
#plt.show()



       









