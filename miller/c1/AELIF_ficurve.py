import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

from AELIF import AELIFneuron


# TEST PLOT SINGLE AELIF
t,V,Iapp,_,spikes=AELIFneuron(E_l=-75e-3,V_th=-50e-3,V_r=-80e-3,delta_th=2e-3,G_l=10e-9,C_m=100e-12,a=2e-9,b=0.02e-9,tau_sra=200e-3,Iappval=300e-12,tmax=1.5,Iapp_start=0.5,dt=1e-5)


plt.figure()
plt.subplot(2,1,1)
plt.plot(t,V)
plt.ylabel('Vm')
plt.title('AELIF')

plt.subplot(2,1,2)
plt.plot(t,Iapp) #plt.ylabel('Iapp')
plt.xlabel('Time')

plt.show()


#F-I curve (inverse of first vs last ISI ('steady state'))
#currents to be tested
Iapps=np.linspace(150e-12,800e-12,40)

#record rates
initial_rate=np.zeros(Iapps.shape)
final_rate=np.zeros(Iapps.shape)

#run simulations
for trial in trange(len(Iapps)):
    #simulate neuron
    _,_,_,_,spikes=AELIFneuron(E_l=-75e-3,V_th=-50e-3,V_r=-80e-3,delta_th=2e-3,G_l=10e-9,C_m=100e-12,a=2e-9,b=0.02e-9,tau_sra=200e-3,Iappval=Iapps[trial],tmax=1.5,Iapp_start=0.5,dt=1e-5)
    #if no spike elicited
    if len(spikes)<=1:
        initial_rate[trial]=0
        final_rate[trial]=0
    #else record rates
    else:  
        initial_rate[trial]=1/(spikes[1]-spikes[0])
        final_rate[trial]=1/(spikes[-1]-spikes[-2])

#plot
plt.figure()
plt.plot(Iapps,initial_rate,marker='o',label='initial')
plt.plot(Iapps,final_rate,marker='x',label='final')
plt.ylabel("Firing rate (Hz)")
plt.xlabel("Applied current (A)")
plt.legend()

plt.show()











