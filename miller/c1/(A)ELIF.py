import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random
from LIFclass import LIFneuron

class LIFneuronSRA(LIFneuron):
    '''separate class with spike rate adaptation'''
    
    def __init__(self,E_l=-70,C_m=100,G_l=10,
                 V_th=-50,V_r=-80,I_app=201,sigma_i=0.1):
        super().__init__(E_l,C_m,G_l,
                 V_th,V_r,I_app,sigma_i)
        
    def simulate(self,tmin=0,tmax=1000,dt=0.01,unstim_prop=0.2,
                 tau_th=3,deltath=20, #time constant for raised threshold decay and threshold increment
                 kNernst=-80,tau_g=0.2,deltaG=2000, #reset value, conductance timeconstant, increment
                 noise=False):
        '''dynamic threshold and refractory conductance with option for SRA'''
        t=np.arange(tmin,tmax,dt) #init time vector
        V=np.zeros(len(t)) #init results vector
        Iappvec=np.zeros(len(t)) #init vector for applied current
        spikeind=[] #init array for spike indices

        tau=self.C_m/self.G_l #time constant
        unstim_time=round(len(t)*unstim_prop) #number of timesteps before and after stim

        Iappvec[unstim_time:len(t)-unstim_time]=self.I_app #applied current magnitude
        V[0]=self.E_l #set initial potential to leak potential

        #threshold model parameters
        V_thref=np.zeros(len(t)) #threshold potential vector
        V_thref[0]=self.V_th #set intiial raised threshold
        #conductance model parameters
        G_k=np.zeros(len(t)) #k conductance initially set to 0


        for i in range(len(t)-1):
            #fwd euler threshold potential
            dVthdt=(self.V_th-V_thref[i])/tau_th
            V_thref[i+1]=V_thref[i]+dVthdt*dt

            #fwd euler k conductance (decays to 0)
            dG_kdt=-G_k[i]/tau_g
            G_k[i+1]=G_k[i]+dG_kdt*dt

            #fwd euler membrane potential with ref conductance term
            dvdt=(self.E_l-V[i])/tau+\
                (Iappvec[i]+G_k[i]*(kNernst-V[i]))/self.C_m
            if noise:
                V[i+1]=V[i]+dt*dvdt+self.sigma_i*random.gauss(0,1)*dt**0.5
            else:
                V[i+1]=V[i]+dt*dvdt

            #spike
            if V[i+1]>V_thref[i+1]:
                spikeind.append(i+1)
                G_k[i+1]+=deltaG #increment G
                V_thref[i+1]+=deltath #increment threshold
        return t,V,Iappvec,spikeind,V_thref,G_k





    
test=LIFneuronSRA()
test1_1,test1_2,test1_3,test1_4,test1_5,test1_6=test.simulate()

plt.figure()
plt.plot(test1_1,test1_2)
plt.show()

