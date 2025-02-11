import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

class LIFneuron:
    '''
    Leaky-Integrate-and-Fire neuron. Takes input parameters:
    E_l: Leak potential(mV)
    C_m: Membrane capacitance(pF)
    G_l: Leak conductance(nS)
    V_th: Threshold potential(mV)
    V_r: Reset potential(mV)
    I_app: Applied current magnitude for 3/5ths the simulation duration(pA)
    sigma_i: Noise scaling---if applicable
    '''

    def __init__(self,E_l=-70,C_m=100,G_l=10,
                 V_th=-50,V_r=-80,I_app=201,sigma_i=0.1):
        self.E_l=E_l
        self.C_m=C_m
        self.G_l=G_l
        self.V_th=V_th
        self.V_r=V_r
        self.I_app=I_app
        self.sigma_i=sigma_i
        self.attrvec={'leak potential':self.E_l,'membrane capacitance':self.C_m,
                      'leak conductance':self.G_l,'threshold potential':self.V_th,
                      'reset potential':self.V_r,'applied current':self.I_app,
                      'noise scaling':self.sigma_i}
    

    def get(self,attrvec):
        '''
        Return requested properties, raises exception if undefined
        '''
        out={}
        for attr in attrvec:
            if attr in ['leak potential','membrane capacitance','leak conductance',
                        'threshold potential','reset potential','applied current','noise scaling']:
                out[attr]=(self.attrvec[attr])
            else:
                raise Exception("Unknown requested property")
        return out
    

    def simulate(self,tmin=0,tmax=1000,dt=0.01,unstim_prop=0.2,noise=False):
        '''
        Takes time limits in ms (tmin,tmax), increment(dt), and proportion of time vec to wait and stop
        before and after stimulating (unstim_prop) *must be between 0 and 1*
        

        Returns time vector, forward euler numerical estimate of membrane potential
        values, applied current vector, array of spike indices
        '''
        t=np.arange(tmin,tmax,dt) #init time vector
        V=np.zeros(len(t)) #init results vector
        Iappvec=np.zeros(len(t)) #init vector for applied current
        spikeind=[] #init array for spike indices

        tau=self.C_m/self.G_l #time constant
        unstim_time=round(len(t)*unstim_prop) #number of timesteps before and after stim

        Iappvec[unstim_time:len(t)-unstim_time]=self.I_app #applied current magnitude
        V[0]=self.E_l #set initial potential to leak potential

        #fwd euler
        for i in range(len(t)-1):
            dvdt=(self.E_l-V[i])/tau+Iappvec[i]/self.C_m
            if noise:
                V[i+1]=V[i]+dt*dvdt+self.sigma_i*random.gauss(0,1)*dt**0.5
            else:
                V[i+1]=V[i]+dt*dvdt

            if V[i+1]>self.V_th:
                spikeind.append(i+1)
                V[i+1]=self.V_r

        return t,V,Iappvec,spikeind
    


class LIFneuronV2(LIFneuron):
    '''LIF, but with different types of refractory period'''
    def __init__(self,E_l=-70,C_m=100,G_l=10,
                 V_th=-50,V_r=-80,I_app=201,sigma_i=0.1):
        self.E_l=E_l
        self.C_m=C_m
        self.G_l=G_l
        self.V_th=V_th
        self.V_r=V_r
        self.I_app=I_app
        self.sigma_i=sigma_i
        self.attrvec={'leak potential':self.E_l,'membrane capacitance':self.C_m,
                      'leak conductance':self.G_l,'threshold potential':self.V_th,
                      'reset potential':self.V_r,'applied current':self.I_app,
                      'noise scaling':self.sigma_i}

    def simulate(self,tmin=0,tmax=1000,dt=0.01,unstim_prop=0.1,
                 refractory_model='clamp',tref_0=10, #clamp refractory period
                 tau_th=3,deltath=20, #time constant for raised threshold decay and threshold increment
                 noise=False,SRA=False):
        '''
        Takes time limits in ms (tmin,tmax), increment(dt), and proportion of time vec to wait and stop
        before and after stimulating (unstim_prop) *must be between 0 and 1*   

        Returns time vector, forward euler numerical estimate of membrane potential
        values, applied current vector, array of spike indices

        V2: now incorporates requested model for refractory period
        '''
        models=['clamp','threshold','conductance',
                'threshold and conductance']
        if not refractory_model in models:
            raise Exception("Unknown refractory model")

        t=np.arange(tmin,tmax,dt) #init time vector
        V=np.zeros(len(t)) #init results vector
        Iappvec=np.zeros(len(t)) #init vector for applied current
        spikeind=[] #init array for spike indices

        tau=self.C_m/self.G_l #time constant
        unstim_time=round(len(t)*unstim_prop) #number of timesteps before and after stim

        Iappvec[unstim_time:len(t)-unstim_time]=self.I_app #applied current magnitude
        V[0]=self.E_l #set initial potential to leak potential

        #if clamp
        if refractory_model=='clamp':
            tref=tref_0
            refractory=False

            #fwd euler with flag for refractory period
            for i in range(len(t)-1):
                if refractory:
                    tref-=dt
                    V[i+1]=self.V_r
                    if tref<=0:
                        refractory=False
                else:
                    dvdt=(self.E_l-V[i])/tau+Iappvec[i]/self.C_m
                    if noise:
                        V[i+1]=V[i]+dt*dvdt+self.sigma_i*random.gauss(0,1)*dt**0.5
                    else:
                        V[i+1]=V[i]+dt*dvdt
                    #spike
                    if V[i+1]>self.V_th:
                        spikeind.append(i+1)
                        V[i+1]=self.V_r
                        tref=tref_0
                        refractory=True
            return t,V,Iappvec,spikeind


        #if threshold 
        if refractory_model=='threshold':
            V_thref=np.zeros(len(t)) #threshold potential vector
            V_thref[0]=self.V_th #set intiial raised threshold

            for i in range(len(t)-1):
                #fwd euler threshold potential
                dVthdt=(self.V_th-V_thref[i])/tau_th
                V_thref[i+1]=V_thref[i]+dVthdt*dt

                #fwd euler membrane potential
                dvdt=(self.E_l-V[i])/tau+Iappvec[i]/self.C_m
                if noise:
                    V[i+1]=V[i]+dt*dvdt+self.sigma_i*random.gauss(0,1)*dt**0.5
                else:
                    V[i+1]=V[i]+dt*dvdt

                #spike
                if V[i+1]>V_thref[i+1]:
                    spikeind.append(i+1)
                    V[i+1]=self.V_r
                    V_thref[i+1]+=deltath
            return  t,V,Iappvec,spikeind,V_thref
        
        #if conductance



















        



