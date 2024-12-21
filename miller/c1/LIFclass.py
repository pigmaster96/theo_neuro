import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

class LIFneuron:
    '''
    Leaky-Integrate-and-Fire neuron. Takes input parameters:
    E_l: Leak potential(mV)
    C_m: Membrane capacitance(pF)
    G_l: Leak conductance(nS)
    V_th: Threshold potential(mV)
    V_r: Reset potential(mV)
    I_app: Applied current magnitude for 3/5ths the simulation duration(pA)
    '''

    def __init__(self,E_l=-70,C_m=100,G_l=10,
                 V_th=-50,V_r=-80,I_app=201):
        self.E_l=E_l
        self.C_m=C_m
        self.G_l=G_l
        self.V_th=V_th
        self.V_r=V_r
        self.I_app=I_app
        self.attrvec={'leak potential':self.E_l,'membrane capacitance':self.C_m,
                      'leak conductance':self.G_l,'threshold potential':self.V_th,
                      'reset potential':self.V_r,'applied current':self.I_app}
    

    def get(self,attrvec):
        '''
        Return requested properties, raises exception if undefined
        '''
        out={}
        for attr in attrvec:
            if attr in ['leak potential','membrane capacitance','leak conductance',
                        'threshold potential','reset potential','applied current']:
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
            V[i+1]=V[i]+dt*dvdt
            if V[i+1]>self.V_th:
                spikeind.append(i+1)
                V[i+1]=self.V_r

        return t,V,Iappvec,spikeind